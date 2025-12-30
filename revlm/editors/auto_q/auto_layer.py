"""AutoLayer: Find optimal layers for VLM embeddings using Vision Q and Language Q.

Usage:
    auto = AutoLayer(config, model)
    layers = auto.get_candidate_layers()
    best, scores = auto.find_best(dataset, layers, n_samples=10)
    auto.plot(scores)
"""

import random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from .modularity_core import ModularityCore
from ..utils import parent_module, brackets_to_periods, Augmenter


# Layers to exclude
EXCLUDE_PATTERNS = [
    "embed_tokens", "embeddings", "patch_embed", "patch_embedding",
    "class_embedding", "position_embedding", "pos_embed", "query_tokens",
    "layernorm", "layer_norm", "LayerNorm", "input_layernorm",
    "post_attention_layernorm", "pre_layrnorm", "post_layernorm",
    "q_norm", "k_norm", "norm1", "norm2", ".norm.",
    "lm_head", "crossattention",
    "q_proj", "k_proj", "v_proj", "qkv", "o_proj", "attn.proj",
    "rotary", "rope", "attention.attention",
    "qformer", "intermediate", "up_proj", "down_proj"
]


class AutoLayer(ModularityCore):
    """Score layers by Vision Q and Language Q.
    
    Scores computed:
    - Vision Q (entangled): <image, text> n×n pairs, cluster by image
    - Language Q (entangled): <image, text> n×n pairs, cluster by text
    - Pure Vision Q: <image, ""> with image augmentations
    - Pure Language Q: <blank, text> with text augmentations
    
    Edge filtering options:
        AutoLayer(..., edge_filter="percentile", edge_filter_kwargs={"percentile": 0.25})
        AutoLayer(..., edge_filter="knn", edge_filter_kwargs={"k": 10, "mutual": True})
        AutoLayer(..., edge_filter="disparity", edge_filter_kwargs={"alpha": 0.05})
    """

    def __init__(self, config, model, n_samples=100, n_aug=10, blank_image_size="match",
                 edge_filter="none", edge_filter_kwargs=None):
        """
        Args:
            config: Config object with device
            model: VLM wrapper
            n_samples: Number of samples for Q computation
            n_aug: Augmentations per sample for pure scores
            blank_image_size: Size for blank images in pure language Q.
                - "match": Match original image size (like old implementation)
                - tuple (W, H): Fixed size, e.g. (224, 224)
            edge_filter: Filter method - "none", "percentile", "knn", or "disparity"
            edge_filter_kwargs: Dict of kwargs for the filter method. Defaults:
                - percentile: {"percentile": 0.25}
                - knn: {"k": 10, "mutual": True}
                - disparity: {"alpha": 0.05, "pre_topk": 50}
        """
        super().__init__(getattr(config, "device", torch.device("cpu")))
        
        self.config = config
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.n_samples = n_samples
        self.n_aug = n_aug  # Augmentations per sample for pure scores
        self.blank_image_size = blank_image_size  # "match" or (W, H) tuple
        self.edge_filter = edge_filter
        self.edge_filter_kwargs = edge_filter_kwargs or {}
        
        self._hooks = []
        self._all_acts = {}
        self._images = None
        self._texts = None
        self._augmenter = None

    def get_candidate_layers(self, include_all=False):
        """Get candidate layer names."""
        layers = [n for n, _ in self.model.named_parameters() if n.endswith(".weight")]
        if not include_all:
            layers = [l for l in layers if not any(pat in l for pat in EXCLUDE_PATTERNS)]
        vis, merger, lang = self._classify_layers(layers)
        print(f"[AutoLayer] {len(layers)} layers (vision: {len(vis)}, merger: {len(merger)}, language: {len(lang)})")
        return layers

    def _classify_layers(self, layers):
        """Classify into vision, merger, language."""
        def is_merger(l): return any(p in l.lower() for p in ["multi_modal_projector", "merger", "language_projection"])
        def is_vision(l): return not is_merger(l) and any(p in l.lower() for p in ["vision", "visual", "qformer"]) and "language" not in l.lower()
        def is_lang(l): return not is_merger(l) and "language" in l.lower()
        return [l for l in layers if is_vision(l)], [l for l in layers if is_merger(l)], [l for l in layers if is_lang(l)]

    def _hook_all_layers(self, layer_names):
        """Register hooks on all layers."""
        self._remove_hooks()
        self._all_acts = {}
        
        for layer_name in layer_names:
            name = layer_name.rsplit(".", 1)[0] if layer_name.endswith((".weight", ".bias")) else layer_name
            try:
                mod = parent_module(self.model, brackets_to_periods(name))
                layer = getattr(mod, name.rsplit(".", 1)[-1])
                
                def make_hook(lname):
                    def hook_fn(m, inp, out):
                        act = inp[0].detach() if isinstance(inp[0], torch.Tensor) else out.detach()
                        self._all_acts[lname] = act
                    return hook_fn
                
                handle = layer.register_forward_hook(make_hook(layer_name))
                self._hooks.append(handle)
            except Exception:
                pass

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._all_acts = {}

    def _pool_act(self, act):
        """Pool to [1, hidden]."""
        if act is None:
            return None
        act = act.to(self.device, torch.float32)
        if act.dim() == 3:
            return act.mean(dim=1)
        elif act.dim() == 2:
            return act if act.shape[0] == 1 else act.mean(dim=0, keepdim=True)
        elif act.dim() == 1:
            return act.unsqueeze(0)
        elif act.dim() >= 4:
            return None
        return None

    @torch.no_grad()
    def _encode_all(self, image, text):
        """Single forward, return pooled acts for all hooked layers."""
        self.model.eval()
        self._all_acts = {}
        self.model(**self.wrapper.encode([image], [text], tokenize=False))
        
        result = {}
        for k, v in self._all_acts.items():
            pooled = self._pool_act(v)
            if pooled is not None:
                result[k] = pooled.cpu()
        self._all_acts = {}
        return result

    def _get_augmenter(self):
        """Lazy init augmenter."""
        if self._augmenter is None:
            self._augmenter = Augmenter(self.wrapper)
        return self._augmenter

    def _get_edge_filter_tuple(self):
        """Convert edge_filter config to tuple format for compute_Q."""
        if self.edge_filter == "none" or not self.edge_filter:
            return None
        elif self.edge_filter == "percentile":
            percentile = self.edge_filter_kwargs.get("percentile", 0.25)
            return ("percentile", percentile)
        elif self.edge_filter == "knn":
            k = self.edge_filter_kwargs.get("k", 10)
            mutual = self.edge_filter_kwargs.get("mutual", True)
            return ("knn", k, mutual)
        elif self.edge_filter == "disparity":
            alpha = self.edge_filter_kwargs.get("alpha", 0.05)
            return ("disparity", alpha)
        else:
            raise ValueError(f"Unknown edge_filter: {self.edge_filter}")

    @staticmethod
    def build_aug_target(n_samples, n_aug):
        """Target for augmentation: samples should cluster (anchor + augs together)."""
        group_size = 1 + n_aug
        N = n_samples * group_size
        labels = torch.arange(N) // group_size
        return (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    @torch.no_grad()
    def _encode_pure_vision(self, layers, n_aug, pbar=None):
        """Encode <image, ""> with image augmentations for pure vision Q."""
        all_embs = {l: [] for l in layers}
        
        for img in self._images:
            # Anchor: <image, "">
            embs = self._encode_all(img, "")
            for layer in layers:
                if layer in embs:
                    all_embs[layer].append(embs[layer])
            if pbar:
                pbar.update(1)
            
            # Augmentations
            augmenter = self._get_augmenter()
            for _ in range(n_aug):
                aug_img = augmenter.image(img)
                embs = self._encode_all(aug_img, "")
                for layer in layers:
                    if layer in embs:
                        all_embs[layer].append(embs[layer])
                if pbar:
                    pbar.update(1)
        
        return all_embs

    def _get_blank_image(self, idx=None):
        """Get blank image with configured size.
        
        Args:
            idx: Sample index (used when blank_image_size="match" to get original image size)
        """
        if self.blank_image_size == "match" and self._images and idx is not None:
            # Match original image size
            size = self._images[idx].size
        elif isinstance(self.blank_image_size, tuple):
            # Fixed size
            size = self.blank_image_size
        else:
            # Default fallback
            size = (224, 224)
        return Image.new("RGB", size, (128, 128, 128))

    @torch.no_grad()
    def _encode_pure_language(self, layers, n_aug, pbar=None):
        """Encode <blank, text> with text augmentations for pure language Q."""
        all_embs = {l: [] for l in layers}
        
        for idx, text in enumerate(self._texts):
            blank = self._get_blank_image(idx)
            
            # Anchor: <blank, text>
            embs = self._encode_all(blank, text)
            for layer in layers:
                if layer in embs:
                    all_embs[layer].append(embs[layer])
            if pbar:
                pbar.update(1)
            
            # Augmentations
            augmenter = self._get_augmenter()
            for _ in range(n_aug):
                aug_text = augmenter.question(text) if text else ""
                embs = self._encode_all(blank, aug_text)
                for layer in layers:
                    if layer in embs:
                        all_embs[layer].append(embs[layer])
                if pbar:
                    pbar.update(1)
        
        return all_embs

    @torch.no_grad()
    def find_best(self, dataset, layers, n_samples=None, n_aug=None, verbose=True):
        """Find best layers for vision and language.
        
        Computes 5 scores per layer:
        - vision_Q: entangled <img,text> n×n pairs, cluster by image
        - language_Q: entangled <img,text> n×n pairs, cluster by text
        - harmonic: harmonic mean of vision_Q and language_Q
        - pure_vision_Q: <image, ""> with image augs, cluster by sample
        - pure_language_Q: <blank, text> with text augs, cluster by sample
        
        Returns:
            best: Dict with best layers per category
            scores: Dict with {layer: {5 scores}}
        """
        n_samples = n_samples or self.n_samples
        n_aug = n_aug or self.n_aug
        
        # Sample data
        data = getattr(dataset, "data", dataset)
        samples = random.sample(list(data), min(n_samples, len(data)))
        n = len(samples)
        
        # Preload
        self._images = []
        self._texts = []
        for s in samples:
            img = s["image"]
            img = Image.open(img).convert("RGB") if isinstance(img, str) else img
            self._images.append(img)
            self._texts.append(s.get("question", ""))
        
        # Forward count
        n_entangled = n * n
        n_pure_per = n * (1 + n_aug)
        n_forwards = n_entangled + 2 * n_pure_per
        
        if verbose:
            print(f"[AutoLayer] {n} samples, {len(layers)} layers")
            print(f"            Entangled: {n}×{n} = {n_entangled} pairs")
            print(f"            Pure Vision: {n} × (1 + {n_aug}) = {n_pure_per}")
            print(f"            Pure Language: {n} × (1 + {n_aug}) = {n_pure_per}")
            print(f"            Total: {n_forwards} forwards")
        
        # Hook all layers
        self._hook_all_layers(layers)
        
        pbar = tqdm(total=n_forwards, desc="encoding") if verbose else None
        
        # 1. Encode entangled pairs: <img_i, text_j>
        entangled_embs = {l: [] for l in layers}
        for i in range(n):
            for j in range(n):
                embs = self._encode_all(self._images[i], self._texts[j])
                for layer in layers:
                    if layer in embs:
                        entangled_embs[layer].append(embs[layer])
                if pbar:
                    pbar.update(1)
        
        # 2. Encode pure vision: <image, ""> + image augs
        pure_vision_embs = self._encode_pure_vision(layers, n_aug, pbar)
        
        # 3. Encode pure language: <blank, text> + text augs
        pure_language_embs = self._encode_pure_language(layers, n_aug, pbar)
        
        if pbar:
            pbar.close()
        
        self._remove_hooks()
        
        # Compute Q for each layer
        scores = {}
        vis_layers, merger_layers, lang_layers = self._classify_layers(layers)
        aug_target = self.build_aug_target(n, n_aug).to(self.device)
        
        for layer in (tqdm(layers, desc="scoring") if verbose else layers):
            # Check minimum embeddings
            if len(entangled_embs[layer]) < n * n * 0.5:
                continue
            if len(pure_vision_embs[layer]) < n * (1 + n_aug) * 0.5:
                continue
            if len(pure_language_embs[layer]) < n * (1 + n_aug) * 0.5:
                continue
            
            # Get edge filter tuple
            edge_filter = self._get_edge_filter_tuple()
            
            # Entangled scores
            ent_embs = torch.cat(entangled_embs[layer], dim=0)
            ent_scores = self.compute_scores(ent_embs, n, edge_filter)
            
            # Pure vision score
            pv_embs = torch.cat(pure_vision_embs[layer], dim=0)
            pure_vis_Q = self.compute_Q(pv_embs, aug_target, edge_filter)
            
            # Pure language score
            pl_embs = torch.cat(pure_language_embs[layer], dim=0)
            pure_lang_Q = self.compute_Q(pl_embs, aug_target, edge_filter)
            
            scores[layer] = {
                "vision_Q": ent_scores["vision_Q"],
                "language_Q": ent_scores["language_Q"],
                "harmonic": ent_scores["harmonic"],
                "pure_vision_Q": pure_vis_Q,
                "pure_language_Q": pure_lang_Q,
            }
            
            if verbose:
                s = scores[layer]
                tqdm.write(f"  {layer[-45:]}: vis={s['vision_Q']:.3f}, lang={s['language_Q']:.3f}, "
                          f"H={s['harmonic']:.3f}, p_vis={s['pure_vision_Q']:.3f}, p_lang={s['pure_language_Q']:.3f}")
            
            torch.cuda.empty_cache()
        
        # Shift Q values by GLOBAL min (preserves relative relationship)
        if scores:
            # Global min across vision_Q and language_Q
            all_entangled = [s["vision_Q"] for s in scores.values()] + [s["language_Q"] for s in scores.values()]
            global_min = min(all_entangled)
            
            # Global min across pure scores
            all_pure = [s["pure_vision_Q"] for s in scores.values()] + [s["pure_language_Q"] for s in scores.values()]
            global_min_pure = min(all_pure)
            
            # Shift all scores by their respective global min
            for layer in scores:
                s = scores[layer]
                s["vision_Q_shifted"] = s["vision_Q"] - global_min
                s["language_Q_shifted"] = s["language_Q"] - global_min
                s["pure_vision_Q_shifted"] = s["pure_vision_Q"] - global_min_pure
                s["pure_language_Q_shifted"] = s["pure_language_Q"] - global_min_pure
                
                # Recompute harmonic on shifted values (now both >= 0)
                v, l = s["vision_Q_shifted"], s["language_Q_shifted"]
                s["harmonic_shifted"] = 2 * v * l / (v + l) if (v + l) > 0 else 0.0
            
            if verbose:
                print(f"\n[Shift] global_min={global_min:.4f}, global_min_pure={global_min_pure:.4f}")
        
        # Find best per category (use shifted harmonic for harmonic-based selection)
        def find_best_in(subset, key):
            valid = {l: scores[l] for l in subset if l in scores}
            return max(valid, key=lambda l: valid[l][key]) if valid else None
        
        def build_best_dict(key):
            """Build best dict for a metric: overall + per layer type."""
            return {
                "overall": find_best_in(scores.keys(), key),
                "vision_layer": find_best_in(vis_layers, key),
                "merger_layer": find_best_in(merger_layers, key),
                "language_layer": find_best_in(lang_layers, key),
            }
        
        best = {
            "vision_Q": build_best_dict("vision_Q"),
            "language_Q": build_best_dict("language_Q"),
            "harmonic": build_best_dict("harmonic_shifted"),
            "pure_vision_Q": build_best_dict("pure_vision_Q"),
            "pure_language_Q": build_best_dict("pure_language_Q"),
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print("Best layers per metric:")
            for metric_name, bests in best.items():
                print(f"  {metric_name}:")
                for group, layer in bests.items():
                    if layer:
                        val = scores[layer].get(metric_name, scores[layer].get("harmonic_shifted", 0))
                        print(f"    {group:15} → {layer} ({val:.3f})")
            print(f"{'='*70}")
        
        return best, scores

    # ==================== Save / Load / Aggregate ====================

    def _get_model_tag(self):
        """Get model tag for saving."""
        model_name = getattr(getattr(self.config, "model", None), "name", "unknown")
        return (model_name.split("/")[-1] or "model").replace(" ", "_")

    def save_results(self, best, scores, run_id=None, out_dir=None):
        """Save best layers and scores to JSON."""
        import json
        import os
        
        out_dir = out_dir or "results/auto_layer"
        model_tag = self._get_model_tag()
        os.makedirs(out_dir, exist_ok=True)
        
        suffix = f"_run{run_id}" if run_id is not None else ""
        out_path = os.path.join(out_dir, f"{model_tag}{suffix}.json")
        
        out_dict = {
            "model_tag": model_tag,
            "n_samples": self.n_samples,
            "n_aug": self.n_aug,
            "blank_image_size": self.blank_image_size if isinstance(self.blank_image_size, str) else list(self.blank_image_size),
            "run_id": run_id,
            "best": best,
            "scores": scores,
        }
        
        with open(out_path, "w") as f:
            json.dump(out_dict, f, indent=2)
        
        print(f"[AutoLayer] Saved to {out_path}")
        return out_path

    def load_results(self, run_id=None, out_dir=None):
        """Load single run results from JSON."""
        import json
        import os
        
        out_dir = out_dir or "results/auto_layer"
        model_tag = self._get_model_tag()
        suffix = f"_run{run_id}" if run_id is not None else ""
        in_path = os.path.join(out_dir, f"{model_tag}{suffix}.json")
        
        if not os.path.exists(in_path):
            print(f"[AutoLayer] No saved results at {in_path}")
            return None
        
        with open(in_path, "r") as f:
            data = json.load(f)
        
        print(f"[AutoLayer] Loaded from {in_path}")
        return data["best"], data["scores"]

    def load_results_k(self, out_dir=None):
        """Load all runs and aggregate into mean/std per layer per metric."""
        import json
        import os
        import glob
        
        out_dir = out_dir or "results/auto_layer"
        pattern = os.path.join(out_dir, f"{self._get_model_tag()}_run*.json")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"[AutoLayer] No run files found: {pattern}")
            return None
        
        print(f"[AutoLayer] Loading {len(files)} runs...")
        all_scores = []
        for f in files:
            with open(f) as fp:
                all_scores.append(json.load(fp)["scores"])
        
        # Get all metrics from first run
        sample_layer = list(all_scores[0].keys())[0]
        metrics = list(all_scores[0][sample_layer].keys())
        
        # Aggregate
        agg = {}
        for layer in all_scores[0]:
            agg[layer] = {}
            for m in metrics:
                vals = [s[layer][m] for s in all_scores if layer in s]
                agg[layer][m] = {"mean": np.mean(vals), "std": np.std(vals)}
        
        print(f"[AutoLayer] Aggregated {len(files)} runs, {len(agg)} layers")
        return agg

    def get_best_from_agg(self, agg_scores, metric="harmonic"):
        """Get best layers from aggregated scores."""
        layers = list(agg_scores.keys())
        vis_layers, merger_layers, lang_layers = self._classify_layers(layers)
        
        def find_best_in(subset, key):
            valid = {l: agg_scores[l] for l in subset if l in agg_scores}
            return max(valid, key=lambda l: valid[l][key]["mean"]) if valid else None
        
        def build_best_dict(key):
            """Build best dict for a metric: overall + per layer type."""
            return {
                "overall": find_best_in(layers, key),
                "vision_layer": find_best_in(vis_layers, key),
                "merger_layer": find_best_in(merger_layers, key),
                "language_layer": find_best_in(lang_layers, key),
            }
        
        # Use shifted versions if available
        sample_layer = layers[0]
        harmonic_key = "harmonic_shifted" if "harmonic_shifted" in agg_scores[sample_layer] else "harmonic"
        
        best = {
            "vision_Q": build_best_dict("vision_Q"),
            "language_Q": build_best_dict("language_Q"),
            "harmonic": build_best_dict(harmonic_key),
            "pure_vision_Q": build_best_dict("pure_vision_Q"),
            "pure_language_Q": build_best_dict("pure_language_Q"),
        }
        
        print(f"Best layers (from mean):")
        for metric_name, bests in best.items():
            print(f"  {metric_name}:")
            for group, layer in bests.items():
                if layer:
                    key = harmonic_key if metric_name == "harmonic" else metric_name
                    score = agg_scores[layer][key]["mean"]
                    std = agg_scores[layer][key]["std"]
                    print(f"    {group:15} → {layer} ({score:.3f}±{std:.3f})")
        
        return best

    # ==================== Plotting ====================

    def _is_aggregated(self, scores):
        """Check if scores are aggregated (have mean/std)."""
        sample_layer = list(scores.keys())[0]
        sample_val = scores[sample_layer]["vision_Q"]
        return isinstance(sample_val, dict) and "mean" in sample_val

    def plot(self, scores, figsize=(15, 6)):
        """Plot all Q scores vs layer index.
        
        Supports both single-run scores and aggregated scores (with error bars).
        Top row: Entangled scores (Vision Q, Language Q, Harmonic)
        Bottom row: Pure scores (Pure Vision Q, Pure Language Q)
        """
        import matplotlib.pyplot as plt
        
        layers = list(scores.keys())
        vis_layers, merger_layers, lang_layers = self._classify_layers(layers)
        is_agg = self._is_aggregated(scores)
        
        # Color by layer type
        colors = []
        for l in layers:
            if l in vis_layers:
                colors.append('green')
            elif l in merger_layers:
                colors.append('orange')
            elif l in lang_layers:
                colors.append('blue')
            else:
                colors.append('gray')
        
        indices = np.arange(len(layers))
        
        # Check if pure scores and shifted scores exist
        sample_layer = layers[0]
        has_pure = "pure_vision_Q" in scores[sample_layer]
        has_shifted = "harmonic_shifted" in scores[sample_layer]
        
        # Use shifted harmonic if available (more meaningful with negatives)
        harmonic_key = "harmonic_shifted" if has_shifted else "harmonic"
        
        if has_pure:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            
            plot_data = [
                (axes[0, 0], "vision_Q", 'Vision Q\n(<image, text>)'),
                (axes[0, 1], "language_Q", 'Language Q\n(<image, text>)'),
                (axes[0, 2], harmonic_key, 'Harmonic'),
                (axes[1, 0], "pure_vision_Q", 'Pure Vision Q\n(<image, "">)'),
                (axes[1, 1], "pure_language_Q", 'Pure Language Q\n(<blank, text>)'),
            ]
            axes[1, 2].axis('off')  # Empty subplot
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            plot_data = [
                (axes[0], "vision_Q", 'Vision Q'),
                (axes[1], "language_Q", 'Language Q'),
                (axes[2], harmonic_key, 'Harmonic'),
            ]
        
        for ax, key, title in plot_data:
            if is_agg:
                vals = np.array([scores[l][key]["mean"] for l in layers])
                stds = np.array([scores[l][key]["std"] for l in layers])
                # Plot with error bars per color group
                for color in ['green', 'orange', 'blue', 'gray']:
                    mask = np.array([c == color for c in colors])
                    if mask.any():
                        ax.errorbar(indices[mask], vals[mask], yerr=stds[mask], 
                                   fmt='o', ms=4, capsize=2, color=color, alpha=0.7)
            else:
                vals = np.array([scores[l][key] for l in layers])
                ax.scatter(indices, vals, c=colors, s=20, alpha=0.7)
            
            best_idx = np.argmax(vals)
            ax.scatter([best_idx], [vals[best_idx]], c='red', s=100, marker='*', zorder=5)
            ax.set_xlabel('Layer Index')
            ax.set_ylabel(f'{key} (↑)')
            ax.set_title(title)
            ax.grid(alpha=0.3)
        
        # Legend on first plot
        if has_pure:
            ax0 = axes[0, 0]
        else:
            ax0 = axes[0]
        ax0.scatter([], [], c='green', s=30, label='vision')
        ax0.scatter([], [], c='orange', s=30, label='merger')
        ax0.scatter([], [], c='blue', s=30, label='language')
        ax0.legend(fontsize=8)
        
        plt.tight_layout()
        plt.show()

    def cleanup(self):
        self._remove_hooks()
        self._images = None
        self._texts = None

