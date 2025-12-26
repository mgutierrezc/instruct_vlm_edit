"""
AutoLayer: Automatic layer selection for VLM embeddings.

Usage:
    auto = AutoLayer(config, model)
    layers = auto.get_candidate_layers()
    best, scores = auto.find_best(dataset, layers)
    auto.save_results(best, scores)
    auto.plot_scores(scores)
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from .utils import Augmenter, parent_module, brackets_to_periods

# Layers to exclude (variable outputs, not representations)
EXCLUDE_PATTERNS = [
    # Embeddings
    "embed_tokens", "embeddings", "patch_embed", "patch_embedding",
    "class_embedding", "position_embedding", "pos_embed", "query_tokens",
    # LayerNorm
    "layernorm", "layer_norm", "LayerNorm", "input_layernorm", 
    "post_attention_layernorm", "pre_layrnorm", "post_layernorm",
    "q_norm", "k_norm", "norm1", "norm2", ".norm.",
    # Output heads
    "lm_head",
    # Cross-attention
    "crossattention",
    # QKV projections
    "q_proj", "k_proj", "v_proj", "qkv",
    # Attention outputs
    "o_proj", "attn.proj",
    # Attention intermediates
    "rotary", "rope", "attention.attention",
    # # QFormer
    # "qformer", 
    "intermediate"
]


class AutoLayer:
    """Score layers by embedding robustness to augmentations."""

    def __init__(self, config, model):
        self.config = config
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))
        self.augmenter = Augmenter(self.wrapper)
        self._hooks = []
        self._all_acts = {}
        self._cache = {}
        self._samples = None
        self.n_samples = 20
        self.n_aug = 10
        self.use_percentile = False
        self.percentile_threshold = 0.0
        self.blank_image_for_lang = True   # Use <blank, text> for language robustness
        self.blank_text_for_vision = True  # Use <image, ""> for vision robustness

    def get_candidate_layers(self, include_all=False):
        layers = [n for n, p in self.model.named_parameters() if n.endswith(".weight")]
        if not include_all:
            layers = [l for l in layers if not any(pat in l for pat in EXCLUDE_PATTERNS)]
        vis = [l for l in layers if self._is_vision(l)]
        lang = [l for l in layers if self._is_language(l)]
        print(f"[AutoLayer] {len(layers)} candidate layers (vision: {len(vis)}, language: {len(lang)})")
        return layers

    def _hook_all_layers(self, layer_names):
        """Register hooks on ALL layers at once."""
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

    def _pool_act(self, act, layer_name=None):
        """Pool activation to [1, hidden_dim]."""
        if act is None:
            raise RuntimeError("Hook failed")
        act = act.to(self.device, torch.float32)
        
        if act.dim() == 3:
            return act.mean(dim=1)
        elif act.dim() == 2:
            if act.shape[0] == 1:
                return act
            elif act.shape[0] % 1 == 0:
                patches = act.shape[0]
                return act.view(1, patches, -1).mean(dim=1)
            else:
                return act.mean(dim=0, keepdim=True)
        elif act.dim() == 1:
            return act.unsqueeze(0)
        elif act.dim() >= 4:
            print(f"[AutoLayer] WARNING: Skipping 4D layer {layer_name}: {act.shape}")
            return None
        else:
            raise RuntimeError(f"Expected 1D/2D/3D activation, got {act.shape}")

    @torch.no_grad()
    def _encode_all(self, image, text):
        """Single forward pass, return pooled activations for ALL hooked layers."""
        self.model.eval()
        self._all_acts = {}
        self.model(**self.wrapper.encode([image], [text], tokenize=False))
        
        result = {}
        for k, v in self._all_acts.items():
            pooled = self._pool_act(v, layer_name=k)
            if pooled is not None:
                result[k] = pooled.cpu()
        self._all_acts = {}
        return result

    def _is_vision(self, layer_name):
        """Vision layers: vision tower + merger."""
        l = layer_name.lower()
        if "language" in l and not "language_projection" in l:
            return False
        return any(p in l for p in ["vision", "visual", "projector", "merger", "qformer", "language_projection"])

    def _is_language(self, layer_name):
        """Language layers: LLM backbone (excludes language_projection which is vision)."""
        l = layer_name.lower()
        return "language" in l and "language_projection" not in l

    # ==================== Metrics ====================

    def _compute_metrics(self, embs, n_samples, n_aug, use_percentile=False, percentile_threshold=0.0):
        """Compute metrics (all higher = better)."""
        group_size = 1 + n_aug
        N = len(embs)
        
        # Pairwise L2 distance -> similarity (0-1)
        l2_dist = torch.cdist(embs, embs, p=2)
        sim = 1 / (1 + l2_dist)
        
        # Upper triangle (exclude diagonal)
        triu_idx = torch.triu_indices(N, N, offset=1, device=embs.device)
        sim_triu = sim[triu_idx[0], triu_idx[1]]
        
        if use_percentile:
            # Convert to percentile ranks
            ranks = sim_triu.argsort().argsort().float()
            vals_triu = (ranks / max(len(sim_triu) - 1, 1))
            vals_triu = (vals_triu * 1000).round() / 1000
        else:
            # Use raw similarity directly
            vals_triu = sim_triu
        
        # Full symmetric matrix for log-likelihood
        sim_full = torch.zeros_like(sim)
        sim_full[triu_idx[0], triu_idx[1]] = vals_triu
        sim_full[triu_idx[1], triu_idx[0]] = vals_triu
        
        # Target (1 in-block, 0 off-block)
        labels = torch.arange(N, device=embs.device) // group_size
        target = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        target_triu = target[triu_idx[0], triu_idx[1]]
        
        # Apply threshold filter
        mask = vals_triu >= percentile_threshold
        vals_filtered = vals_triu * mask.float()
        
        # Compute scores
        mse_raw = ((vals_filtered - target_triu) ** 2).mean().item()
        mse = 1 / (1 + mse_raw)
        
        n_in_block = target_triu.sum().item()
        dot = (vals_filtered * target_triu).sum().item() / max(n_in_block, 1)
        
        log_probs = torch.log(sim_full + 1e-16)
        ll = (torch.sum(target * log_probs) / N).item()
        
        # Weighted modularity Q (uses raw similarity, not filtered)
        # Q = (1/2m) * sum_ij (w_ij - k_i*k_j/2m) * 1[z_i=z_j]
        k = sim.sum(dim=1)  # weighted degree per node
        m = sim.sum() / 2   # total weight
        null_model = torch.outer(k, k) / (2 * m)
        Q = ((sim - null_model) * target).sum() / (2 * m)
        modularity = Q.item()
        
        return {"mse": mse, "dot": dot, "ll": ll, "Q": modularity}

    # ==================== Public API ====================

    def _safe_append(self, lst, emb):
        """Append only if shape matches existing entries."""
        if len(lst) == 0 or lst[0].shape == emb.shape:
            lst.append(emb)

    @torch.no_grad()
    def score_layers(self, dataset, layers, n_samples=None, n_aug=None, verbose=True):
        """Score ALL layers in one pass."""
        n_samples = n_samples or self.n_samples
        n_aug = n_aug or self.n_aug
        
        if self._samples is None:
            data = getattr(dataset, "data", dataset)
            self._samples = random.sample(list(data), min(n_samples, len(data)))
        
        self._hook_all_layers(layers)
        all_embs = {l: {"vis": [], "lang": [], "both": []} for l in layers}
        
        n_forwards = n_samples * (1 + 3 * n_aug)  # anchor + vis_aug + lang_aug + both_aug
        if self.blank_image_for_lang:
            n_forwards += n_samples  # extra forwards for blank image anchors
        if self.blank_text_for_vision:
            n_forwards += n_samples  # extra forwards for blank text anchors
        pbar = tqdm(total=n_forwards, desc="encoding", disable=not verbose)
        
        for i, s in enumerate(self._samples[:n_samples]):
            if i > 0 and i % 10 == 0:
                torch.cuda.empty_cache()
            
            img = s["image"]
            img = Image.open(img).convert("RGB") if isinstance(img, str) else img
            text = s.get("question", "")
            
            # Blank image/text for isolated robustness testing
            blank_img = Image.new("RGB", img.size, (128, 128, 128)) if self.blank_image_for_lang else None
            lang_img = blank_img if self.blank_image_for_lang else img
            vis_text = "" if self.blank_text_for_vision else text
            
            # Anchor for "both" robustness (always <image, text>)
            embs = self._encode_all(img, text)
            for layer in layers:
                if layer in embs:
                    self._safe_append(all_embs[layer]["both"], embs[layer])
                    if not self.blank_text_for_vision:
                        self._safe_append(all_embs[layer]["vis"], embs[layer])
                    if not self.blank_image_for_lang:
                        self._safe_append(all_embs[layer]["lang"], embs[layer])
            pbar.update(1)
            
            # Vision anchor with blank text (if enabled)
            if self.blank_text_for_vision:
                embs = self._encode_all(img, vis_text)
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["vis"], embs[layer])
                pbar.update(1)
            
            # Language anchor with blank image (if enabled)
            if self.blank_image_for_lang:
                embs = self._encode_all(lang_img, text)
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["lang"], embs[layer])
                pbar.update(1)
            
            # Vision augmentations (may use blank text)
            for _ in range(n_aug):
                embs = self._encode_all(self.augmenter.image(img), vis_text)
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["vis"], embs[layer])
                pbar.update(1)
            
            # Language augmentations (may use blank image)
            for _ in range(n_aug):
                embs = self._encode_all(lang_img, self.augmenter.question(text) if text else "")
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["lang"], embs[layer])
                pbar.update(1)
            
            # Both augmentations (<aug(image), aug(text)>)
            for _ in range(n_aug):
                embs = self._encode_all(self.augmenter.image(img), self.augmenter.question(text) if text else "")
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["both"], embs[layer])
                pbar.update(1)
        
        pbar.close()
        self._remove_hooks()
        
        # Compute metrics
        expected = n_samples * (1 + n_aug)
        vis_scores, lang_scores, both_scores = {}, {}, {}
        
        for layer in (tqdm(layers, desc="metrics") if verbose else layers):
            vis_list, lang_list, both_list = all_embs[layer]["vis"], all_embs[layer]["lang"], all_embs[layer]["both"]
            
            if len(vis_list) < expected * 0.5 or len(lang_list) < expected * 0.5 or len(both_list) < expected * 0.5:
                if verbose:
                    tqdm.write(f"  Skipping {layer}: insufficient samples")
                continue
            
            vis_embs = torch.cat(vis_list, dim=0)
            lang_embs = torch.cat(lang_list, dim=0)
            both_embs = torch.cat(both_list, dim=0)
            
            self._cache[layer] = {
                "vision": (vis_embs, n_samples, n_aug),
                "language": (lang_embs, n_samples, n_aug),
                "both": (both_embs, n_samples, n_aug),
            }
            
            vis_scores[layer] = self._compute_metrics(vis_embs.to(self.device), n_samples, n_aug, self.use_percentile, self.percentile_threshold)
            lang_scores[layer] = self._compute_metrics(lang_embs.to(self.device), n_samples, n_aug, self.use_percentile, self.percentile_threshold)
            both_scores[layer] = self._compute_metrics(both_embs.to(self.device), n_samples, n_aug, self.use_percentile, self.percentile_threshold)
            torch.cuda.empty_cache()
            
            if verbose:
                tqdm.write(f"  {layer}: vis={vis_scores[layer]['Q']:.3f}, lang={lang_scores[layer]['Q']:.3f}, both={both_scores[layer]['Q']:.3f}")
        
        return {"vision": vis_scores, "language": lang_scores, "both": both_scores}

    def _find_best_in(self, scores_dict, layer_subset, metric, is_agg=False):
        """Find best layer within a subset (higher = better)."""
        subset = {k: v for k, v in scores_dict.items() if k in layer_subset}
        if not subset:
            return None
        if is_agg:
            return max(subset, key=lambda k: subset[k][metric]["mean"])
        return max(subset, key=lambda k: subset[k][metric])

    def find_best(self, dataset, layers, n_samples=None, n_aug=None, metric="Q", verbose=True):
        """Find best layers for vision and language robustness."""
        n_samples = n_samples or self.n_samples
        n_aug = n_aug or self.n_aug
        self._samples = None
        self._cache = {}
        
        vis_layers = [l for l in layers if self._is_vision(l)]
        lang_layers = [l for l in layers if self._is_language(l)]
        
        n_forwards = n_samples * (1 + 3 * n_aug)  # anchor + vis_aug + lang_aug + both_aug
        if self.blank_image_for_lang:
            n_forwards += n_samples
        if self.blank_text_for_vision:
            n_forwards += n_samples
        if verbose:
            print(f"[AutoLayer] {len(layers)} layers ({len(vis_layers)} vision, {len(lang_layers)} language)")
            print(f"            {n_samples} samples × {n_aug} augs → {n_forwards} forwards")
            if self.blank_text_for_vision:
                print(f"            Vision robustness: <image, \"\"> mode")
            if self.blank_image_for_lang:
                print(f"            Language robustness: <blank, text> mode")
            print(f"            Both robustness: <image, text> vs <aug(image), aug(text)>")
        
        scores = self.score_layers(dataset, layers, n_samples, n_aug, verbose)
        
        best = {
            "vision_robustness": {
                "overall": self._find_best_in(scores["vision"], layers, metric),
                "vision_layer": self._find_best_in(scores["vision"], vis_layers, metric),
                "language_layer": self._find_best_in(scores["vision"], lang_layers, metric),
            },
            "language_robustness": {
                "overall": self._find_best_in(scores["language"], layers, metric),
                "vision_layer": self._find_best_in(scores["language"], vis_layers, metric),
                "language_layer": self._find_best_in(scores["language"], lang_layers, metric),
            },
            "both_robustness": {
                "overall": self._find_best_in(scores["both"], layers, metric),
                "vision_layer": self._find_best_in(scores["both"], vis_layers, metric),
                "language_layer": self._find_best_in(scores["both"], lang_layers, metric),
            },
        }
        
        if verbose:
            print(f"\n{'='*60}")
            for rob_type, bests in best.items():
                mode_key = "vision" if "vision_rob" in rob_type else ("language" if "language_rob" in rob_type else "both")
                print(f"{rob_type}:")
                for group, layer in bests.items():
                    if layer:
                        score = scores[mode_key][layer][metric]
                        print(f"  {group:15} → {layer} ({metric}={score:.3f})")
            print(f"{'='*60}")
        
        return best, scores

    def _get_model_tag(self):
        """Get model tag from config."""
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
            "model_name": getattr(getattr(self.config, "model", None), "name", "unknown"),
            "n_samples": self.n_samples,
            "n_aug": self.n_aug,
            "run_id": run_id,
            "best": best,
            "scores": scores,
        }
        
        with open(out_path, "w") as f:
            json.dump(out_dict, f, indent=2)
        
        print(f"[AutoLayer] Saved to {out_path}")
        return out_path

    def load_results(self, run_id=None, out_dir=None):
        """Load best layers and scores from JSON."""
        import json
        import os
        
        out_dir = out_dir or "results/auto_layer"
        model_tag = self._get_model_tag()
        suffix = f"_run{run_id}" if run_id is not None else ""
        in_path = os.path.join(out_dir, f"{model_tag}{suffix}.json")
        
        if not os.path.exists(in_path):
            print(f"[AutoLayer] No saved results at {in_path}")
            return None, None
        
        with open(in_path, "r") as f:
            data = json.load(f)
        
        print(f"[AutoLayer] Loaded from {in_path}")
        return data["best"], data["scores"]

    def load_results_k(self, out_dir=None):
        """Load all runs and aggregate into mean/std per layer."""
        import json
        import os
        import glob
        
        out_dir = out_dir or "results/auto_layer"
        model_tag = self._get_model_tag()
        pattern = os.path.join(out_dir, f"{model_tag}_run*.json")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"[AutoLayer] No run files found: {pattern}")
            return None
        
        print(f"[AutoLayer] Loading {len(files)} runs...")
        
        all_scores = []
        for f in files:
            with open(f, "r") as fp:
                all_scores.append(json.load(fp)["scores"])
        
        metrics = ["mse", "dot", "ll", "Q"]
        agg = {}
        for mode in ["vision", "language", "both"]:
            agg[mode] = {}
            if mode not in all_scores[0]:
                continue
            layers = list(all_scores[0][mode].keys())
            for layer in layers:
                agg[mode][layer] = {}
                for m in metrics:
                    vals = [s[mode][layer][m] for s in all_scores if mode in s and layer in s[mode]]
                    agg[mode][layer][m] = {"mean": np.mean(vals), "std": np.std(vals)}
        
        print(f"[AutoLayer] Aggregated {len(files)} runs")
        return agg

    def get_best_from_agg(self, agg_scores, metric="Q"):
        """Get best layers from aggregated scores."""
        layers = list(agg_scores["vision"].keys())
        vis_layers = [l for l in layers if self._is_vision(l)]
        lang_layers = [l for l in layers if self._is_language(l)]
        
        best = {
            "vision_robustness": {
                "overall": self._find_best_in(agg_scores["vision"], layers, metric, is_agg=True),
                "vision_layer": self._find_best_in(agg_scores["vision"], vis_layers, metric, is_agg=True),
                "language_layer": self._find_best_in(agg_scores["vision"], lang_layers, metric, is_agg=True),
            },
            "language_robustness": {
                "overall": self._find_best_in(agg_scores["language"], layers, metric, is_agg=True),
                "vision_layer": self._find_best_in(agg_scores["language"], vis_layers, metric, is_agg=True),
                "language_layer": self._find_best_in(agg_scores["language"], lang_layers, metric, is_agg=True),
            },
        }
        
        if "both" in agg_scores:
            best["both_robustness"] = {
                "overall": self._find_best_in(agg_scores["both"], layers, metric, is_agg=True),
                "vision_layer": self._find_best_in(agg_scores["both"], vis_layers, metric, is_agg=True),
                "language_layer": self._find_best_in(agg_scores["both"], lang_layers, metric, is_agg=True),
            }
        
        print(f"Best layers (from {metric} mean):")
        for rob, bests in best.items():
            mode_key = "vision" if "vision_rob" in rob else ("language" if "language_rob" in rob else "both")
            print(f"  {rob}:")
            for group, layer in bests.items():
                if layer:
                    score = agg_scores[mode_key][layer][metric]["mean"]
                    parts = layer.split('.')
                    short = parts[-3] if len(parts) >= 3 else parts[-1]
                    print(f"    {group}: {short} ({score:.3f})")
        return best

    # ==================== Plotting ====================

    def plot_scores(self, scores, metric="Q", normalize=True, figsize=None):
        """Line plot of scores with optional error bars (vision=green, language=blue)."""
        import matplotlib.pyplot as plt
        
        metrics = [metric] if isinstance(metric, str) else metric
        modes = [m for m in ["vision", "language", "both"] if m in scores]
        fig, axes = plt.subplots(len(metrics), len(modes), figsize=figsize or (5 * len(modes), 3 * len(metrics)), squeeze=False)
        
        for row, m in enumerate(metrics):
            for col, mode in enumerate(modes):
                mode_scores = scores[mode]
                ax = axes[row, col]
                layers = list(mode_scores.keys())
                
                sample_val = mode_scores[layers[0]][m]
                is_agg = isinstance(sample_val, dict) and "mean" in sample_val
                
                if is_agg:
                    vals = np.array([mode_scores[l][m]["mean"] for l in layers])
                    stds = np.array([mode_scores[l][m]["std"] for l in layers])
                else:
                    vals = np.array([mode_scores[l][m] for l in layers])
                    stds = None
                
                indices = np.arange(len(layers))
                is_vis = np.array([self._is_vision(l) for l in layers])
                is_lang = np.array([self._is_language(l) for l in layers])
                
                if normalize:
                    vmin, vmax = vals.min(), vals.max()
                    if vmax - vmin > 1e-8:
                        vals = (vals - vmin) / (vmax - vmin)
                        if stds is not None:
                            stds = stds / (vmax - vmin)
                
                # Plot vision layers (green) and language layers (blue) separately
                vis_idx = indices[is_vis]
                lang_idx = indices[is_lang]
                vis_vals = vals[is_vis]
                lang_vals = vals[is_lang]
                
                if len(vis_idx) > 0:
                    if stds is not None:
                        ax.errorbar(vis_idx, vis_vals, yerr=stds[is_vis], fmt='o-', ms=3, lw=1, 
                                   capsize=2, alpha=0.8, color='green', label='vision')
                    else:
                        ax.plot(vis_idx, vis_vals, 'o-', ms=3, lw=1, color='green', label='vision')
                
                if len(lang_idx) > 0:
                    if stds is not None:
                        ax.errorbar(lang_idx, lang_vals, yerr=stds[is_lang], fmt='o-', ms=3, lw=1, 
                                   capsize=2, alpha=0.8, color='blue', label='language')
                    else:
                        ax.plot(lang_idx, lang_vals, 'o-', ms=3, lw=1, color='blue', label='language')
                
                # Mark best overall (red star), best vision (green triangle), best language (blue triangle)
                best = np.argmax(vals)
                ax.scatter([indices[best]], [vals[best]], c='red', s=120, zorder=6, marker='*', edgecolors='black')
                
                if len(vis_vals) > 0:
                    best_vis = vis_idx[np.argmax(vis_vals)]
                    ax.scatter([best_vis], [vals[best_vis]], c='green', s=80, zorder=5, marker='^', edgecolors='black')
                
                if len(lang_vals) > 0:
                    best_lang = lang_idx[np.argmax(lang_vals)]
                    ax.scatter([best_lang], [vals[best_lang]], c='blue', s=80, zorder=5, marker='^', edgecolors='black')
                
                ax.set_xlabel("Layer" if row == len(metrics) - 1 else "")
                ax.set_ylabel(f"{m} (↑)" if col == 0 else "")
                ax.set_title(f"{mode.capitalize()} - {m}" if row == 0 else "")
                ax.legend(fontsize=7, loc='lower right')
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_all_metrics(self, scores, normalize=True, figsize=(14, 9)):
        """Plot all metrics in a 3x4 grid with vision/language layers colored differently."""
        import matplotlib.pyplot as plt
        
        metrics = ["mse", "dot", "ll", "Q"]
        modes = [m for m in ["vision", "language", "both"] if m in scores]
        fig, axes = plt.subplots(len(modes), 4, figsize=figsize, squeeze=False)
        
        for col, metric in enumerate(metrics):
            for row, mode in enumerate(modes):
                mode_scores = scores[mode]
                ax = axes[row, col]
                layers = list(mode_scores.keys())
                
                # Detect aggregated format
                sample_val = mode_scores[layers[0]][metric]
                is_agg = isinstance(sample_val, dict) and "mean" in sample_val
                
                if is_agg:
                    vals = np.array([mode_scores[l][metric]["mean"] for l in layers])
                    stds = np.array([mode_scores[l][metric]["std"] for l in layers])
                else:
                    vals = np.array([mode_scores[l][metric] for l in layers])
                    stds = None
                
                # Identify vision vs language layers
                is_vis = np.array([self._is_vision(l) for l in layers])
                is_lang = np.array([self._is_language(l) for l in layers])
                
                indices = np.arange(len(layers))
                
                if normalize:
                    vmin, vmax = vals.min(), vals.max()
                    if vmax - vmin > 1e-8:
                        vals = (vals - vmin) / (vmax - vmin)
                        if stds is not None:
                            stds = stds / (vmax - vmin)
                
                # Plot vision layers (green) and language layers (blue) separately
                vis_idx = indices[is_vis]
                lang_idx = indices[is_lang]
                vis_vals = vals[is_vis]
                lang_vals = vals[is_lang]
                
                if len(vis_idx) > 0:
                    if stds is not None:
                        ax.errorbar(vis_idx, vis_vals, yerr=stds[is_vis], fmt='o-', ms=3, lw=1, 
                                   capsize=2, alpha=0.8, color='green', label='vision')
                    else:
                        ax.plot(vis_idx, vis_vals, 'o-', ms=3, lw=1, color='green', label='vision')
                
                if len(lang_idx) > 0:
                    if stds is not None:
                        ax.errorbar(lang_idx, lang_vals, yerr=stds[is_lang], fmt='o-', ms=3, lw=1, 
                                   capsize=2, alpha=0.8, color='blue', label='language')
                    else:
                        ax.plot(lang_idx, lang_vals, 'o-', ms=3, lw=1, color='blue', label='language')
                
                # Mark best overall (red star), best vision (green triangle), best language (blue triangle)
                best = np.argmax(vals)
                ax.scatter([indices[best]], [vals[best]], c='red', s=120, zorder=6, marker='*')
                
                if len(vis_vals) > 0:
                    best_vis = vis_idx[np.argmax(vis_vals)]
                    ax.scatter([best_vis], [vals[best_vis]], c='green', s=80, zorder=5, marker='^')
                
                if len(lang_vals) > 0:
                    best_lang = lang_idx[np.argmax(lang_vals)]
                    ax.scatter([best_lang], [vals[best_lang]], c='blue', s=80, zorder=5, marker='^')
                
                ax.set_title(f"{mode} - {metric} (↑)" if row == 0 else f"{metric} (↑)")
                ax.set_xlabel("Layer" if row == len(modes) - 1 else "")
                ax.grid(alpha=0.3)
                if col == 0:
                    ax.set_ylabel(mode.capitalize())
                if col == 3 and row == 0:  # Legend on top-right plot
                    ax.legend(fontsize=7, loc='lower right')
        
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def plot_embeddings(self, layer_name, dataset=None, n_samples=10, n_aug=5, figsize=(5, 3)):
        """Network plot showing anchors + vision augs + language augs."""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if layer_name not in self._cache or dataset is not None:
            if dataset is None:
                print(f"No cache for {layer_name}. Provide dataset to encode.")
                return
            
            parts = layer_name.split('.')
            short = parts[-3] if len(parts) >= 3 else parts[-1]
            print(f"[AutoLayer] Encoding {n_samples} samples for {short}...")
            data = getattr(dataset, "data", dataset)
            samples = random.sample(list(data), min(n_samples, len(data)))
            
            self._hook_all_layers([layer_name])
            vis_embs, lang_embs = [], []
            
            for s in samples:
                img = s["image"]
                img = Image.open(img).convert("RGB") if isinstance(img, str) else img
                text = s.get("question", "")
                
                embs = self._encode_all(img, text)
                if layer_name in embs:
                    vis_embs.append(embs[layer_name])
                    lang_embs.append(embs[layer_name])
                
                for _ in range(n_aug):
                    embs = self._encode_all(self.augmenter.image(img), text)
                    if layer_name in embs:
                        vis_embs.append(embs[layer_name])
                
                for _ in range(n_aug):
                    embs = self._encode_all(img, self.augmenter.question(text) if text else "")
                    if layer_name in embs:
                        lang_embs.append(embs[layer_name])
            
            self._remove_hooks()
            vis_embs = torch.cat(vis_embs, dim=0).cpu().numpy()
            lang_embs = torch.cat(lang_embs, dim=0).cpu().numpy()
        else:
            vis_data = self._cache[layer_name].get("vision")
            lang_data = self._cache[layer_name].get("language")
            if not vis_data or not lang_data:
                print(f"Need both vision and language embeddings cached.")
                return
            vis_embs, n_samples, n_aug = vis_data
            lang_embs, _, _ = lang_data
            vis_embs, lang_embs = vis_embs.numpy(), lang_embs.numpy()
        
        group_size = 1 + n_aug
        n_show = min(n_samples, len(vis_embs) // group_size)
        
        # Build combined embeddings
        all_embs, node_types, sample_ids = [], [], []
        for i in range(n_show):
            all_embs.append(vis_embs[i * group_size])
            node_types.append("anchor")
            sample_ids.append(i)
            for j in range(n_aug):
                all_embs.append(vis_embs[i * group_size + 1 + j])
                node_types.append("vision")
                sample_ids.append(i)
            for j in range(n_aug):
                all_embs.append(lang_embs[i * group_size + 1 + j])
                node_types.append("language")
                sample_ids.append(i)
        
        all_embs = np.stack(all_embs)
        
        # Build graph
        dists = np.linalg.norm(all_embs[:, None] - all_embs[None, :], axis=-1)
        sims = 1 / (1 + dists)
        
        G = nx.Graph()
        for i in range(len(all_embs)):
            G.add_node(i, ntype=node_types[i], sample_idx=sample_ids[i])
        
        thresh = np.percentile(sims[np.triu_indices(len(all_embs), k=1)], 25)
        for i in range(len(all_embs)):
            for j in range(i + 1, len(all_embs)):
                if sims[i, j] > thresh:
                    G.add_edge(i, j, weight=sims[i, j])
        
        pos = nx.spring_layout(G, weight='weight', seed=42, k=2/np.sqrt(len(all_embs)))
        
        anchors = [i for i in G.nodes if G.nodes[i]['ntype'] == 'anchor']
        vis_augs = [i for i in G.nodes if G.nodes[i]['ntype'] == 'vision']
        lang_augs = [i for i in G.nodes if G.nodes[i]['ntype'] == 'language']
        
        cmap = plt.cm.get_cmap('tab10', n_show)
        
        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=anchors, 
                               node_color=[cmap(G.nodes[i]['sample_idx']) for i in anchors], 
                               node_size=150, node_shape='o', ax=ax, edgecolors='black', linewidths=1)
        nx.draw_networkx_nodes(G, pos, nodelist=vis_augs, 
                               node_color=[cmap(G.nodes[i]['sample_idx']) for i in vis_augs], 
                               node_size=40, alpha=0.7, node_shape='^', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=lang_augs, 
                               node_color=[cmap(G.nodes[i]['sample_idx']) for i in lang_augs], 
                               node_size=40, alpha=0.7, node_shape='o', ax=ax)
        
        ax.scatter([], [], c='gray', s=100, marker='o', edgecolors='black', linewidths=1, label=f'anchor ({len(anchors)})')
        ax.scatter([], [], c='gray', s=40, marker='^', label=f'vis aug ({len(vis_augs)})')
        ax.scatter([], [], c='gray', s=40, marker='o', label=f'lang aug ({len(lang_augs)})')
        ax.legend(loc='upper right', fontsize=8)
        parts = layer_name.split('.')
        short = parts[-3] if len(parts) >= 3 else parts[-1]
        ax.set_title(f'{short} ({n_show} samples)')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def cleanup(self):
        self._remove_hooks()
        self._cache = {}
        self._samples = None
