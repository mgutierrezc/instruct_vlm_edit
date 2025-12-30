"""AutoScaler: Find optimal lang_scaler for dual-layer VLM embeddings.

Uses Vision Q and Language Q to find the best trade-off.

Usage:
    searcher = AutoScaler(config, model, inner_params_vision, inner_params_lang)
    results = searcher.search(dataset, lang_scalers=[1, 10, 100])
    searcher.plot(results)
"""

import random
import torch
from PIL import Image
from tqdm import tqdm
from .modularity_core import ModularityCore
from ..utils import parent_module, brackets_to_periods, Augmenter


class AutoScaler(ModularityCore):
    """Find optimal lang_scaler by measuring vision/language modularity tradeoff.
    
    Edge filtering options:
        AutoScaler(..., edge_filter="percentile", edge_filter_kwargs={"percentile": 0.25})
        AutoScaler(..., edge_filter="knn", edge_filter_kwargs={"k": 10, "mutual": True})
        AutoScaler(..., edge_filter="disparity", edge_filter_kwargs={"alpha": 0.05})
    
    Language encoder options:
        lang_encoder="internal" - use VLM's internal language layer (default)
        lang_encoder="sbert"    - use sentence-transformers/paraphrase-mpnet-base-v2
    """

    def __init__(self, config, model, inner_params_vision, inner_params_lang=None, n_samples=10,
                 n_aug=4, edge_filter="none", edge_filter_kwargs=None, lang_encoder="sbert"):
        """
        Args:
            config: Config object with device
            model: VLM wrapper (has .model and .encode)
            inner_params_vision: List of vision layer param names (use first)
            inner_params_lang: List of language layer param names (use first). Optional if lang_encoder="sbert"
            n_samples: Number of samples to use
            n_aug: Number of augmentations per sample for bimodal_Q
            edge_filter: Filter method - "none", "percentile", "knn", or "disparity"
            edge_filter_kwargs: Dict of kwargs for the filter method
            lang_encoder: "internal" (VLM layer) or "sbert" (sentence-transformers)
        """
        super().__init__(getattr(config, "device", torch.device("cpu")))
        
        self.config = config
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.n_samples = n_samples
        self.n_aug = n_aug
        self.edge_filter = edge_filter
        self.edge_filter_kwargs = edge_filter_kwargs or {}
        self.lang_encoder = lang_encoder
        self._augmenter = None
        
        # Activations storage
        self._vision_act = None
        self._lang_act = None
        self._blank_image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        # Setup hooks
        if not inner_params_vision:
            raise ValueError("Requires inner_params_vision")
        
        self._vision_hook = self._setup_hook(inner_params_vision[0], "_vision_act")
        self._lang_hook = None
        self._sbert = None
        if lang_encoder == "internal":
            if not inner_params_lang:
                raise ValueError("inner_params_lang required when lang_encoder='internal'")
            self._lang_hook = self._setup_hook(inner_params_lang[0], "_lang_act")
        elif lang_encoder == "sbert":
            pass # Lazy load sentence-transformers
        else:
            raise ValueError(f"Unknown lang_encoder: {lang_encoder}")
        
        # Cache
        self._images = None
        self._texts = None

    def _setup_hook(self, param_name, attr_name):
        """Register forward hook on layer."""
        name = param_name.rsplit(".", 1)[0] if param_name.endswith((".weight", ".bias")) else param_name
        mod = parent_module(self.model, brackets_to_periods(name))
        layer = getattr(mod, name.rsplit(".", 1)[-1])
        return layer.register_forward_hook(
            lambda m, i, o, an=attr_name: setattr(self, an, i[0].detach() if isinstance(i[0], torch.Tensor) else None)
        )

    def _pool_act(self, act):
        """Pool activation to [B, hidden]."""
        if act is None:
            raise RuntimeError("Hook failed to capture activation")
        act = act.to(self.device, torch.float32)
        if act.dim() == 3:
            return act.mean(dim=1)
        elif act.dim() == 2:
            return act if act.shape[0] == 1 else act.mean(dim=0, keepdim=True)
        elif act.dim() == 1:
            return act.unsqueeze(0)
        else:
            raise RuntimeError(f"Unexpected activation shape: {act.shape}")

    def _get_sbert(self):
        """Lazy load sentence-transformers model."""
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2", device=self.device)
        return self._sbert

    def _encode_sbert(self, text):
        """Get sentence embedding from SBERT."""
        sbert = self._get_sbert()
        emb = sbert.encode([text], convert_to_tensor=True, device=self.device)
        return emb.to(torch.float32)

    def _get_augmenter(self):
        """Lazy init augmenter."""
        if self._augmenter is None:
            self._augmenter = Augmenter(self.wrapper)
        return self._augmenter

    @staticmethod
    def build_aug_target(n_samples, n_aug):
        """Target for augmentation: samples should cluster (anchor + augs together)."""
        group_size = 1 + n_aug
        N = n_samples * group_size
        labels = torch.arange(N) // group_size
        return (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    @torch.no_grad()
    def _encode_bimodal(self, lang_scaler, verbose=True):
        """Encode <image, text> with both image AND text augmentations for bimodal_Q."""
        n = len(self._images)
        augmenter = self._get_augmenter()
        embs = []
        
        pairs = [(i, aug_idx) for i in range(n) for aug_idx in range(1 + self.n_aug)]
        for i, aug_idx in (tqdm(pairs, desc="bimodal", leave=False) if verbose else pairs):
            img, text = self._images[i], self._texts[i]
            if aug_idx == 0:
                # Anchor
                emb = self._encode_dual(img, text, lang_scaler)
            else:
                # Augment both
                aug_img = augmenter.image(img)
                aug_text = augmenter.question(text) if text else ""
                emb = self._encode_dual(aug_img, aug_text, lang_scaler)
            embs.append(emb)
        
        return torch.cat(embs, dim=0)

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

    @torch.no_grad()
    def _encode_dual(self, image, text, lang_scaler):
        """Get dual-layer embedding: concat(vis(<img,text>), scaler*lang(text))."""
        self.model.eval()
        
        # Vision embedding from VLM
        self._vision_act = None
        self.model(**self.wrapper.encode([image], [text], tokenize=False))
        vis_emb = self._pool_act(self._vision_act)
        self._vision_act = None
        
        # Language embedding: internal (VLM layer) or sbert
        if self.lang_encoder == "sbert":
            lang_emb = self._encode_sbert(text) * lang_scaler
        else:
            self._lang_act = None
            self.model(**self.wrapper.encode([self._blank_image], [text], tokenize=False))
            lang_emb = self._pool_act(self._lang_act) * lang_scaler
            self._lang_act = None
        
        return torch.cat([vis_emb, lang_emb], dim=-1).cpu()

    @torch.no_grad()
    def _encode_single(self, image, text, layer="vision"):
        """Get single-layer embedding."""
        # For sbert lang encoder, use sbert directly for lang baseline
        if layer == "lang" and self.lang_encoder == "sbert":
            return self._encode_sbert(text).cpu()
        
        self.model.eval()
        self._vision_act = None
        self._lang_act = None
        self.model(**self.wrapper.encode([image], [text], tokenize=False))
        
        emb = self._pool_act(self._vision_act if layer == "vision" else self._lang_act)
        self._vision_act = None
        self._lang_act = None
        return emb.cpu()

    @torch.no_grad()
    def _encode_all_pairs(self, encode_fn, verbose=True):
        """Encode all n×n (image, text) pairs."""
        n = len(self._images)
        embs = []
        pairs = [(i, j) for i in range(n) for j in range(n)]
        
        for i, j in (tqdm(pairs, desc="encoding", leave=False) if verbose else pairs):
            embs.append(encode_fn(self._images[i], self._texts[j]))
        
        return torch.cat(embs, dim=0)

    @torch.no_grad()
    def search(self, dataset, lang_scalers=None, shift_mode="global", verbose=True):
        """Search for optimal lang_scaler.
        
        Args:
            dataset: Dataset with .data containing image/question samples
            lang_scalers: List of scalers to try
            shift_mode: How to shift Q values for harmonic mean:
                - "global": shift both Vision Q and Language Q by global min (default)
                - "langq_only": shift only Language Q by its min (keeps Vision Q raw)
                - "none": no shifting (raw Q values in harmonic)
            verbose: Print progress
            
        Returns:
            Dict with "scalers" and "baselines"
        """
        if lang_scalers is None:
            import numpy as np
            # Log-spaced from 0.1 to 100 (3 decades, step=0.1 in log10 = 10 pts/decade)
            raw = np.logspace(-1, 2, 31)
            # Round <1 to 2 decimals, >=1 to integers
            rounded = np.where(raw < 1, np.round(raw, 2), np.round(raw, 0))
            lang_scalers = np.unique(rounded).tolist()
        
        # Sample data
        data = getattr(dataset, "data", dataset)
        samples = random.sample(list(data), min(self.n_samples, len(data)))
        n = len(samples)
        
        # Preload
        self._images = []
        self._texts = []
        for s in samples:
            img = s["image"]
            img = Image.open(img).convert("RGB") if isinstance(img, str) else img
            self._images.append(img)
            self._texts.append(s.get("question", ""))
        
        if verbose:
            print(f"[AutoScaler] {n} samples → {n*n} pairs, scalers={lang_scalers}")
            print(f"  Lang encoder: {self.lang_encoder}")
            if self.edge_filter != "none":
                print(f"  Edge filter: {self.edge_filter} {self.edge_filter_kwargs}")
        
        # Get edge filter tuple
        edge_filter = self._get_edge_filter_tuple()
        
        # Baselines
        baselines = {}
        for layer in ["vision", "lang"]:
            if verbose:
                print(f"  Computing {layer}_layer baseline...")
            embs = self._encode_all_pairs(lambda img, txt: self._encode_single(img, txt, layer), verbose)
            baselines[f"{layer}_layer"] = self.compute_scores(embs, n, edge_filter)
            if verbose:
                b = baselines[f"{layer}_layer"]
                print(f"    {layer}_layer: vis_Q={b['vision_Q']:.4f}, lang_Q={b['language_Q']:.4f}, H={b['harmonic']:.4f}")
            torch.cuda.empty_cache()
        
        # Build augmentation target for bimodal_Q
        aug_target = self.build_aug_target(n, self.n_aug).to(self.device)
        
        # Scaler sweep
        results = {}
        for scaler in (tqdm(lang_scalers, desc="scalers") if verbose else lang_scalers):
            # Entangled n×n scores
            embs = self._encode_all_pairs(lambda img, txt: self._encode_dual(img, txt, scaler), verbose)
            results[scaler] = self.compute_scores(embs, n, edge_filter)
            
            # Bimodal_Q: <image, text> + both augs
            bimodal_embs = self._encode_bimodal(scaler, verbose)
            bimodal_Q = self.compute_Q(bimodal_embs, aug_target, edge_filter)
            results[scaler]["bimodal_Q"] = bimodal_Q
            
            if verbose:
                r = results[scaler]
                tqdm.write(f"  scaler={scaler}: vis_Q={r['vision_Q']:.4f}, lang_Q={r['language_Q']:.4f}, bi_Q={r['bimodal_Q']:.4f}")
            torch.cuda.empty_cache()
        
        # Shift Q values based on shift_mode
        if results:
            all_vision_Q = [r["vision_Q"] for r in results.values()]
            all_language_Q = [r["language_Q"] for r in results.values()]
            
            if shift_mode == "global":
                # Shift both by global min
                global_min = min(all_vision_Q + all_language_Q)
                vision_shift = global_min
                lang_shift = global_min
                if verbose:
                    print(f"\n[Shift] mode=global, min={global_min:.4f}")
            elif shift_mode == "langq_only":
                # Shift only Language Q by its min
                lang_min = min(all_language_Q)
                vision_shift = 0.0
                lang_shift = lang_min
                if verbose:
                    print(f"\n[Shift] mode=langq_only, lang_min={lang_min:.4f}")
            else:  # "none"
                vision_shift = 0.0
                lang_shift = 0.0
                if verbose:
                    print(f"\n[Shift] mode=none")
            
            for scaler in results:
                r = results[scaler]
                r["vision_Q_shifted"] = r["vision_Q"] - vision_shift
                r["language_Q_shifted"] = r["language_Q"] - lang_shift
                
                v, l = r["vision_Q_shifted"], r["language_Q_shifted"]
                r["harmonic_shifted"] = 2 * v * l / (v + l) if (v > 0 and l > 0) else 0.0
        
        # Use shifted harmonic for best selection
        best = max(results, key=lambda s: results[s]["harmonic_shifted"])
        if verbose:
            print(f"[AutoScaler] Best scaler={best} (H_shifted={results[best]['harmonic_shifted']:.4f})")
        
        return {"scalers": results, "baselines": baselines, "shift_mode": shift_mode}

    # ==================== Save / Load / Aggregate ====================

    def _get_model_tag(self):
        """Get model tag for saving."""
        model_name = getattr(getattr(self.config, "model", None), "name", "unknown")
        tag = (model_name.split("/")[-1] or "model").replace(" ", "_")
        lang_enc = getattr(self, "lang_encoder", "internal")
        if lang_enc != "internal":
            tag += f"_{lang_enc}"
        return tag

    def save_results(self, results, run_id=None, out_dir=None):
        """Save results to JSON."""
        import json
        import os
        
        out_dir = out_dir or "results/auto_scaler"
        model_tag = self._get_model_tag()
        os.makedirs(out_dir, exist_ok=True)
        
        suffix = f"_run{run_id}" if run_id is not None else ""
        out_path = os.path.join(out_dir, f"{model_tag}{suffix}.json")
        
        out_dict = {
            "model_tag": model_tag,
            "n_samples": self.n_samples,
            "run_id": run_id,
            "shift_mode": results.get("shift_mode", "global"),
            "scalers": results["scalers"],
            "baselines": results.get("baselines", {}),
        }
        
        with open(out_path, "w") as f:
            json.dump(out_dict, f, indent=2)
        
        print(f"[AutoScaler] Saved to {out_path}")
        return out_path

    def load_results(self, run_id=None, out_dir=None):
        """Load single run results from JSON."""
        import json
        import os
        
        out_dir = out_dir or "results/auto_scaler"
        model_tag = self._get_model_tag()
        suffix = f"_run{run_id}" if run_id is not None else ""
        in_path = os.path.join(out_dir, f"{model_tag}{suffix}.json")
        
        if not os.path.exists(in_path):
            print(f"[AutoScaler] No saved results at {in_path}")
            return None
        
        with open(in_path, "r") as f:
            data = json.load(f)
        
        # Convert string keys back to float for scalers
        scalers = {float(k): v for k, v in data["scalers"].items()}
        
        print(f"[AutoScaler] Loaded from {in_path}")
        return {"scalers": scalers, "baselines": data.get("baselines", {})}

    def load_results_k(self, out_dir=None):
        """Load all runs and aggregate into mean/std per scaler."""
        import json
        import os
        import glob
        import numpy as np
        
        out_dir = out_dir or "results/auto_scaler"
        pattern = os.path.join(out_dir, f"{self._get_model_tag()}_run*.json")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"[AutoScaler] No run files found: {pattern}")
            return None
        
        print(f"[AutoScaler] Loading {len(files)} runs...")
        all_results = []
        for f in files:
            with open(f) as fp:
                data = json.load(fp)
                scalers = {float(k): v for k, v in data["scalers"].items()}
                all_results.append({"scalers": scalers, "baselines": data.get("baselines", {})})
        
        # Get all scalers from first run
        scaler_keys = list(all_results[0]["scalers"].keys())
        metrics = ["vision_Q", "language_Q", "harmonic"]
        
        # Aggregate scalers
        agg_scalers = {}
        for s in scaler_keys:
            agg_scalers[s] = {}
            for m in metrics:
                vals = [r["scalers"][s][m] for r in all_results if s in r["scalers"]]
                agg_scalers[s][m] = {"mean": np.mean(vals), "std": np.std(vals)}
        
        # Aggregate baselines
        agg_baselines = {}
        for bl_key in ["vision_layer", "lang_layer"]:
            if bl_key in all_results[0]["baselines"]:
                agg_baselines[bl_key] = {}
                for m in metrics:
                    vals = [r["baselines"][bl_key][m] for r in all_results if bl_key in r["baselines"]]
                    agg_baselines[bl_key][m] = {"mean": np.mean(vals), "std": np.std(vals)}
        
        print(f"[AutoScaler] Aggregated {len(files)} runs, {len(agg_scalers)} scalers")
        return {"scalers": agg_scalers, "baselines": agg_baselines}

    def get_best_from_agg(self, agg_results, metric="harmonic"):
        """Get best scaler from aggregated results."""
        scalers = agg_results["scalers"]
        best = max(scalers, key=lambda s: scalers[s][metric]["mean"])
        
        print(f"Best scaler (from mean {metric}):")
        print(f"  scaler={best}: {metric}={scalers[best][metric]['mean']:.4f}±{scalers[best][metric]['std']:.4f}")
        
        return best

    # ==================== Plotting ====================

    def _is_aggregated(self, results):
        """Check if results are aggregated (have mean/std)."""
        sample_scaler = list(results["scalers"].keys())[0]
        sample_val = results["scalers"][sample_scaler]["vision_Q"]
        return isinstance(sample_val, dict) and "mean" in sample_val

    def plot(self, results, figsize=(10, 5)):
        """Plot results with optional error bars for aggregated results."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        scalers = results["scalers"]
        baselines = results.get("baselines", {})
        shift_mode = results.get("shift_mode", "global")
        is_agg = self._is_aggregated(results)
        
        x_values = sorted(scalers.keys())
        
        # Determine which Q values and harmonic to plot based on shift_mode
        sample_scaler = x_values[0]
        has_shifted = "harmonic_shifted" in scalers[sample_scaler]
        
        # Labels adapt to shift_mode
        if shift_mode == "none":
            harm_key = "harmonic"
            harm_label = "Harmonic (raw)"
            vis_label = "Vision Q"
            lang_label = "Language Q"
        elif shift_mode == "langq_only":
            harm_key = "harmonic_shifted" if has_shifted else "harmonic"
            harm_label = "Harmonic (lang shifted)"
            vis_label = "Vision Q"
            lang_label = "Language Q"
        else:  # "global"
            harm_key = "harmonic_shifted" if has_shifted else "harmonic"
            harm_label = "Harmonic (shifted)"
            vis_label = "Vision Q"
            lang_label = "Language Q"
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Check if bimodal_Q exists
        has_bimodal = "bimodal_Q" in scalers[sample_scaler]
        
        if is_agg:
            vis_Q = np.array([scalers[x]["vision_Q"]["mean"] for x in x_values])
            lang_Q = np.array([scalers[x]["language_Q"]["mean"] for x in x_values])
            harmonic = np.array([scalers[x][harm_key]["mean"] for x in x_values])
            
            vis_std = np.array([scalers[x]["vision_Q"]["std"] for x in x_values])
            lang_std = np.array([scalers[x]["language_Q"]["std"] for x in x_values])
            harm_std = np.array([scalers[x][harm_key]["std"] for x in x_values])
            
            ax.errorbar(x_values, vis_Q, yerr=vis_std, fmt='o-', color='green', 
                       label=vis_label, ms=5, lw=1.5, capsize=3)
            ax.errorbar(x_values, lang_Q, yerr=lang_std, fmt='o-', color='blue', 
                       label=lang_label, ms=5, lw=1.5, capsize=3)
            ax.errorbar(x_values, harmonic, yerr=harm_std, fmt='s-', color='red', 
                       label=harm_label, ms=6, lw=2, capsize=3)
            
            if has_bimodal:
                bimodal = np.array([scalers[x]["bimodal_Q"]["mean"] for x in x_values])
                bimodal_std = np.array([scalers[x]["bimodal_Q"]["std"] for x in x_values])
                ax.errorbar(x_values, bimodal, yerr=bimodal_std, fmt='^-', color='orange', 
                           label='Bimodal Q', ms=5, lw=1.5, capsize=3)
        else:
            vis_Q = [scalers[x]["vision_Q"] for x in x_values]
            lang_Q = [scalers[x]["language_Q"] for x in x_values]
            harmonic = [scalers[x][harm_key] for x in x_values]
            
            ax.plot(x_values, vis_Q, 'o-', color='green', label=vis_label, ms=5, lw=1.5)
            ax.plot(x_values, lang_Q, 'o-', color='blue', label=lang_label, ms=5, lw=1.5)
            ax.plot(x_values, harmonic, 's-', color='red', label=harm_label, ms=6, lw=2)
            
            if has_bimodal:
                bimodal = [scalers[x]["bimodal_Q"] for x in x_values]
                ax.plot(x_values, bimodal, '^-', color='orange', label='Bimodal Q', ms=5, lw=1.5)
        
        ax.set_xscale('log')
        
        # Mark best (use shifted harmonic)
        if is_agg:
            harm_vals = [scalers[x][harm_key]["mean"] for x in x_values]
        else:
            harm_vals = harmonic
        best_idx = np.argmax(harm_vals)
        ax.scatter([x_values[best_idx]], [harm_vals[best_idx]], c='red', s=150, marker='*',
                   zorder=5, edgecolors='black', label=f'Best ({x_values[best_idx]})')
        
        # Baselines - plot all three metrics for each baseline
        if baselines:
            xmin, xmax = x_values[0], x_values[-1]
            for bl_key, style, lbl in [("vision_layer", "--", "VisLayer"), ("lang_layer", ":", "LangLayer")]:
                if bl_key in baselines:
                    bl = baselines[bl_key]
                    if is_agg:
                        v_val = bl["vision_Q"]["mean"]
                        l_val = bl["language_Q"]["mean"]
                        h_val = bl["harmonic"]["mean"]
                    else:
                        v_val = bl["vision_Q"]
                        l_val = bl["language_Q"]
                        h_val = bl["harmonic"]
                    # Plot Vision Q baseline (green)
                    ax.hlines(v_val, xmin, xmax, colors='green', linestyles=style, lw=1.5, alpha=0.5)
                    # Plot Language Q baseline (blue) 
                    ax.hlines(l_val, xmin, xmax, colors='blue', linestyles=style, lw=1.5, alpha=0.5)
                    # Plot Harmonic baseline (gray)
                    ax.hlines(h_val, xmin, xmax, colors='gray', linestyles=style, lw=1.5, alpha=0.7)
                    # Legend entry
                    ax.plot([], [], style, color='gray', lw=1.5, label=f'{lbl} (H={h_val:.3f})')
        
        ax.set_xlabel('lang_scaler')
        ax.set_ylabel('Modularity Q (↑)')
        title = f'Vision Q vs Language Q Trade-off (shift={shift_mode})'
        ax.set_title(title)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def visualize(self, scaler=1.0, mode="network"):
        """Visualize embeddings for current samples with network or heatmap plots.
        
        Args:
            scaler: lang_scaler for dual embedding
            mode: "network" or "heatmap"
        """
        if self._images is None:
            raise RuntimeError("Call search() first to load samples")
        
        n = len(self._images)
        print(f"[AutoScaler] Visualizing dual embedding with scaler={scaler}...")
        
        # Encode all pairs with given scaler
        embs = self._encode_all_pairs(lambda img, txt: self._encode_dual(img, txt, scaler), verbose=True)
        sim = self.compute_similarity(embs)
        
        if mode == "heatmap":
            self.plot_heatmaps(embs, n)
        else:  # network
            vision_target = self.build_vision_target(n).to(self.device)
            language_target = self.build_language_target(n).to(self.device)
            
            self.plot_network(sim, vision_target, n, title=f"Dual (scaler={scaler}) - Vision Clustering")
            self.plot_network(sim, language_target, n, title=f"Dual (scaler={scaler}) - Language Clustering")

    def cleanup(self):
        """Remove hooks and free resources."""
        if hasattr(self, '_vision_hook') and self._vision_hook:
            self._vision_hook.remove()
        if hasattr(self, '_lang_hook') and self._lang_hook:
            self._lang_hook.remove()
        self._images = None
        self._texts = None
        self._sbert = None

