"""
AutoLayer: Automatic layer selection for VLM embeddings.

Usage:
    auto = AutoLayer(config, model)
    best, scores = auto.find_best(dataset, layers)
    auto.save_results(best, scores)
    auto.plot_scores(scores)
    auto.plot_embeddings(best["vision_robustness"]["vision_layer"])

Layer naming: vision layers contain "vision", language layers contain "language".
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from .utils import Augmenter, parent_module, brackets_to_periods


class AutoLayer:
    """Score layers by embedding robustness to augmentations."""

    def __init__(self, config, model):
        self.config = config
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))
        self.augmenter = Augmenter(self.wrapper)
        self._hooks = []
        self._all_acts = {}  # {layer_name: activation}
        self._cache = {}
        self._samples = None
        # Settings
        self.n_samples = 20
        self.n_aug = 10

    def _hook_all_layers(self, layer_names):
        """Register hooks on ALL layers at once."""
        self._remove_hooks()
        self._all_acts = {}
        
        for layer_name in layer_names:
            name = layer_name.rsplit(".", 1)[0] if layer_name.endswith((".weight", ".bias")) else layer_name
            try:
                mod = parent_module(self.model, brackets_to_periods(name))
                layer = getattr(mod, name.rsplit(".", 1)[-1])
                # Capture layer_name in closure
                def make_hook(lname):
                    def hook_fn(m, inp, out):
                        act = inp[0].detach() if isinstance(inp[0], torch.Tensor) else out.detach()
                        self._all_acts[lname] = act
                    return hook_fn
                handle = layer.register_forward_hook(make_hook(layer_name))
                self._hooks.append(handle)
            except Exception:
                pass  # Skip layers that can't be hooked

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._all_acts = {}

    def _pool_act(self, act):
        """Pool activation to [1, hidden_dim]."""
        act = act.to(self.device, torch.float32)
        if act.dim() == 3:
            return act.mean(dim=1)
        elif act.dim() == 2 and act.shape[0] != 1:
            return act.mean(dim=0, keepdim=True)
        return act

    @torch.no_grad()
    def _encode_all(self, image, text):
        """Single forward pass, return pooled activations for ALL hooked layers."""
        self.model.eval()
        self._all_acts = {}
        self.model(**self.wrapper.encode([image], [text], tokenize=False))
        return {k: self._pool_act(v) for k, v in self._all_acts.items()}

    # ==================== Metrics (all lower = better) ====================

    def _compute_metrics(self, embs, n_samples, n_aug):
        """Compute all metrics at once."""
        group_size = 1 + n_aug
        embs_norm = F.normalize(embs, dim=-1)
        
        # InfoNCE
        sim = embs_norm @ embs_norm.t() / 0.1
        labels = torch.arange(len(embs), device=embs.device) // group_size
        nce_losses = []
        for i in range(len(embs)):
            mask = torch.arange(len(embs), device=embs.device) != i
            pos = labels[mask] == labels[i]
            if pos.sum() > 0:
                nce_losses.append((torch.logsumexp(sim[i, mask], 0) - torch.logsumexp(sim[i, mask][pos], 0)).item())
        
        # L2, Cosine, Ratio
        l2_dists, cos_dists, intra, anchors = [], [], [], []
        for i in range(n_samples):
            anchor, augs = embs[i * group_size], embs[i * group_size + 1:(i + 1) * group_size]
            anchor_n = embs_norm[i * group_size]
            augs_n = embs_norm[i * group_size + 1:(i + 1) * group_size]
            anchors.append(anchor)
            for j, (aug, aug_n) in enumerate(zip(augs, augs_n)):
                d = torch.norm(aug - anchor).item()
                l2_dists.append(d)
                intra.append(d)
                cos_dists.append(1 - (anchor_n @ aug_n).item())
        
        anchors = torch.stack(anchors)
        inter = [torch.norm(anchors[i] - anchors[j]).item() for i in range(n_samples) for j in range(i + 1, n_samples)]
        
        return {
            "info_nce": np.mean(nce_losses) if nce_losses else float('inf'),
            "mean_l2": np.mean(l2_dists) if l2_dists else float('inf'),
            "mean_cosine": np.mean(cos_dists) if cos_dists else float('inf'),
            "ratio": np.mean(intra) / max(np.mean(inter), 1e-8) if inter else float('inf'),
        }

    # ==================== Public API ====================

    @torch.no_grad()
    def score_layers(self, dataset, layers, n_samples=None, n_aug=None, verbose=True):
        """Score ALL layers in one pass (50× faster than per-layer).
        
        Hook all layers → forward passes → compute metrics for each layer.
        """
        n_samples = n_samples or self.n_samples
        n_aug = n_aug or self.n_aug
        
        # Sample data once
        if self._samples is None:
            data = getattr(dataset, "data", dataset)
            self._samples = random.sample(list(data), min(n_samples, len(data)))
        
        # Hook ALL layers at once
        self._hook_all_layers(layers)
        active_layers = [l for l in layers if any(l in h.__dict__.get('layer_name', l) or True for h in self._hooks)]
        
        # Collect embeddings for all layers: {layer: {"vis": [...], "lang": [...]}}
        all_embs = {l: {"vis": [], "lang": []} for l in layers}
        
        n_forwards = n_samples * (1 + 2 * n_aug)
        pbar = tqdm(total=n_forwards, desc="encoding", disable=not verbose)
        
        for s in self._samples[:n_samples]:
            img = s["image"]
            img = Image.open(img).convert("RGB") if isinstance(img, str) else img
            text = s.get("question", "")
            
            # Anchor (shared for both vision & language)
            embs = self._encode_all(img, text)
            for layer in layers:
                if layer in embs:
                    all_embs[layer]["vis"].append(embs[layer])
                    all_embs[layer]["lang"].append(embs[layer])
            pbar.update(1)
            
            # Vision augmentations
            for _ in range(n_aug):
                embs = self._encode_all(self.augmenter.image(img), text)
                for layer in layers:
                    if layer in embs:
                        all_embs[layer]["vis"].append(embs[layer])
                pbar.update(1)
            
            # Language augmentations
            for _ in range(n_aug):
                embs = self._encode_all(img, self.augmenter.question(text) if text else "")
                for layer in layers:
                    if layer in embs:
                        all_embs[layer]["lang"].append(embs[layer])
                pbar.update(1)
        
        pbar.close()
        self._remove_hooks()
        
        # Compute metrics for each layer
        vis_scores, lang_scores = {}, {}
        for layer in (tqdm(layers, desc="metrics") if verbose else layers):
            vis_list, lang_list = all_embs[layer]["vis"], all_embs[layer]["lang"]
            if not vis_list or not lang_list:
                continue
            
            vis_embs = torch.cat(vis_list, dim=0)
            lang_embs = torch.cat(lang_list, dim=0)
            
            # Cache for plotting
            self._cache[layer] = {
                "vision": (vis_embs.cpu(), n_samples, n_aug),
                "language": (lang_embs.cpu(), n_samples, n_aug),
            }
            
            vis_scores[layer] = self._compute_metrics(vis_embs, n_samples, n_aug)
            lang_scores[layer] = self._compute_metrics(lang_embs, n_samples, n_aug)
            
            if verbose:
                tqdm.write(f"  {layer.split('.')[-3]}: vis={vis_scores[layer]['info_nce']:.3f}, lang={lang_scores[layer]['info_nce']:.3f}")
        
        return {"vision": vis_scores, "language": lang_scores}

    def _find_best_in(self, scores_dict, layer_subset, metric):
        """Find best layer within a subset."""
        subset = {k: v for k, v in scores_dict.items() if k in layer_subset}
        return min(subset, key=lambda k: subset[k][metric]) if subset else None

    def find_best(self, dataset, layers, n_samples=None, n_aug=None, metric="info_nce", verbose=True):
        """Find best layers for vision and language robustness.
        
        Returns:
            best: dict with structure {robustness_type: {layer_group: best_layer}}
            scores: raw scores dict
        """
        n_samples = n_samples or self.n_samples
        n_aug = n_aug or self.n_aug
        self._samples = None
        self._cache = {}
        
        # Group layers by type
        vis_layers = [l for l in layers if "vision" in l.lower()]
        lang_layers = [l for l in layers if "language" in l.lower()]
        
        n_forwards = n_samples * (1 + 2 * n_aug)
        if verbose:
            print(f"[AutoLayer] {len(layers)} layers ({len(vis_layers)} vision, {len(lang_layers)} language)")
            print(f"            {n_samples} samples × {n_aug} augs → {n_forwards} forwards")
        
        scores = self.score_layers(dataset, layers, n_samples, n_aug, verbose)
        
        # Find best for each combination
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
        }
        
        if verbose:
            print(f"\n{'='*60}")
            for rob_type, bests in best.items():
                print(f"{rob_type}:")
                for group, layer in bests.items():
                    if layer:
                        score = scores["vision" if "vision" in rob_type else "language"][layer][metric]
                        print(f"  {group:15} → {layer.split('.')[-3]} ({metric}={score:.3f})")
            print(f"{'='*60}")
        
        return best, scores

    def _get_model_tag(self):
        """Get model tag from config (same pattern as config_utils)."""
        model_name = getattr(getattr(self.config, "model", None), "name", "unknown")
        return (model_name.split("/")[-1] or "model").replace(" ", "_")

    def save_results(self, best, scores, run_id=None, out_dir=None):
        """Save best layers and scores to JSON.
        
        Args:
            best: Output from find_best() - dict of best layers
            scores: Output from find_best() - raw scores
            run_id: Optional run index for multiple runs (e.g., 0, 1, 2...)
            out_dir: Output directory (default: results/auto_layer)
        """
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
        """Load best layers and scores from JSON.
        
        Returns:
            tuple: (best, scores) or (None, None) if not found
        """
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
        """Load all runs and aggregate into mean/std per layer per metric.
        
        Returns:
            agg_scores: {"vision": {layer: {"info_nce": {"mean": x, "std": y}, ...}}, "language": ...}
        """
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
        
        # Collect all scores
        all_scores = []
        for f in files:
            with open(f, "r") as fp:
                data = json.load(fp)
                all_scores.append(data["scores"])
        
        # Aggregate: compute mean/std per layer per metric
        metrics = ["info_nce", "mean_l2", "mean_cosine", "ratio"]
        agg = {}
        for mode in ["vision", "language"]:
            agg[mode] = {}
            layers = list(all_scores[0][mode].keys())
            for layer in layers:
                agg[mode][layer] = {}
                for m in metrics:
                    vals = [s[mode][layer][m] for s in all_scores if layer in s[mode]]
                    agg[mode][layer][m] = {"mean": np.mean(vals), "std": np.std(vals)}
        
        print(f"[AutoLayer] Aggregated {len(files)} runs")
        return agg

    def get_best_from_agg(self, agg_scores, metric="info_nce"):
        """Get best layers from aggregated scores (mean of k runs)."""
        layers = list(agg_scores["vision"].keys())
        vis_layers = [l for l in layers if "vision" in l.lower()]
        lang_layers = [l for l in layers if "language" in l.lower()]
        
        def find_best(scores_dict, subset):
            subset = {k: v for k, v in scores_dict.items() if k in subset}
            return min(subset, key=lambda k: subset[k][metric]["mean"]) if subset else None
        
        best = {
            "vision_robustness": {
                "overall": find_best(agg_scores["vision"], layers),
                "vision_layer": find_best(agg_scores["vision"], vis_layers),
                "language_layer": find_best(agg_scores["vision"], lang_layers),
            },
            "language_robustness": {
                "overall": find_best(agg_scores["language"], layers),
                "vision_layer": find_best(agg_scores["language"], vis_layers),
                "language_layer": find_best(agg_scores["language"], lang_layers),
            },
        }
        
        print(f"Best layers (from {metric} mean):")
        for rob, bests in best.items():
            print(f"  {rob}:")
            for group, layer in bests.items():
                if layer:
                    score = agg_scores["vision" if "vision" in rob else "language"][layer][metric]["mean"]
                    print(f"    {group}: {layer.split('.')[-3]} ({score:.3f})")
        return best

    # ==================== Plotting ====================

    def _normalize(self, vals):
        """Transform to 0-1 range where higher = better."""
        vals = np.array(vals)
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-8:
            return np.ones_like(vals)
        return (vmax - vals) / (vmax - vmin)  # flip: lower raw → higher normalized

    def plot_scores(self, scores, metric="info_nce", normalize=True, figsize=None):
        """Line plot of scores. Supports error bars if scores from load_results_k()."""
        import matplotlib.pyplot as plt
        
        metrics = [metric] if isinstance(metric, str) else metric
        fig, axes = plt.subplots(len(metrics), 2, figsize=figsize or (10, 3 * len(metrics)), squeeze=False)
        
        for row, m in enumerate(metrics):
            for col, (mode, mode_scores) in enumerate(scores.items()):
                ax = axes[row, col]
                layers = list(mode_scores.keys())
                
                # Detect aggregated format: {"mean": x, "std": y} vs raw value
                sample_val = mode_scores[layers[0]][m]
                is_agg = isinstance(sample_val, dict) and "mean" in sample_val
                
                if is_agg:
                    vals = np.array([mode_scores[l][m]["mean"] for l in layers])
                    stds = np.array([mode_scores[l][m]["std"] for l in layers])
                else:
                    vals = np.array([mode_scores[l][m] for l in layers])
                    stds = None
                
                indices = np.array([int([p for p in l.split(".") if p.isdigit()][-1]) if any(p.isdigit() for p in l.split(".")) else i for i, l in enumerate(layers)])
                is_vis = np.array(["vision" in l.lower() for l in layers])
                is_lang = np.array(["language" in l.lower() for l in layers])
                
                order = np.argsort(indices)
                indices, vals, is_vis, is_lang = indices[order], vals[order], is_vis[order], is_lang[order]
                if stds is not None:
                    stds = stds[order]
                if normalize:
                    vmin, vmax = vals.min(), vals.max()
                    if vmax - vmin > 1e-8:
                        vals = (vmax - vals) / (vmax - vmin)
                        if stds is not None:
                            stds = stds / (vmax - vmin)  # scale std too
                
                best_fn = np.argmax if normalize else np.argmin
                best = best_fn(vals)
                best_vis = best_fn(np.where(is_vis, vals, -np.inf if normalize else np.inf)) if is_vis.any() else None
                best_lang = best_fn(np.where(is_lang, vals, -np.inf if normalize else np.inf)) if is_lang.any() else None
                
                # Plot with or without error bars
                if stds is not None:
                    ax.errorbar(indices, vals, yerr=stds, fmt='o-', ms=4, lw=1.5, capsize=2, alpha=0.8)
                else:
                    ax.plot(indices, vals, 'o-', ms=4, lw=1.5)
                
                ax.scatter([indices[best]], [vals[best]], c='red', s=100, zorder=5, label=f'best: L{indices[best]}')
                if best_vis is not None and is_vis[best_vis]:
                    ax.scatter([indices[best_vis]], [vals[best_vis]], c='green', s=80, marker='^', zorder=4, label=f'vis: L{indices[best_vis]}')
                if best_lang is not None and is_lang[best_lang]:
                    ax.scatter([indices[best_lang]], [vals[best_lang]], c='blue', s=80, marker='s', zorder=4, label=f'lang: L{indices[best_lang]}')
                
                ax.set_xlabel("Layer" if row == len(metrics) - 1 else "")
                ax.set_ylabel(f"{m} ({'↑' if normalize else '↓'})" if col == 0 else "")
                ax.set_title(f"{mode.capitalize()} - {m}" if row == 0 else "")
                ax.legend(fontsize=7)
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_all_metrics(self, scores, normalize=True, figsize=(12, 8)):
        """Plot all 4 metrics in a 2x4 grid.
        
        Args:
            normalize: If True, transform to 0-1 where higher = better (default True)
        """
        import matplotlib.pyplot as plt
        
        metrics = ["info_nce", "mean_l2", "mean_cosine", "ratio"]
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        
        for col, metric in enumerate(metrics):
            for row, (mode, mode_scores) in enumerate(scores.items()):
                ax = axes[row, col]
                layers = list(mode_scores.keys())
                vals = [mode_scores[l][metric] for l in layers]
                indices = [int([p for p in l.split(".") if p.isdigit()][-1]) if any(p.isdigit() for p in l.split(".")) else i for i, l in enumerate(layers)]
                
                order = np.argsort(indices)
                indices, vals = np.array(indices)[order], np.array(vals)[order]
                
                if normalize:
                    vals = self._normalize(vals)
                    best = np.argmax(vals)
                else:
                    best = np.argmin(vals)
                
                ax.plot(indices, vals, 'o-', ms=3, lw=1)
                ax.scatter([indices[best]], [vals[best]], c='red', s=50, zorder=5)
                title = f"{mode} - {metric}" if row == 0 else metric
                ax.set_title(title + (" (↑)" if normalize else " (↓)"))
                ax.set_xlabel("Layer" if row == 1 else "")
                ax.grid(alpha=0.3)
                if col == 0:
                    ax.set_ylabel(mode.capitalize())
        
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def plot_embeddings(self, layer_name, dataset=None, n_samples=10, n_aug=5, figsize=(5, 3)):
        """Network plot showing anchors + vision augs + language augs.
        
        Args:
            layer_name: Layer to visualize
            dataset: Dataset to sample from (required if no cache)
            n_samples: Number of samples to plot
            n_aug: Number of augmentations per sample
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Encode on the fly if needed
        if layer_name not in self._cache or dataset is not None:
            if dataset is None:
                print(f"No cache for {layer_name}. Provide dataset to encode.")
                return
            
            print(f"[AutoLayer] Encoding {n_samples} samples for {layer_name.split('.')[-3]}...")
            data = getattr(dataset, "data", dataset)
            samples = random.sample(list(data), min(n_samples, len(data)))
            
            self._hook_all_layers([layer_name])
            vis_embs, lang_embs = [], []
            
            for s in samples:
                img = s["image"]
                img = Image.open(img).convert("RGB") if isinstance(img, str) else img
                text = s.get("question", "")
                
                # Anchor
                embs = self._encode_all(img, text)
                if layer_name in embs:
                    vis_embs.append(embs[layer_name])
                    lang_embs.append(embs[layer_name])
                
                # Vision augs
                for _ in range(n_aug):
                    embs = self._encode_all(self.augmenter.image(img), text)
                    if layer_name in embs:
                        vis_embs.append(embs[layer_name])
                
                # Language augs
                for _ in range(n_aug):
                    embs = self._encode_all(img, self.augmenter.question(text) if text else "")
                    if layer_name in embs:
                        lang_embs.append(embs[layer_name])
            
            self._remove_hooks()
            vis_embs = torch.cat(vis_embs, dim=0).cpu().numpy()
            lang_embs = torch.cat(lang_embs, dim=0).cpu().numpy()
        else:
            # Use cache
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
        
        thresh = np.percentile(sims[np.triu_indices(len(all_embs), k=1)], 50)
        for i in range(len(all_embs)):
            for j in range(i + 1, len(all_embs)):
                if sims[i, j] > thresh:
                    G.add_edge(i, j, weight=sims[i, j])
        
        pos = nx.spring_layout(G, weight='weight', seed=42, k=2/np.sqrt(len(all_embs)))
        
        # Split by type
        anchors = [i for i in G.nodes if G.nodes[i]['ntype'] == 'anchor']
        vis_augs = [i for i in G.nodes if G.nodes[i]['ntype'] == 'vision']
        lang_augs = [i for i in G.nodes if G.nodes[i]['ntype'] == 'language']
        
        cmap = plt.cm.get_cmap('tab10', n_show)
        
        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5, ax=ax)
        # Anchors: large circles
        nx.draw_networkx_nodes(G, pos, nodelist=anchors, node_color=[cmap(G.nodes[i]['sample_idx']) for i in anchors], 
                               node_size=150, node_shape='o', ax=ax, edgecolors='black', linewidths=1)
        # Vision augs: triangles
        nx.draw_networkx_nodes(G, pos, nodelist=vis_augs, node_color=[cmap(G.nodes[i]['sample_idx']) for i in vis_augs], 
                               node_size=40, alpha=0.7, node_shape='^', ax=ax)
        # Language augs: small circles
        nx.draw_networkx_nodes(G, pos, nodelist=lang_augs, node_color=[cmap(G.nodes[i]['sample_idx']) for i in lang_augs], 
                               node_size=40, alpha=0.7, node_shape='o', ax=ax)
        
        # Legend
        ax.scatter([], [], c='gray', s=100, marker='o', edgecolors='black', linewidths=1, label=f'anchor ({len(anchors)})')
        ax.scatter([], [], c='gray', s=40, marker='^', label=f'vis aug ({len(vis_augs)})')
        ax.scatter([], [], c='gray', s=40, marker='o', label=f'lang aug ({len(lang_augs)})')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'{layer_name.split(".")[-3]} ({n_show} samples)')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def cleanup(self):
        self._remove_hooks()
        self._cache = {}
        self._samples = None
