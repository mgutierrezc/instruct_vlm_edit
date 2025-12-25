"""
AutoLayer: Automatic layer selection for VLM embeddings.

Usage:
    auto = AutoLayer(config, model)
    best_vis, best_lang, scores = auto.find_best(dataset, layers)
    auto.plot_scores(scores)
    auto.plot_embeddings(best_vis, "vision")  # uses cached embs
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
        self._hook = None
        self._act = None
        self._cache = {}
        self._samples = None
        # Settings
        self.n_samples = 20
        self.n_aug = 10

    def _hook_layer(self, layer_name):
        if self._hook:
            self._hook.remove()
        name = layer_name.rsplit(".", 1)[0] if layer_name.endswith((".weight", ".bias")) else layer_name
        mod = parent_module(self.model, brackets_to_periods(name))
        layer = getattr(mod, name.rsplit(".", 1)[-1])
        self._hook = layer.register_forward_hook(
            lambda m, inp, out: setattr(self, "_act", inp[0].detach() if isinstance(inp[0], torch.Tensor) else out.detach())
        )

    @torch.no_grad()
    def _encode(self, image, text):
        self.model.eval()
        self._act = None
        self.model(**self.wrapper.encode([image], [text], tokenize=False))
        act = self._act.to(self.device, torch.float32)
        return act.mean(dim=1) if act.dim() == 3 else (act.mean(dim=0, keepdim=True) if act.dim() == 2 and act.shape[0] != 1 else act)

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
    def score_layer(self, dataset, layer_name, n_samples=None, n_aug=None):
        """Score a layer for BOTH vision and language (single pass, shared anchors)."""
        n_samples = n_samples or self.n_samples
        n_aug = n_aug or self.n_aug
        if self._samples is None:
            data = getattr(dataset, "data", dataset)
            self._samples = random.sample(list(data), min(n_samples, len(data)))
        
        self._hook_layer(layer_name)
        vis_embs, lang_embs = [], []
        
        for s in self._samples[:n_samples]:
            img = s["image"]
            img = Image.open(img).convert("RGB") if isinstance(img, str) else img
            text = s.get("question", "")
            
            # Anchor (shared)
            anchor = self._encode(img, text)
            vis_embs.append(anchor)
            lang_embs.append(anchor)
            
            # Vision augmentations
            for _ in range(n_aug):
                vis_embs.append(self._encode(self.augmenter.image(img), text))
            
            # Language augmentations
            for _ in range(n_aug):
                lang_embs.append(self._encode(img, self.augmenter.question(text) if text else ""))
        
        vis_embs = torch.cat(vis_embs, dim=0)
        lang_embs = torch.cat(lang_embs, dim=0)
        
        # Cache both
        if layer_name not in self._cache:
            self._cache[layer_name] = {}
        self._cache[layer_name]["vision"] = (vis_embs.cpu(), n_samples, n_aug)
        self._cache[layer_name]["language"] = (lang_embs.cpu(), n_samples, n_aug)
        
        return {
            "vision": self._compute_metrics(vis_embs, n_samples, n_aug),
            "language": self._compute_metrics(lang_embs, n_samples, n_aug),
        }

    def score_layers(self, dataset, layers, n_samples=None, n_aug=None, verbose=True):
        """Score multiple layers (both vision and language in one pass)."""
        n_samples = n_samples or self.n_samples
        n_aug = n_aug or self.n_aug
        vis_scores, lang_scores = {}, {}
        for layer in (tqdm(layers, desc="scoring") if verbose else layers):
            try:
                both = self.score_layer(dataset, layer, n_samples, n_aug)
                vis_scores[layer] = both["vision"]
                lang_scores[layer] = both["language"]
                if verbose:
                    tqdm.write(f"  {layer.split('.')[-3]}: vis={both['vision']['info_nce']:.3f}, lang={both['language']['info_nce']:.3f}")
            except Exception as e:
                if verbose:
                    tqdm.write(f"  Skip {layer}: {e}")
        return {"vision": vis_scores, "language": lang_scores}

    def find_best(self, dataset, layers, n_samples=None, n_aug=None, metric="info_nce", verbose=True):
        """Find best layers for vision and language (single pass)."""
        n_samples = n_samples or self.n_samples
        n_aug = n_aug or self.n_aug
        self._samples = None
        self._cache = {}
        
        if verbose:
            print(f"[AutoLayer] {len(layers)} layers × {n_samples} samples × {n_aug} augs (single pass)")
        
        scores = self.score_layers(dataset, layers, n_samples, n_aug, verbose)
        vis, lang = scores["vision"], scores["language"]
        
        best_vis = min(vis, key=lambda k: vis[k][metric]) if vis else None
        best_lang = min(lang, key=lambda k: lang[k][metric]) if lang else None
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Best vision:   {best_vis} ({metric}={vis[best_vis][metric]:.3f})")
            print(f"Best language: {best_lang} ({metric}={lang[best_lang][metric]:.3f})")
            print(f"{'='*50}")
            print(f'inner_params_vision: ["{best_vis}"]')
            print(f'inner_params: ["{best_lang}"]')
        
        return best_vis, best_lang, scores

    def _get_model_tag(self):
        """Get model tag from config (same pattern as config_utils)."""
        model_name = getattr(getattr(self.config, "model", None), "name", "unknown")
        return (model_name.split("/")[-1] or "model").replace(" ", "_")

    def save_results(self, scores, out_dir=None):
        """Save scores to JSON file.
        
        Args:
            scores: Output from find_best()
            out_dir: Output directory (default: results/auto_layer)
        """
        import json
        import os
        
        out_dir = out_dir or "results/auto_layer"
        model_tag = self._get_model_tag()
        
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{model_tag}.json")
        
        # Find best layers
        best_vis = min(scores["vision"], key=lambda k: scores["vision"][k]["info_nce"]) if scores["vision"] else None
        best_lang = min(scores["language"], key=lambda k: scores["language"][k]["info_nce"]) if scores["language"] else None
        
        out_dict = {
            "model_tag": model_tag,
            "model_name": getattr(getattr(self.config, "model", None), "name", "unknown"),
            "n_samples": self.n_samples,
            "n_aug": self.n_aug,
            "best_vision_layer": best_vis,
            "best_language_layer": best_lang,
            "scores": scores,
        }
        
        with open(out_path, "w") as f:
            json.dump(out_dict, f, indent=2)
        
        print(f"[AutoLayer] Saved to {out_path}")
        return out_path

    def load_results(self, out_dir=None):
        """Load scores from JSON file.
        
        Returns:
            tuple: (best_vis, best_lang, scores) or (None, None, None) if not found
        """
        import json
        import os
        
        out_dir = out_dir or "results/auto_layer"
        model_tag = self._get_model_tag()
        
        in_path = os.path.join(out_dir, f"{model_tag}.json")
        if not os.path.exists(in_path):
            print(f"[AutoLayer] No saved results at {in_path}")
            return None, None, None
        
        with open(in_path, "r") as f:
            data = json.load(f)
        
        print(f"[AutoLayer] Loaded from {in_path}")
        return data["best_vision_layer"], data["best_language_layer"], data["scores"]

    # ==================== Plotting ====================

    def _normalize(self, vals):
        """Transform to 0-1 range where higher = better."""
        vals = np.array(vals)
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-8:
            return np.ones_like(vals)
        return (vmax - vals) / (vmax - vmin)  # flip: lower raw → higher normalized

    def plot_scores(self, scores, metric="info_nce", normalize=True, figsize=(10, 4)):
        """Line plot of scores over layer index.
        
        Args:
            normalize: If True, transform to 0-1 where higher = better (default True)
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, (mode, mode_scores) in zip(axes, scores.items()):
            layers = list(mode_scores.keys())
            vals = [mode_scores[l][metric] for l in layers]
            indices = [int([p for p in l.split(".") if p.isdigit()][-1]) if any(p.isdigit() for p in l.split(".")) else i for i, l in enumerate(layers)]
            
            order = np.argsort(indices)
            indices, vals = np.array(indices)[order], np.array(vals)[order]
            
            if normalize:
                vals = self._normalize(vals)
                best = np.argmax(vals)
                ylabel = f"{metric} (0-1, ↑better)"
            else:
                best = np.argmin(vals)
                ylabel = f"{metric} (↓better)"
            
            ax.plot(indices, vals, 'o-', ms=4, lw=1.5)
            ax.scatter([indices[best]], [vals[best]], c='red', s=100, zorder=5, label=f'best: layer {indices[best]}')
            ax.set_xlabel("Layer Index")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{mode.capitalize()} Robustness")
            ax.legend()
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

    def plot_embeddings(self, layer_name, max_samples=10, figsize=(7, 6)):
        """Network plot showing anchors + vision augs + language augs together.
        
        - Anchor: large circle
        - Vision aug: triangle
        - Language aug: small circle
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if layer_name not in self._cache:
            print(f"No cached embeddings for {layer_name}. Run score_layer first.")
            return
        
        vis_data = self._cache[layer_name].get("vision")
        lang_data = self._cache[layer_name].get("language")
        if not vis_data or not lang_data:
            print(f"Need both vision and language embeddings cached.")
            return
        
        vis_embs, n_samples, n_aug = vis_data
        lang_embs, _, _ = lang_data
        vis_embs, lang_embs = vis_embs.numpy(), lang_embs.numpy()
        group_size = 1 + n_aug
        
        # Limit samples
        n_show = min(max_samples, n_samples)
        
        # Build combined embeddings: anchors + vis_augs + lang_augs
        all_embs, node_types, sample_ids = [], [], []
        for i in range(n_show):
            # Anchor (from vision, same as language anchor)
            all_embs.append(vis_embs[i * group_size])
            node_types.append("anchor")
            sample_ids.append(i)
            # Vision augs
            for j in range(n_aug):
                all_embs.append(vis_embs[i * group_size + 1 + j])
                node_types.append("vision")
                sample_ids.append(i)
            # Language augs
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
        if self._hook:
            self._hook.remove()
            self._hook = None
        self._cache = {}
        self._samples = None
