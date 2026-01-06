"""BiasLayer: Compute layer-wise vision/text bias using BiasViz.

Usage:
    bias = BiasLayer(config, model)
    layers = bias.get_candidate_layers()
    scores = bias.compute(dataset, layers, n_samples=3)
    bias.save_results(scores, run_id=0)
"""

import random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# Layers to exclude (shared with AutoLayer)
EXCLUDE_PATTERNS = [
    "embed_tokens", "embeddings", "patch_embed", "patch_embedding",
    "class_embedding", "position_embedding", "pos_embed", "query_tokens",
    "layernorm", "layer_norm", "LayerNorm", "input_layernorm",
    "post_attention_layernorm", "pre_layrnorm", "post_layernorm",
    "q_norm", "k_norm", "norm1", "norm2", ".norm.",
    "lm_head", "crossattention",
    "q_proj", "k_proj", "v_proj", "qkv", "o_proj", "attn.proj", "self_attn",
    "rotary", "rope", "attention.attention",
    "qformer", "intermediate", "up_proj", "down_proj"
]


class BiasLayer:
    """Compute layer-wise vision/text bias.
    
    For each layer, computes:
    - vision_bias: positive = image representation too weak
    - text_bias: positive = text representation too weak
    
    Uses BiasViz internally with mode="vision" (raw layer activations).
    """

    def __init__(self, config, model, pool_method="mean"):
        """
        Args:
            config: Config object with device
            model: VLM wrapper
            pool_method: "mean" or "last" token pooling
        """
        self.config = config
        self.device = getattr(config, "device", torch.device("cpu"))
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.pool_method = pool_method

    def get_candidate_layers(self, include_all=False):
        """Get candidate layer names."""
        layers = [n for n, _ in self.model.named_parameters() if n.endswith(".weight")]
        if not include_all:
            layers = [l for l in layers if not any(pat in l for pat in EXCLUDE_PATTERNS)]
        vis, merger, lang = self._classify_layers(layers)
        print(f"[BiasLayer] {len(layers)} layers (vision: {len(vis)}, merger: {len(merger)}, language: {len(lang)})")
        return layers

    def _classify_layers(self, layers):
        """Classify into vision, merger, language."""
        def is_merger(l): return any(p in l.lower() for p in ["multi_modal_projector", "merger", "language_projection"])
        def is_vision(l): return not is_merger(l) and any(p in l.lower() for p in ["vision", "visual", "qformer"]) and "language" not in l.lower()
        def is_lang(l): return not is_merger(l) and "language" in l.lower()
        return [l for l in layers if is_vision(l)], [l for l in layers if is_merger(l)], [l for l in layers if is_lang(l)]

    def _get_model_tag(self):
        """Get model tag for saving."""
        model_name = getattr(getattr(self.config, "model", None), "name", "unknown")
        return (model_name.split("/")[-1] or "model").replace(" ", "_")

    @torch.no_grad()
    def compute(self, dataset, layers, n_samples=3, verbose=True):
        """Compute vision_bias and text_bias at each layer.
        
        Args:
            dataset: VQADataset with image/question pairs
            layers: List of layer names to compute bias for
            n_samples: Number of samples (each = 13 forward passes with n_aug=3)
            verbose: Print progress
        
        Returns:
            Dict[layer, {"vision_bias": float, "text_bias": float, ...}]
        """
        from .bias_viz import BiasViz
        
        # Sample data
        data = getattr(dataset, "data", dataset)
        samples = random.sample(list(data), min(n_samples, len(data)))
        
        images = []
        texts = []
        for s in samples:
            img = s["image"]
            img = Image.open(img).convert("RGB") if isinstance(img, str) else img
            images.append(img)
            texts.append(s.get("question", ""))
        
        if verbose:
            print(f"[BiasLayer] Computing bias at {len(layers)} layers, {len(images)} samples")
        
        bias_scores = {}
        for layer in (tqdm(layers, desc="layer_bias") if verbose else layers):
            # Create BiasViz with mode="vision" (raw layer activations, no SBERT)
            bv = BiasViz(self.config, self.wrapper, mode="vision", pool_method=self.pool_method)
            bv._hook.remove()
            bv._hook = bv._setup_hook(layer)
            
            # Add samples
            for img, text in zip(images, texts):
                bv.add_edit(img, text, [])  # empty cot, BiasViz uses diff_text="pool" by default
            
            # Get bias
            result = bv.compute_bias()
            bias_scores[layer] = {
                "vision_bias": result["vis_bias"],
                "vision_bias_std": result.get("vis_bias_std", 0.0),
                "text_bias": result["text_bias"],
                "text_bias_std": result.get("text_bias_std", 0.0),
            }
            
            if verbose:
                tqdm.write(f"  {layer[-50:]}: vis={result['vis_bias']:+.3f}, txt={result['text_bias']:+.3f}")
            
            bv.cleanup()
            torch.cuda.empty_cache()
        
        return bias_scores

    # ==================== Save / Load / Aggregate ====================

    def save_results(self, bias_scores, run_id=None, out_dir=None):
        """Save bias scores to JSON."""
        import json
        import os
        
        out_dir = out_dir or "results/bias_layer"
        model_tag = self._get_model_tag()
        os.makedirs(out_dir, exist_ok=True)
        
        suffix = f"_run{run_id}" if run_id is not None else ""
        out_path = os.path.join(out_dir, f"{model_tag}{suffix}.json")
        
        with open(out_path, "w") as f:
            json.dump({"bias_scores": bias_scores, "run_id": run_id}, f, indent=2)
        
        print(f"[BiasLayer] Saved to {out_path}")
        return out_path

    def load_results(self, run_id=None, out_dir=None):
        """Load single run results."""
        import json
        import os
        
        out_dir = out_dir or "results/bias_layer"
        model_tag = self._get_model_tag()
        suffix = f"_run{run_id}" if run_id is not None else ""
        in_path = os.path.join(out_dir, f"{model_tag}{suffix}.json")
        
        if not os.path.exists(in_path):
            return None
        
        with open(in_path, "r") as f:
            data = json.load(f)
        return data["bias_scores"]

    def load_results_k(self, out_dir=None):
        """Load and aggregate bias results from all runs."""
        import json
        import os
        import glob
        
        out_dir = out_dir or "results/bias_layer"
        pattern = os.path.join(out_dir, f"{self._get_model_tag()}_run*.json")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"[BiasLayer] No files found: {pattern}")
            return None
        
        print(f"[BiasLayer] Loading {len(files)} runs...")
        all_bias = []
        for f in files:
            with open(f) as fp:
                all_bias.append(json.load(fp)["bias_scores"])
        
        # Aggregate
        agg = {}
        for layer in all_bias[0]:
            agg[layer] = {}
            for m in ["vision_bias", "text_bias"]:
                vals = [b[layer][m] for b in all_bias if layer in b]
                agg[layer][m] = {"mean": np.mean(vals), "std": np.std(vals)}
        
        print(f"[BiasLayer] Aggregated {len(files)} runs, {len(agg)} layers")
        return agg

    # ==================== Plotting ====================

    def plot(self, scores, figsize=(10, 4)):
        """Plot bias scores vs layer index."""
        import matplotlib.pyplot as plt
        
        layers = list(scores.keys())
        vis_layers, merger_layers, lang_layers = self._classify_layers(layers)
        
        # Check if aggregated
        sample_val = scores[layers[0]]["vision_bias"]
        is_agg = isinstance(sample_val, dict) and "mean" in sample_val
        
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
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        if is_agg:
            vis_bias = np.array([scores[l]["vision_bias"]["mean"] for l in layers])
            vis_bias_std = np.array([scores[l]["vision_bias"]["std"] for l in layers])
            txt_bias = np.array([scores[l]["text_bias"]["mean"] for l in layers])
            txt_bias_std = np.array([scores[l]["text_bias"]["std"] for l in layers])
            ax.errorbar(indices, vis_bias, yerr=vis_bias_std, fmt='o', ms=4, 
                       capsize=2, color='green', alpha=0.7, label='vision_bias')
            ax.errorbar(indices, txt_bias, yerr=txt_bias_std, fmt='o', ms=4, 
                       capsize=2, color='blue', alpha=0.7, label='text_bias')
        else:
            vis_bias = np.array([scores[l]["vision_bias"] for l in layers])
            txt_bias = np.array([scores[l]["text_bias"] for l in layers])
            ax.scatter(indices, vis_bias, c='green', s=20, alpha=0.7, label='vision_bias')
            ax.scatter(indices, txt_bias, c='blue', s=20, alpha=0.7, label='text_bias')
        
        ax.axhline(y=0, color='red', linestyle='--', lw=1.5, label='baseline (0)')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Bias (↓ better)')
        ax.set_title('Modality Bias\n(+vis: img weak, +txt: text weak)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add layer type legend
        ax.scatter([], [], c='green', s=30, label='vision layer')
        ax.scatter([], [], c='orange', s=30, label='merger layer')
        ax.scatter([], [], c='blue', s=30, label='language layer')
        ax.legend(fontsize=7, loc='best')
        
        plt.tight_layout()
        plt.show()

    def cleanup(self):
        """Cleanup resources."""
        pass  # No hooks to remove in BiasLayer itself

