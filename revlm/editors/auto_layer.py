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
from PIL import Image
from tqdm import tqdm
from .utils import Augmenter, parent_module, brackets_to_periods

# Layers to exclude (variable outputs, not representations)
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

# All robustness modes in order
_ALL_MODES = ["vision", "language", "bimodal", "bimodal_text_partial", "bimodal_image_partial"]


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
        self.n_samples = 100
        self.n_aug = 10
        self.percentile_threshold = 0.9    # Filter out similarities below this percentile (0-1)
        self.threshold_mask = True         # True: exclude from calc, False: zero them
        self.blank_image_for_lang = False   # Use <blank, text> for language robustness
        self.blank_text_for_vision = False  # Use <image, ""> for vision robustness
        self.contrastive_bimodal = True       # Use contrastive target for "bimodal" robustness
        self.weighted_contrastive = False  # Use weighted target [1, 0.5, 0] for contrastive
        self.verbalize_mode = "none"       # "none": <I,T>, "replace": <blank, verb(I)+T>, "augment": <I, verb(I)+T>
        self._verb_cache = {}              # Cache image descriptions
        self.standardize_embeddings = "none"  # "none", "zscore", or "norm"
        self.use_percentile_sim = False        # Convert L2 dist to percentile-based similarity
        self.compute_concat = False           # Compute __concat__ pseudo-layer scores
    
    def _verbalize(self, image):
        """Generate text description of image using VLM."""
        img_id = id(image)
        if img_id not in self._verb_cache:
            prompt = "Describe this image briefly in one sentence."
            desc = self.wrapper.generate([image], [prompt])[0]
            self._verb_cache[img_id] = desc
        return self._verb_cache[img_id]

    def _prepare_input(self, image, text):
        """Prepare inputs based on verbalize_mode.
        
        Modes:
          - "none":    <I, T>                (original)
          - "replace": <blank, verb(I) + T>  (image → text)
          - "augment": <I, verb(I) + T>      (keep image, add verb)
        """
        if self.verbalize_mode == "none" or image is None:
            return image, text
        
        # Check if already a blank image
        is_blank = getattr(image, '_is_blank', False)
        if is_blank:
            return image, text
        
        desc = self._verbalize(image)
        combined = f"{desc} {text}".strip()
        
        if self.verbalize_mode == "replace":
            blank = Image.new("RGB", image.size, (128, 128, 128))
            blank._is_blank = True
            return blank, combined
        elif self.verbalize_mode == "augment":
            return image, combined
        else:
            return image, text

    def get_candidate_layers(self, include_all=False):
        layers = [n for n, _ in self.model.named_parameters() if n.endswith(".weight")]
        if not include_all:
            layers = [l for l in layers if not any(pat in l for pat in EXCLUDE_PATTERNS)]
        vis, merger, lang = self._classify_layers(layers)
        print(f"[AutoLayer] {len(layers)} candidate layers (vision: {len(vis)}, merger: {len(merger)}, language: {len(lang)})")
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
            # Multiple tokens/patches: mean pool
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

    def _is_merger(self, layer_name):
        """Merger layers: vision-language projection."""
        return any(p in layer_name.lower() for p in ["multi_modal_projector", "merger", "language_projection"])

    def _is_vision(self, layer_name):
        """Vision layers: vision tower (excludes merger/language)."""
        if self._is_merger(layer_name) or self._is_language(layer_name):
            return False
        return any(p in layer_name.lower() for p in ["vision", "visual", "qformer"])

    def _is_language(self, layer_name):
        """Language layers: LLM backbone (excludes merger)."""
        return not self._is_merger(layer_name) and "language" in layer_name.lower()

    def _classify_layers(self, layers):
        """Classify layers into vision, merger, and language groups."""
        return (
            [l for l in layers if self._is_vision(l)],
            [l for l in layers if self._is_merger(l)],
            [l for l in layers if self._is_language(l)],
        )

    def _get_mode_key(self, rob_type):
        """Map robustness type to scores key."""
        if "vision_rob" in rob_type:
            return "vision"
        if "language_rob" in rob_type:
            return "language"
        if "text_partial" in rob_type:
            return "bimodal_text_partial"
        if "image_partial" in rob_type:
            return "bimodal_image_partial"
        return "bimodal"

    def _short_layer_name(self, layer_name):
        """Get shortened layer name for display."""
        parts = layer_name.split('.')
        return parts[-3] if len(parts) >= 3 else parts[-1]

    # ==================== Target Matrix Builders ====================

    def _build_standard_target(self, n_samples, n_aug, device):
        """Standard target: 1 cluster per sample (anchor + augs)."""
        group_size = 1 + n_aug
        N = n_samples * group_size
        labels = torch.arange(N, device=device) // group_size
        return (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    def _build_contrastive_target(self, n_samples, n_aug, device):
        """Contrastive target: 3 sub-clusters per sample (A, B, C), all disjoint.
        
        For sample i:
          A: <I_i, T_i> + augs     (sub-cluster 3*i + 0)
          B: <I_i, T_j> + augs     (sub-cluster 3*i + 1)  
          C: <I_k, T_i> + augs     (sub-cluster 3*i + 2)
        
        Target: 1s within each sub-cluster, 0s between all sub-clusters.
        """
        group_size = 1 + n_aug
        n_subclusters = 3 * n_samples
        N = n_subclusters * group_size
        labels = torch.arange(N, device=device) // group_size
        return (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    def _build_contrastive_target_weighted(self, n_samples, n_aug, device):
        """Weighted contrastive target: captures shared modality relationships.
        
        For sample i's 3 sub-clusters (A, B, C):
          A: <I_i, T_i>    B: <I_i, T_j>    C: <I_k, T_i>
        
        Target weights:
          - Same sub-cluster (diagonal): 1.0
          - A↔B (share image I_i): 0.5
          - A↔C (share text T_i): 0.5  
          - B↔C (share nothing): 0.0
        """
        group_size = 1 + n_aug
        n_subclusters = 3 * n_samples
        N = n_subclusters * group_size
        
        # Start with zeros
        target = torch.zeros(N, N, device=device)
        
        for sample_idx in range(n_samples):
            # Sub-cluster indices for this sample
            a_start = (3 * sample_idx + 0) * group_size
            b_start = (3 * sample_idx + 1) * group_size
            c_start = (3 * sample_idx + 2) * group_size
            
            a_end = a_start + group_size
            b_end = b_start + group_size
            c_end = c_start + group_size
            
            # Diagonal blocks: 1.0 (same sub-cluster)
            target[a_start:a_end, a_start:a_end] = 1.0
            target[b_start:b_end, b_start:b_end] = 1.0
            target[c_start:c_end, c_start:c_end] = 1.0
            
            # A↔B: share image → 0.5
            target[a_start:a_end, b_start:b_end] = 0.5
            target[b_start:b_end, a_start:a_end] = 0.5
            
            # A↔C: share text → 0.5
            target[a_start:a_end, c_start:c_end] = 0.5
            target[c_start:c_end, a_start:a_end] = 0.5
            
            # B↔C: share nothing → 0.0 (already zeros)
        
        return target

    def _build_contrastive_target_text_partial(self, n_samples, n_aug, device):
        """Text sensitivity target: ONLY cross-subcluster A↔C.
        
        For sample i's 3 sub-clusters (A, B, C):
          A: <I_i, T_i>    B: <I_i, T_j>    C: <I_k, T_i>
        
        A and C share text T_i → should cluster together
        Tests: Do embeddings cluster by text regardless of image?
        
        Target: ONLY A↔C=1, everything else=0
        """
        group_size = 1 + n_aug
        n_subclusters = 3 * n_samples
        N = n_subclusters * group_size
        
        target = torch.zeros(N, N, device=device)
        
        for sample_idx in range(n_samples):
            a_start = (3 * sample_idx + 0) * group_size
            c_start = (3 * sample_idx + 2) * group_size
            
            a_end = a_start + group_size
            c_end = c_start + group_size
            
            # ONLY A↔C: same text T_i → cluster
            target[a_start:a_end, c_start:c_end] = 1.0
            target[c_start:c_end, a_start:a_end] = 1.0
        
        return target

    def _build_contrastive_target_image_partial(self, n_samples, n_aug, device):
        """Image sensitivity target: ONLY cross-subcluster A↔B.
        
        For sample i's 3 sub-clusters (A, B, C):
          A: <I_i, T_i>    B: <I_i, T_j>    C: <I_k, T_i>
        
        A and B share image I_i → should cluster together
        Tests: Do embeddings cluster by image regardless of text?
        
        Target: ONLY A↔B=1, everything else=0
        """
        group_size = 1 + n_aug
        n_subclusters = 3 * n_samples
        N = n_subclusters * group_size
        
        target = torch.zeros(N, N, device=device)
        
        for sample_idx in range(n_samples):
            a_start = (3 * sample_idx + 0) * group_size
            b_start = (3 * sample_idx + 1) * group_size
            
            a_end = a_start + group_size
            b_end = b_start + group_size
            
            # ONLY A↔B: same image I_i → cluster
            target[a_start:a_end, b_start:b_end] = 1.0
            target[b_start:b_end, a_start:a_end] = 1.0
        
        return target

    # ==================== Metrics ====================

    def _compute_metrics_with_target(self, embs, target, percentile_threshold=0.0, use_mask=True):
        """Compute metrics with custom target matrix (all higher = better).
        
        Args:
            percentile_threshold: If > 0, keep only top (1-threshold) pairs.
            use_mask: If True, exclude filtered pairs; if False, zero them.
        """
        N = len(embs)
        l2_dist = torch.cdist(embs, embs, p=2)
        
        # Distance -> similarity
        if self.use_percentile_sim:
            dist_flat = l2_dist.flatten()
            ranks = dist_flat.argsort().argsort().float()
            sim = (1.0 - ranks / (len(dist_flat) - 1)).view(N, N)
            sim = (sim * 1000).round() / 1000
        else:
            # Linear transform: normalize to [0, 1]
            l2_max = l2_dist.max()
            l2_min = l2_dist.min()
            denom = l2_max - l2_min
            if denom < 1e-8:
                # Degenerate case: all embeddings nearly identical
                sim = torch.ones_like(l2_dist)
            else:
                sim = (l2_max - l2_dist) / denom
        
        # Threshold mask (keep top pairs)
        triu_idx = torch.triu_indices(N, N, offset=1, device=embs.device)
        mask = None
        if percentile_threshold > 0:
            thresh_val = torch.quantile(sim[triu_idx[0], triu_idx[1]], percentile_threshold)
            mask = sim >= thresh_val
            mask.fill_diagonal_(True)
        
        # Apply mask: zero out filtered pairs for all metrics
        sim_masked = sim * mask.float() if mask is not None else sim
        
        # Upper triangle
        sim_triu = sim_masked[triu_idx[0], triu_idx[1]]
        target_triu = target[triu_idx[0], triu_idx[1]]
        
        # MSE and dot
        if use_mask and mask is not None:
            mask_triu = mask[triu_idx[0], triu_idx[1]]
            s, t = sim_triu[mask_triu], target_triu[mask_triu]
            mse_raw = ((s - t) ** 2).sum().item() / max(len(s), 1)
            dot = (s * t).sum().item() / max(t.sum().item(), 1)
        else:
            mse_raw = ((sim_triu - target_triu) ** 2).mean().item()
            dot = (sim_triu * target_triu).sum().item() / max(target_triu.sum().item(), 1)
        
        # LL
        ll = (target * torch.log(sim_masked + 1e-16)).sum().item() / N
        
        # Q (modularity) how well does the ground-truth partition explain this network’s weights.
        sim_no_diag = sim_masked.clone()
        sim_no_diag.fill_diagonal_(0.0)
        m = sim_no_diag.sum() / 2
        if m > 0:
            k = sim_no_diag.sum(dim=1)
            Q = ((sim_no_diag - torch.outer(k, k) / (2 * m)) * target).sum() / (2 * m)
            modularity = Q.item()
        else:
            modularity = 0.0
        
        return {"mse": 1 / (1 + mse_raw), "dot": dot, "ll": ll, "Q": modularity}

    def _compute_metrics(self, embs, n_samples, n_aug, percentile_threshold=0.0, use_mask=True):
        """Compute metrics with standard target (1 cluster per sample)."""
        target = self._build_standard_target(n_samples, n_aug, embs.device)
        return self._compute_metrics_with_target(embs, target, percentile_threshold, use_mask)

    def _compute_metrics_contrastive(self, embs, n_samples, n_aug, percentile_threshold=0.0, use_mask=True):
        """Compute metrics with contrastive target (3 sub-clusters per sample)."""
        if self.weighted_contrastive:
            target = self._build_contrastive_target_weighted(n_samples, n_aug, embs.device)
        else:
            target = self._build_contrastive_target(n_samples, n_aug, embs.device)
        return self._compute_metrics_with_target(embs, target, percentile_threshold, use_mask)

    # ==================== Public API ====================

    def _safe_append(self, lst, emb):
        """Append only if shape matches existing entries."""
        if len(lst) == 0 or lst[0].shape == emb.shape:
            lst.append(emb)

    def _standardize(self, embs):
        """Standardize embeddings to make layers comparable.
        
        Options (self.standardize_embeddings):
          "none": No standardization
          "zscore": Z-score per dimension (mean=0, std=1)
          "norm": Scale by mean L2 norm (preserves direction ratios)
        """
        mode = self.standardize_embeddings
        if not mode or mode == "none":
            return embs
        if mode == "norm":
            scale = embs.norm(dim=-1).mean() + 1e-8
            return embs / scale
        # Default: zscore
        mean = embs.mean(dim=0, keepdim=True)
        std = embs.std(dim=0, keepdim=True) + 1e-8
        return (embs - mean) / std

    def _encode_bimodal_contrastive(self, samples, layers, all_embs, n_aug, pbar):
        """Encode contrastive 'bimodal' robustness: 3 sub-clusters per sample.
        
        For sample i with (I_i, T_i):
          A: <I_i, T_i>     + n_aug × <aug(I_i), aug(T_i)>     (original binding)
          B: <I_i, T_j>     + n_aug × <aug(I_i), aug(T_j)>     (counterfact text)
          C: <I_k, T_i>     + n_aug × <aug(I_k), aug(T_i)>     (counterfact image)
        
        With verbalize_mode != "none": applies verbalization via _prepare_input()
        """
        n = len(samples)
        
        # Preload all images and texts
        images, texts = [], []
        for s in samples:
            img = s["image"]
            img = Image.open(img).convert("RGB") if isinstance(img, str) else img
            images.append(img)
            texts.append(s.get("question", ""))
        
        # Pre-verbalize all images if enabled (cache them)
        if self.verbalize_mode != "none":
            for img in images:
                self._verbalize(img)
        
        for i in range(n):
            if i > 0 and i % 10 == 0:
                torch.cuda.empty_cache()
            
            img_i = images[i]
            text_i = texts[i]
            # Counterfactual: rotate indices
            text_j = texts[(i + 1) % n]
            img_k = images[(i + 2) % n]
            
            # Sub-cluster A: <I_i, T_i> + augs
            img_in, text_in = self._prepare_input(img_i, text_i)
            embs = self._encode_all(img_in, text_in)
            for layer in layers:
                if layer in embs:
                    self._safe_append(all_embs[layer]["bimodal"], embs[layer])
            pbar.update(1)
            for _ in range(n_aug):
                aug_img = self.augmenter.image(img_i)
                aug_text = self.augmenter.question(text_i) if text_i else ""
                img_in, text_in = self._prepare_input(aug_img, aug_text)
                embs = self._encode_all(img_in, text_in)
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["bimodal"], embs[layer])
                pbar.update(1)
            
            # Sub-cluster B: <I_i, T_j> + augs (counterfact text)
            img_in, text_in = self._prepare_input(img_i, text_j)
            embs = self._encode_all(img_in, text_in)
            for layer in layers:
                if layer in embs:
                    self._safe_append(all_embs[layer]["bimodal"], embs[layer])
            pbar.update(1)
            for _ in range(n_aug):
                aug_img = self.augmenter.image(img_i)
                aug_text = self.augmenter.question(text_j) if text_j else ""
                img_in, text_in = self._prepare_input(aug_img, aug_text)
                embs = self._encode_all(img_in, text_in)
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["bimodal"], embs[layer])
                pbar.update(1)
            
            # Sub-cluster C: <I_k, T_i> + augs (counterfact image)
            img_in, text_in = self._prepare_input(img_k, text_i)
            embs = self._encode_all(img_in, text_in)
            for layer in layers:
                if layer in embs:
                    self._safe_append(all_embs[layer]["bimodal"], embs[layer])
            pbar.update(1)
            for _ in range(n_aug):
                aug_img = self.augmenter.image(img_k)
                aug_text = self.augmenter.question(text_i) if text_i else ""
                img_in, text_in = self._prepare_input(aug_img, aug_text)
                embs = self._encode_all(img_in, text_in)
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["bimodal"], embs[layer])
                pbar.update(1)

    def _encode_bimodal_standard(self, img, text, layers, all_embs, n_aug, pbar):
        """Encode standard 'bimodal' robustness: 1 cluster per sample."""
        # Anchor
        img_in, text_in = self._prepare_input(img, text)
        embs = self._encode_all(img_in, text_in)
        for layer in layers:
            if layer in embs:
                self._safe_append(all_embs[layer]["bimodal"], embs[layer])
        pbar.update(1)
        # Augmentations
        for _ in range(n_aug):
            aug_img = self.augmenter.image(img)
            aug_text = self.augmenter.question(text) if text else ""
            img_in, text_in = self._prepare_input(aug_img, aug_text)
            embs = self._encode_all(img_in, text_in)
            for layer in layers:
                if layer in embs:
                    self._safe_append(all_embs[layer]["bimodal"], embs[layer])
            pbar.update(1)

    @torch.no_grad()
    def _compute_concat_scores(self, n_samples, n_aug, verbose=True):
        """Compute scores for __concat__ layer using config's inner_params_vision and inner_params_lang.
        
        Returns dict with scores for all 5 modes, or None if config doesn't have the params.
        """
        # Get layer names from config
        model_cfg = getattr(self.config, "model", None)
        if model_cfg is None:
            if verbose:
                print("[Concat] No model config found, skipping __concat__")
            return None
        
        vis_param_list = getattr(model_cfg, "inner_params_vision", None)
        lang_param_list = getattr(model_cfg, "inner_params_lang", None)
        
        if not vis_param_list or not lang_param_list:
            if verbose:
                print("[Concat] Config missing inner_params_vision or inner_params_lang, skipping __concat__")
            return None
        
        vis_layer = vis_param_list[0]  # e.g., "model.visual.blocks.21.mlp.linear_fc1.weight"
        lang_layer = lang_param_list[0]  # e.g., "model.language_model.layers.35.mlp.gate_proj.weight"
        
        if verbose:
            print(f"\n[Concat] Computing __concat__ pseudo-layer scores...")
            print(f"         vis_layer  = {vis_layer}")
            print(f"         lang_layer = {lang_layer}")
        
        samples = self._samples[:n_samples]
        n = len(samples)
        
        self._hook_all_layers([vis_layer, lang_layer])
        
        def get_img(s):
            img = s["image"]
            return Image.open(img).convert("RGB") if isinstance(img, str) else img
        
        def encode_concat(img, text, scale_lang=1.0):
            """Encode as concat(vis(<img,"">), lang(<blank,text>), vis(<img,text>))."""
            blank = Image.new("RGB", img.size, (128, 128, 128))
            v = self._encode_all(img, "").get(vis_layer)  # image-only
            vl = self._encode_all(img, text).get(vis_layer)  # image-text entangled
            l = self._encode_all(blank, text).get(lang_layer)  # text-only
            if v is None or vl is None or l is None:
                return None
            # Scale up language component for more text weight
            l = l * scale_lang
            return torch.cat([v, l, vl], dim=-1)
        
        vis_embs, lang_embs, bimodal_embs = [], [], []
        
        for i, s in enumerate(samples):
            img_i = get_img(s)
            text_i = s.get("question", "")
            
            # Vision: vary image, keep text
            emb = encode_concat(img_i, text_i)
            if emb is not None:
                vis_embs.append(emb)
            for _ in range(n_aug):
                emb = encode_concat(self.augmenter.image(img_i), text_i)
                if emb is not None:
                    vis_embs.append(emb)
            
            # Language: vary text, keep image
            emb = encode_concat(img_i, text_i)
            if emb is not None:
                lang_embs.append(emb)
            for _ in range(n_aug):
                emb = encode_concat(img_i, self.augmenter.question(text_i) if text_i else "")
                if emb is not None:
                    lang_embs.append(emb)
            
            # Bimodal contrastive: A, B, C
            img_k = get_img(samples[(i+2) % n])
            text_j = samples[(i+1) % n].get("question", "")
            
            for img, text in [(img_i, text_i), (img_i, text_j), (img_k, text_i)]:
                emb = encode_concat(img, text)
                if emb is not None:
                    bimodal_embs.append(emb)
                for _ in range(n_aug):
                    emb = encode_concat(self.augmenter.image(img), self.augmenter.question(text) if text else "")
                    if emb is not None:
                        bimodal_embs.append(emb)
        
        self._remove_hooks()
        
        if not vis_embs or not lang_embs or not bimodal_embs:
            if verbose:
                print("[Concat] No embeddings collected, skipping")
            return None
        
        vis_tensor = torch.cat(vis_embs, dim=0)
        lang_tensor = torch.cat(lang_embs, dim=0)
        bimodal_tensor = torch.cat(bimodal_embs, dim=0)
        
        # Z-score standardize (same as regular layers)
        vis_tensor = self._standardize(vis_tensor).to(self.device)
        lang_tensor = self._standardize(lang_tensor).to(self.device)
        bimodal_tensor = self._standardize(bimodal_tensor).to(self.device)
        
        result = {
            "vision": self._compute_metrics(vis_tensor, n_samples, n_aug, self.percentile_threshold, self.threshold_mask),
            "language": self._compute_metrics(lang_tensor, n_samples, n_aug, self.percentile_threshold, self.threshold_mask),
            "bimodal": self._compute_metrics_contrastive(bimodal_tensor, n_samples, n_aug, self.percentile_threshold, self.threshold_mask),
        }
        
        if self.contrastive_bimodal:
            result["bimodal_text_partial"] = self._compute_metrics_with_target(
                bimodal_tensor, self._build_contrastive_target_text_partial(n_samples, n_aug, self.device),
                self.percentile_threshold, self.threshold_mask)
            result["bimodal_image_partial"] = self._compute_metrics_with_target(
                bimodal_tensor, self._build_contrastive_target_image_partial(n_samples, n_aug, self.device),
                self.percentile_threshold, self.threshold_mask)
        
        if verbose:
            print(f"[Concat] vis={result['vision']['Q']:.3f}, lang={result['language']['Q']:.3f}, "
                  f"bimodal={result['bimodal']['Q']:.3f}", end="")
            if self.contrastive_bimodal:
                print(f", text_p={result['bimodal_text_partial']['Q']:.3f}, img_p={result['bimodal_image_partial']['Q']:.3f}")
            else:
                print()
        
        return result

    @torch.no_grad()
    def score_layers(self, dataset, layers, n_samples=None, n_aug=None, verbose=True):
        """Score ALL layers in one pass."""
        n_samples = n_samples or self.n_samples
        n_aug = n_aug or self.n_aug
        
        if self._samples is None:
            data = getattr(dataset, "data", dataset)
            self._samples = random.sample(list(data), min(n_samples, len(data)))
        
        self._hook_all_layers(layers)
        all_embs = {l: {"vis": [], "lang": [], "bimodal": []} for l in layers}
        
        # Forward count: vision + language + bimodal
        # Vision: (1 anchor if blank_text) + n_aug augs
        # Language: (1 anchor if blank_image) + n_aug augs
        n_forwards = n_samples * n_aug * 2  # base: augs only
        if self.blank_text_for_vision:
            n_forwards += n_samples  # vision anchors
        if self.blank_image_for_lang:
            n_forwards += n_samples  # language anchors
        # Bimodal: contrastive = 3 sub-clusters, standard = 1 cluster
        if self.contrastive_bimodal:
            n_forwards += n_samples * 3 * (1 + n_aug)
        else:
            n_forwards += n_samples * (1 + n_aug)
        
        pbar = tqdm(total=n_forwards, desc="encoding", disable=not verbose)
        
        # Encode vision & language
        for i, s in enumerate(self._samples[:n_samples]):
            if i > 0 and i % 10 == 0:
                torch.cuda.empty_cache()
            
            img = s["image"]
            img = Image.open(img).convert("RGB") if isinstance(img, str) else img
            text = s.get("question", "")
            
            # For vision: test image robustness
            # With verbalize: <blank, verb(I)> vs <blank, verb(aug(I))>
            # Without: <I, ""> vs <aug(I), "">
            if self.blank_text_for_vision:
                img_in, text_in = self._prepare_input(img, "")
                embs = self._encode_all(img_in, text_in)
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["vis"], embs[layer])
                pbar.update(1)
            
            # For language: test text robustness
            # With verbalize: uses _prepare_input to add verb(I) context
            # Without: <blank, T> vs <blank, aug(T)>
            if self.blank_image_for_lang:
                if self.verbalize_mode != "none":
                    img_in, text_in = self._prepare_input(img, text)
                else:
                    blank_img = Image.new("RGB", img.size, (128, 128, 128))
                    img_in, text_in = blank_img, text
                embs = self._encode_all(img_in, text_in)
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["lang"], embs[layer])
                pbar.update(1)
            
            # Vision augmentations
            for _ in range(n_aug):
                aug_img = self.augmenter.image(img)
                img_in, text_in = self._prepare_input(aug_img, "")
                embs = self._encode_all(img_in, text_in)
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["vis"], embs[layer])
                pbar.update(1)
            
            # Language augmentations
            for _ in range(n_aug):
                aug_text = self.augmenter.question(text) if text else ""
                if self.verbalize_mode != "none":
                    img_in, text_in = self._prepare_input(img, aug_text)
                else:
                    blank_img = Image.new("RGB", img.size, (128, 128, 128))
                    img_in, text_in = blank_img, aug_text
                embs = self._encode_all(img_in, text_in)
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["lang"], embs[layer])
                pbar.update(1)
            
            # Standard bimodal (if not contrastive)
            if not self.contrastive_bimodal:
                self._encode_bimodal_standard(img, text, layers, all_embs, n_aug, pbar)
        
        # Contrastive bimodal (needs all samples together for counterfactuals)
        if self.contrastive_bimodal:
            self._encode_bimodal_contrastive(self._samples[:n_samples], layers, all_embs, n_aug, pbar)
        
        pbar.close()
        self._remove_hooks()
        
        # Compute metrics
        # Group sizes: with anchor = 1+n_aug, without anchor = n_aug
        vis_has_anchor = self.blank_text_for_vision
        lang_has_anchor = self.blank_image_for_lang
        
        expected_vis = n_samples * (1 + n_aug if vis_has_anchor else n_aug)
        expected_lang = n_samples * (1 + n_aug if lang_has_anchor else n_aug)
        expected_bimodal = n_samples * 3 * (1 + n_aug) if self.contrastive_bimodal else n_samples * (1 + n_aug)
        
        vis_scores, lang_scores, bimodal_scores = {}, {}, {}
        bimodal_text_partial_scores, bimodal_image_partial_scores = {}, {}
        
        for layer in (tqdm(layers, desc="metrics") if verbose else layers):
            vis_list, lang_list, bimodal_list = all_embs[layer]["vis"], all_embs[layer]["lang"], all_embs[layer]["bimodal"]
            
            if len(vis_list) < expected_vis * 0.5 or len(lang_list) < expected_lang * 0.5 or len(bimodal_list) < expected_bimodal * 0.5:
                if verbose:
                    tqdm.write(f"  Skipping {layer}: insufficient samples")
                continue
            
            vis_embs = torch.cat(vis_list, dim=0)
            lang_embs = torch.cat(lang_list, dim=0)
            bimodal_embs = torch.cat(bimodal_list, dim=0)
            
            # Z-score standardize per layer (makes layers comparable)
            vis_embs = self._standardize(vis_embs)
            lang_embs = self._standardize(lang_embs)
            bimodal_embs = self._standardize(bimodal_embs)
            
            self._cache[layer] = {
                "vision": (vis_embs, n_samples, n_aug),
                "language": (lang_embs, n_samples, n_aug),
                "bimodal": (bimodal_embs, n_samples, n_aug),
            }
            
            # _build_standard_target expects n_aug such that group_size = 1 + n_aug
            # When no anchor: actual_group = n_aug, so pass n_aug - 1
            vis_n_aug_param = n_aug if vis_has_anchor else n_aug - 1
            lang_n_aug_param = n_aug if lang_has_anchor else n_aug - 1
            
            vis_scores[layer] = self._compute_metrics(vis_embs.to(self.device), n_samples, vis_n_aug_param, self.percentile_threshold, self.threshold_mask)
            lang_scores[layer] = self._compute_metrics(lang_embs.to(self.device), n_samples, lang_n_aug_param, self.percentile_threshold, self.threshold_mask)
            
            # Use contrastive metrics for "bimodal" if enabled
            if self.contrastive_bimodal:
                bimodal_embs_gpu = bimodal_embs.to(self.device)
                bimodal_scores[layer] = self._compute_metrics_contrastive(bimodal_embs_gpu, n_samples, n_aug, self.percentile_threshold, self.threshold_mask)
                
                # Compute partial sensitivity scores (text and image)
                text_target = self._build_contrastive_target_text_partial(n_samples, n_aug, self.device)
                image_target = self._build_contrastive_target_image_partial(n_samples, n_aug, self.device)
                bimodal_text_partial_scores[layer] = self._compute_metrics_with_target(bimodal_embs_gpu, text_target, self.percentile_threshold, self.threshold_mask)
                bimodal_image_partial_scores[layer] = self._compute_metrics_with_target(bimodal_embs_gpu, image_target, self.percentile_threshold, self.threshold_mask)
            else:
                bimodal_scores[layer] = self._compute_metrics(bimodal_embs.to(self.device), n_samples, n_aug, self.percentile_threshold, self.threshold_mask)
            torch.cuda.empty_cache()
            
            if verbose:
                msg = f"  {layer}: vis={vis_scores[layer]['Q']:.3f}, lang={lang_scores[layer]['Q']:.3f}, bimodal={bimodal_scores[layer]['Q']:.3f}"
                if self.contrastive_bimodal:
                    msg += f", text_p={bimodal_text_partial_scores[layer]['Q']:.3f}, img_p={bimodal_image_partial_scores[layer]['Q']:.3f}"
                tqdm.write(msg)
        
        result = {"vision": vis_scores, "language": lang_scores, "bimodal": bimodal_scores}
        if self.contrastive_bimodal:
            result["bimodal_text_partial"] = bimodal_text_partial_scores
            result["bimodal_image_partial"] = bimodal_image_partial_scores
        
        # Add __concat__ pseudo-layer using config's inner_params_vision and inner_params_lang
        if not self.compute_concat:
            return result
        concat_scores = self._compute_concat_scores(n_samples, n_aug, verbose)
        if concat_scores:
            for mode in concat_scores:
                if mode in result:
                    result[mode]["__concat__"] = concat_scores[mode]
        
        return result

    def _find_best_in(self, scores_dict, layer_subset, metric, is_agg=False):
        """Find best layer within a subset (higher = better). Skips __concat__."""
        subset = {k: v for k, v in scores_dict.items() if k in layer_subset and k != "__concat__"}
        if not subset:
            return None
        if is_agg:
            return max(subset, key=lambda k: subset[k][metric]["mean"])
        return max(subset, key=lambda k: subset[k][metric])

    def _print_config(self, layers, vis_layers, merger_layers, lang_layers, n_samples, n_aug, n_forwards):
        """Print configuration summary."""
        print(f"[AutoLayer] {len(layers)} layers ({len(vis_layers)} vision, {len(merger_layers)} merger, {len(lang_layers)} language)")
        print(f"            {n_samples} samples × {n_aug} augs → {n_forwards} forwards")
        
        vm = self.verbalize_mode
        if vm == "replace":
            print("            VERBALIZE MODE: replace → <blank, verb(I) + T>")
        elif vm == "augment":
            print("            VERBALIZE MODE: augment → <I, verb(I) + T>")
        
        if self.blank_text_for_vision:
            vis_modes = {"replace": "<blank, verb(I)> vs <blank, verb(aug(I))>",
                         "augment": "<I, verb(I)> vs <aug(I), verb(aug(I))>",
                         "none": '<I, ""> mode'}
            print(f"            Vision robustness: {vis_modes.get(vm, vis_modes['none'])}")
        
        if self.blank_image_for_lang:
            modes = {"replace": "<blank, verb(I) + T> vs <blank, verb(I) + aug(T)>",
                     "augment": "<I, verb(I) + T> vs <I, verb(I) + aug(T)>"}
            print(f"            Language robustness: {modes.get(vm, '<blank, T> mode')}")
        
        if self.contrastive_bimodal:
            mode_str = "CONTRASTIVE" + (" + WEIGHTED [1, 0.5, 0]" if self.weighted_contrastive else "")
            suffix = f"(verb_mode={vm})" if vm != "none" else "(3 sub-clusters: A=<I,T>, B=<I,T'>, C=<I',T>)"
            print(f"            Bimodal robustness: {mode_str} {suffix}")
        else:
            print("            Bimodal robustness: <I, T> vs <aug(I), aug(T)>")

    def find_best(self, dataset, layers, n_samples=None, n_aug=None, metric="Q", verbose=True):
        """Find best layers for vision and language robustness."""
        n_samples = n_samples or self.n_samples
        n_aug = n_aug or self.n_aug
        self._samples = None
        self._cache = {}
        self._verb_cache = {}
        
        vis_layers, merger_layers, lang_layers = self._classify_layers(layers)
        
        # Forward count
        n_forwards = n_samples * n_aug * 2  # base: augs only
        if self.blank_text_for_vision:
            n_forwards += n_samples  # vision anchors
        if self.blank_image_for_lang:
            n_forwards += n_samples  # language anchors
        n_forwards += n_samples * (3 if self.contrastive_bimodal else 1) * (1 + n_aug)
        
        if verbose:
            self._print_config(layers, vis_layers, merger_layers, lang_layers, n_samples, n_aug, n_forwards)
        
        scores = self.score_layers(dataset, layers, n_samples, n_aug, verbose)
        
        best = {
            "vision_robustness": self._build_best_dict(scores["vision"], layers, vis_layers, merger_layers, lang_layers, metric),
            "language_robustness": self._build_best_dict(scores["language"], layers, vis_layers, merger_layers, lang_layers, metric),
            "bimodal_robustness": self._build_best_dict(scores["bimodal"], layers, vis_layers, merger_layers, lang_layers, metric),
        }
        
        # Add partial sensitivity best layers if contrastive mode
        if self.contrastive_bimodal and "bimodal_text_partial" in scores:
            best["bimodal_text_partial"] = self._build_best_dict(scores["bimodal_text_partial"], layers, vis_layers, merger_layers, lang_layers, metric)
            best["bimodal_image_partial"] = self._build_best_dict(scores["bimodal_image_partial"], layers, vis_layers, merger_layers, lang_layers, metric)
        
        if verbose:
            print(f"\n{'='*60}")
            for rob_type, bests in best.items():
                mode_key = self._get_mode_key(rob_type)
                print(f"{rob_type}:")
                for group, layer in bests.items():
                    if layer and mode_key in scores:
                        print(f"  {group:15} → {layer} ({metric}={scores[mode_key][layer][metric]:.3f})")
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
        
        metrics = ["mse", "dot", "ll", "Q"]
        agg = {}
        for mode in _ALL_MODES:
            if mode not in all_scores[0]:
                continue
            agg[mode] = {}
            for layer in all_scores[0][mode]:
                agg[mode][layer] = {}
                for m in metrics:
                    vals = [s[mode][layer][m] for s in all_scores if mode in s and layer in s[mode]]
                    agg[mode][layer][m] = {"mean": np.mean(vals), "std": np.std(vals)}
        
        print(f"[AutoLayer] Aggregated {len(files)} runs")
        return agg

    def _build_best_dict(self, scores_dict, layers, vis_layers, merger_layers, lang_layers, metric, is_agg=False):
        """Build best layer dict for a single mode."""
        return {
            "overall": self._find_best_in(scores_dict, layers, metric, is_agg),
            "vision_layer": self._find_best_in(scores_dict, vis_layers, metric, is_agg),
            "merger_layer": self._find_best_in(scores_dict, merger_layers, metric, is_agg),
            "language_layer": self._find_best_in(scores_dict, lang_layers, metric, is_agg),
        }

    def get_best_from_agg(self, agg_scores, metric="Q"):
        """Get best layers from aggregated scores."""
        layers = list(agg_scores["vision"].keys())
        vis_layers, merger_layers, lang_layers = self._classify_layers(layers)
        
        best = {
            "vision_robustness": self._build_best_dict(agg_scores["vision"], layers, vis_layers, merger_layers, lang_layers, metric, is_agg=True),
            "language_robustness": self._build_best_dict(agg_scores["language"], layers, vis_layers, merger_layers, lang_layers, metric, is_agg=True),
        }
        
        for mode in ["bimodal", "bimodal_text_partial", "bimodal_image_partial"]:
            if mode in agg_scores:
                key = "bimodal_robustness" if mode == "bimodal" else mode
                best[key] = self._build_best_dict(agg_scores[mode], layers, vis_layers, merger_layers, lang_layers, metric, is_agg=True)
        
        print(f"Best layers (from {metric} mean):")
        for rob, bests in best.items():
            mode_key = self._get_mode_key(rob)
            print(f"  {rob}:")
            for group, layer in bests.items():
                if layer and mode_key in agg_scores:
                    score = agg_scores[mode_key][layer][metric]["mean"]
                    print(f"    {group}: {self._short_layer_name(layer)} ({score:.3f})")
        return best

    # ==================== Plotting ====================

    _TITLE_MAP = {"bimodal_text_partial": "Text Partial", "bimodal_image_partial": "Image Partial"}
    _GROUP_COLORS = [('green', 'vision'), ('orange', 'merger'), ('blue', 'language')]

    def _extract_plot_data(self, mode_scores, metric, real_layers, normalize=False):
        """Extract values and stds for plotting."""
        sample_val = mode_scores[real_layers[0]][metric]
        is_agg = isinstance(sample_val, dict) and "mean" in sample_val
        
        if is_agg:
            vals = np.array([mode_scores[l][metric]["mean"] for l in real_layers])
            stds = np.array([mode_scores[l][metric]["std"] for l in real_layers])
        else:
            vals = np.array([mode_scores[l][metric] for l in real_layers])
            stds = None
        
        # Clamp negative Q to 0 for plotting
        if metric == "Q":
            vals = np.maximum(vals, 0)
        
        if normalize:
            vmin, vmax = vals.min(), vals.max()
            if vmax - vmin > 1e-8:
                vals = (vals - vmin) / (vmax - vmin)
                if stds is not None:
                    stds = stds / (vmax - vmin)
        
        return vals, stds, is_agg

    def _plot_layer_scores(self, ax, mode_scores, mode, metric, real_layers, normalize=False, show_legend=True):
        """Plot layer scores on axis with group colors and markers."""
        vals, stds, is_agg = self._extract_plot_data(mode_scores, metric, real_layers, normalize)
        indices = np.arange(len(real_layers))
        
        # Build masks for each group
        masks = {
            'green': np.array([self._is_vision(l) for l in real_layers]),
            'orange': np.array([self._is_merger(l) for l in real_layers]),
            'blue': np.array([self._is_language(l) for l in real_layers]),
        }
        
        # Plot by group
        for color, label in self._GROUP_COLORS:
            mask = masks[color]
            idx = indices[mask]
            if len(idx) > 0:
                if stds is not None:
                    ax.errorbar(idx, vals[mask], yerr=stds[mask], fmt='o', ms=3, capsize=2, alpha=0.8, color=color, label=label)
                else:
                    ax.plot(idx, vals[mask], 'o', ms=3, color=color, label=label)
        
        # Plot __concat__ as red horizontal line (only meaningful for bimodal modes)
        if "__concat__" in mode_scores and mode in ["bimodal", "bimodal_text_partial", "bimodal_image_partial"]:
            concat_val = mode_scores["__concat__"][metric]["mean"] if is_agg else mode_scores["__concat__"][metric]
            if metric == "Q":
                concat_val = max(concat_val, 0)
            ax.axhline(y=concat_val, color='red', linestyle='--', lw=1.5, label='concat', zorder=3)
        
        # Mark best overall (red star) and best per group (triangles)
        best = np.argmax(vals)
        ax.scatter([indices[best]], [vals[best]], c='red', s=120, zorder=6, marker='*', edgecolors='black')
        
        for color, _ in self._GROUP_COLORS:
            mask = masks[color]
            idx = indices[mask]
            if len(idx) > 0:
                best_idx = idx[np.argmax(vals[mask])]
                ax.scatter([best_idx], [vals[best_idx]], c=color, s=80, zorder=5, marker='^', edgecolors='black')
        
        ax.grid(alpha=0.3)
        if show_legend:
            ax.legend(fontsize=7, loc='upper left')

    def plot_scores(self, scores, metric="Q", normalize=False, figsize=None):
        """Line plot of scores with optional error bars."""
        import matplotlib.pyplot as plt
        
        metrics = [metric] if isinstance(metric, str) else metric
        modes = [m for m in _ALL_MODES if m in scores]
        fig, axes = plt.subplots(len(metrics), len(modes), figsize=figsize or (4 * len(modes), 3 * len(metrics)), squeeze=False)
        
        for row, m in enumerate(metrics):
            for col, mode in enumerate(modes):
                mode_scores = scores[mode]
                real_layers = [l for l in mode_scores.keys() if l != "__concat__"]
                ax = axes[row, col]
                
                self._plot_layer_scores(ax, mode_scores, mode, m, real_layers, normalize)
                
                ax.set_xlabel("Layer" if row == len(metrics) - 1 else "")
                ax.set_ylabel(f"{m} (↑)" if col == 0 else "")
                title = self._TITLE_MAP.get(mode, mode.capitalize())
                ax.set_title(f"{title} - {m}" if row == 0 else "")
        
        # Share y-axis per row
        for row in range(len(metrics)):
            ylims = [axes[row, col].get_ylim() for col in range(len(modes))]
            ymin, ymax = min(y[0] for y in ylims), max(y[1] for y in ylims)
            for col in range(len(modes)):
                axes[row, col].set_ylim(ymin, ymax)
        
        plt.tight_layout()
        plt.show()

    def plot_all_metrics(self, scores, normalize=False, figsize=None):
        """Plot all metrics in a grid."""
        import matplotlib.pyplot as plt
        
        metrics = ["mse", "dot", "ll", "Q"]
        modes = [m for m in _ALL_MODES if m in scores]
        fig, axes = plt.subplots(len(modes), 4, figsize=figsize or (14, 2.5 * len(modes)), squeeze=False)
        
        for col, metric in enumerate(metrics):
            for row, mode in enumerate(modes):
                mode_scores = scores[mode]
                real_layers = [l for l in mode_scores.keys() if l != "__concat__"]
                ax = axes[row, col]
                
                show_legend = (col == 3 and row == 0)
                self._plot_layer_scores(ax, mode_scores, mode, metric, real_layers, normalize, show_legend)
                
                ax.set_title(f"{metric} (↑)" if row == 0 else "")
                ax.set_xlabel("Layer" if row == len(modes) - 1 else "")
                if col == 0:
                    ax.set_ylabel(self._TITLE_MAP.get(mode, mode.capitalize()))
        
        # Share y-axis per column
        for col in range(4):
            ylims = [axes[row, col].get_ylim() for row in range(len(modes))]
            ymin, ymax = min(y[0] for y in ylims), max(y[1] for y in ylims)
            for row in range(len(modes)):
                axes[row, col].set_ylim(ymin, ymax)
        
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
            
            print(f"[AutoLayer] Encoding {n_samples} samples for {self._short_layer_name(layer_name)}...")
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
        
        # Build graph (l2_max - l2 similarity)
        dists = np.linalg.norm(all_embs[:, None] - all_embs[None, :], axis=-1)
        sims = dists.max() - dists
        
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
        ax.set_title(f'{self._short_layer_name(layer_name)} ({n_show} samples)')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def cleanup(self):
        self._remove_hooks()
        self._cache = {}
        self._samples = None
        self._verb_cache = {}
