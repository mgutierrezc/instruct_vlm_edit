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
    "intermediate", "up_proj", "down_proj"
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
        self.contrastive_both = True       # Use contrastive target for "both" robustness
        self.weighted_contrastive = False  # Use weighted target [1, 0.5, 0] for contrastive
        self.verbalize_mode = "none"       # "none": <I,T>, "replace": <blank, verb(I)+T>, "augment": <I, verb(I)+T>
        self._verb_cache = {}              # Cache image descriptions

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
        layers = [n for n, p in self.model.named_parameters() if n.endswith(".weight")]
        if not include_all:
            layers = [l for l in layers if not any(pat in l for pat in EXCLUDE_PATTERNS)]
        vis = [l for l in layers if self._is_vision(l)]
        merger = [l for l in layers if self._is_merger(l)]
        lang = [l for l in layers if self._is_language(l)]
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

    def _is_merger(self, layer_name):
        """Merger layers: vision-language projection (LLaVA: multi_modal_projector, Qwen: merger, BLIP: language_projection)."""
        l = layer_name.lower()
        return any(p in l for p in ["multi_modal_projector", "merger", "language_projection"])

    def _is_vision(self, layer_name):
        """Vision layers: vision tower (excludes merger)."""
        l = layer_name.lower()
        if self._is_merger(layer_name) or self._is_language(layer_name):
            return False
        return any(p in l for p in ["vision", "visual", "qformer"])

    def _is_language(self, layer_name):
        """Language layers: LLM backbone (excludes merger)."""
        l = layer_name.lower()
        if self._is_merger(layer_name):
            return False
        return "language" in l

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
        """Text sensitivity target: cluster by same text.
        
        For sample i's 3 sub-clusters (A, B, C):
          A: <I_i, T_i>    B: <I_i, T_j>    C: <I_k, T_i>
        
        A and C share text T_i → cluster together
        B has different text T_j → separate
        
        Target: A↔A=1, C↔C=1, A↔C=1, B↔B=1, A↔B=0, B↔C=0
        """
        group_size = 1 + n_aug
        n_subclusters = 3 * n_samples
        N = n_subclusters * group_size
        
        target = torch.zeros(N, N, device=device)
        
        for sample_idx in range(n_samples):
            a_start = (3 * sample_idx + 0) * group_size
            b_start = (3 * sample_idx + 1) * group_size
            c_start = (3 * sample_idx + 2) * group_size
            
            a_end = a_start + group_size
            b_end = b_start + group_size
            c_end = c_start + group_size
            
            # Diagonal blocks
            target[a_start:a_end, a_start:a_end] = 1.0
            target[b_start:b_end, b_start:b_end] = 1.0
            target[c_start:c_end, c_start:c_end] = 1.0
            
            # A↔C: same text T_i → cluster
            target[a_start:a_end, c_start:c_end] = 1.0
            target[c_start:c_end, a_start:a_end] = 1.0
            
            # A↔B, B↔C: different text → 0 (already zeros)
        
        return target

    def _build_contrastive_target_image_partial(self, n_samples, n_aug, device):
        """Image sensitivity target: cluster by same image.
        
        For sample i's 3 sub-clusters (A, B, C):
          A: <I_i, T_i>    B: <I_i, T_j>    C: <I_k, T_i>
        
        A and B share image I_i → cluster together
        C has different image I_k → separate
        
        Target: A↔A=1, B↔B=1, A↔B=1, C↔C=1, A↔C=0, B↔C=0
        """
        group_size = 1 + n_aug
        n_subclusters = 3 * n_samples
        N = n_subclusters * group_size
        
        target = torch.zeros(N, N, device=device)
        
        for sample_idx in range(n_samples):
            a_start = (3 * sample_idx + 0) * group_size
            b_start = (3 * sample_idx + 1) * group_size
            c_start = (3 * sample_idx + 2) * group_size
            
            a_end = a_start + group_size
            b_end = b_start + group_size
            c_end = c_start + group_size
            
            # Diagonal blocks
            target[a_start:a_end, a_start:a_end] = 1.0
            target[b_start:b_end, b_start:b_end] = 1.0
            target[c_start:c_end, c_start:c_end] = 1.0
            
            # A↔B: same image I_i → cluster
            target[a_start:a_end, b_start:b_end] = 1.0
            target[b_start:b_end, a_start:a_end] = 1.0
            
            # A↔C, B↔C: different image → 0 (already zeros)
        
        return target

    # ==================== Metrics ====================

    def _compute_metrics_with_target(self, embs, target, use_percentile=False, percentile_threshold=0.0):
        """Compute metrics with custom target matrix (all higher = better)."""
        N = len(embs)
        
        # Pairwise L2 distance -> similarity (0-1)
        l2_dist = torch.cdist(embs, embs, p=2)
        sim = 1 / (1 + l2_dist)
        
        # Upper triangle (exclude diagonal)
        triu_idx = torch.triu_indices(N, N, offset=1, device=embs.device)
        sim_triu = sim[triu_idx[0], triu_idx[1]]
        
        if use_percentile:
            ranks = sim_triu.argsort().argsort().float()
            vals_triu = (ranks / max(len(sim_triu) - 1, 1))
            vals_triu = (vals_triu * 1000).round() / 1000
        else:
            vals_triu = sim_triu
        
        # Full symmetric matrix for log-likelihood
        sim_full = torch.zeros_like(sim)
        sim_full[triu_idx[0], triu_idx[1]] = vals_triu
        sim_full[triu_idx[1], triu_idx[0]] = vals_triu
        
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
        
        # Weighted modularity Q
        k = sim.sum(dim=1)
        m = sim.sum() / 2
        null_model = torch.outer(k, k) / (2 * m)
        Q = ((sim - null_model) * target).sum() / (2 * m)
        modularity = Q.item()
        
        return {"mse": mse, "dot": dot, "ll": ll, "Q": modularity}

    def _compute_metrics(self, embs, n_samples, n_aug, use_percentile=False, percentile_threshold=0.0):
        """Compute metrics with standard target (1 cluster per sample)."""
        target = self._build_standard_target(n_samples, n_aug, embs.device)
        return self._compute_metrics_with_target(embs, target, use_percentile, percentile_threshold)

    def _compute_metrics_contrastive(self, embs, n_samples, n_aug, use_percentile=False, percentile_threshold=0.0):
        """Compute metrics with contrastive target (3 sub-clusters per sample)."""
        if self.weighted_contrastive:
            target = self._build_contrastive_target_weighted(n_samples, n_aug, embs.device)
        else:
            target = self._build_contrastive_target(n_samples, n_aug, embs.device)
        return self._compute_metrics_with_target(embs, target, use_percentile, percentile_threshold)

    # ==================== Public API ====================

    def _safe_append(self, lst, emb):
        """Append only if shape matches existing entries."""
        if len(lst) == 0 or lst[0].shape == emb.shape:
            lst.append(emb)

    def _encode_both_contrastive(self, samples, layers, all_embs, n_aug, pbar):
        """Encode contrastive 'both' robustness: 3 sub-clusters per sample.
        
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
                    self._safe_append(all_embs[layer]["both"], embs[layer])
            pbar.update(1)
            for _ in range(n_aug):
                aug_img = self.augmenter.image(img_i)
                aug_text = self.augmenter.question(text_i) if text_i else ""
                img_in, text_in = self._prepare_input(aug_img, aug_text)
                embs = self._encode_all(img_in, text_in)
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["both"], embs[layer])
                pbar.update(1)
            
            # Sub-cluster B: <I_i, T_j> + augs (counterfact text)
            img_in, text_in = self._prepare_input(img_i, text_j)
            embs = self._encode_all(img_in, text_in)
            for layer in layers:
                if layer in embs:
                    self._safe_append(all_embs[layer]["both"], embs[layer])
            pbar.update(1)
            for _ in range(n_aug):
                aug_img = self.augmenter.image(img_i)
                aug_text = self.augmenter.question(text_j) if text_j else ""
                img_in, text_in = self._prepare_input(aug_img, aug_text)
                embs = self._encode_all(img_in, text_in)
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["both"], embs[layer])
                pbar.update(1)
            
            # Sub-cluster C: <I_k, T_i> + augs (counterfact image)
            img_in, text_in = self._prepare_input(img_k, text_i)
            embs = self._encode_all(img_in, text_in)
            for layer in layers:
                if layer in embs:
                    self._safe_append(all_embs[layer]["both"], embs[layer])
            pbar.update(1)
            for _ in range(n_aug):
                aug_img = self.augmenter.image(img_k)
                aug_text = self.augmenter.question(text_i) if text_i else ""
                img_in, text_in = self._prepare_input(aug_img, aug_text)
                embs = self._encode_all(img_in, text_in)
                for layer in layers:
                    if layer in embs:
                        self._safe_append(all_embs[layer]["both"], embs[layer])
                pbar.update(1)

    def _encode_both_standard(self, img, text, layers, all_embs, n_aug, pbar):
        """Encode standard 'both' robustness: 1 cluster per sample."""
        # Anchor
        img_in, text_in = self._prepare_input(img, text)
        embs = self._encode_all(img_in, text_in)
        for layer in layers:
            if layer in embs:
                self._safe_append(all_embs[layer]["both"], embs[layer])
        pbar.update(1)
        # Augmentations
        for _ in range(n_aug):
            aug_img = self.augmenter.image(img)
            aug_text = self.augmenter.question(text) if text else ""
            img_in, text_in = self._prepare_input(aug_img, aug_text)
            embs = self._encode_all(img_in, text_in)
            for layer in layers:
                if layer in embs:
                    self._safe_append(all_embs[layer]["both"], embs[layer])
            pbar.update(1)

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
        
        # Forward count: vision + language + both
        n_forwards = n_samples * (1 + n_aug) * 2  # vis + lang base
        if self.blank_image_for_lang:
            n_forwards += n_samples
        if self.blank_text_for_vision:
            n_forwards += n_samples
        # Both: contrastive = 3 sub-clusters, standard = 1 cluster
        if self.contrastive_both:
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
            
            # Standard both (if not contrastive)
            if not self.contrastive_both:
                self._encode_both_standard(img, text, layers, all_embs, n_aug, pbar)
        
        # Contrastive both (needs all samples together for counterfactuals)
        if self.contrastive_both:
            self._encode_both_contrastive(self._samples[:n_samples], layers, all_embs, n_aug, pbar)
        
        pbar.close()
        self._remove_hooks()
        
        # Compute metrics
        expected_vis_lang = n_samples * (1 + n_aug)
        expected_both = n_samples * 3 * (1 + n_aug) if self.contrastive_both else n_samples * (1 + n_aug)
        vis_scores, lang_scores, both_scores = {}, {}, {}
        both_text_partial_scores, both_image_partial_scores = {}, {}
        
        for layer in (tqdm(layers, desc="metrics") if verbose else layers):
            vis_list, lang_list, both_list = all_embs[layer]["vis"], all_embs[layer]["lang"], all_embs[layer]["both"]
            
            if len(vis_list) < expected_vis_lang * 0.5 or len(lang_list) < expected_vis_lang * 0.5 or len(both_list) < expected_both * 0.5:
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
            
            # Use contrastive metrics for "both" if enabled
            if self.contrastive_both:
                both_embs_gpu = both_embs.to(self.device)
                both_scores[layer] = self._compute_metrics_contrastive(both_embs_gpu, n_samples, n_aug, self.use_percentile, self.percentile_threshold)
                
                # Compute partial sensitivity scores (text and image)
                text_target = self._build_contrastive_target_text_partial(n_samples, n_aug, self.device)
                image_target = self._build_contrastive_target_image_partial(n_samples, n_aug, self.device)
                both_text_partial_scores[layer] = self._compute_metrics_with_target(both_embs_gpu, text_target, self.use_percentile, self.percentile_threshold)
                both_image_partial_scores[layer] = self._compute_metrics_with_target(both_embs_gpu, image_target, self.use_percentile, self.percentile_threshold)
            else:
                both_scores[layer] = self._compute_metrics(both_embs.to(self.device), n_samples, n_aug, self.use_percentile, self.percentile_threshold)
            torch.cuda.empty_cache()
            
            if verbose:
                if self.contrastive_both:
                    tqdm.write(f"  {layer}: vis={vis_scores[layer]['Q']:.3f}, lang={lang_scores[layer]['Q']:.3f}, both={both_scores[layer]['Q']:.3f}, text_p={both_text_partial_scores[layer]['Q']:.3f}, img_p={both_image_partial_scores[layer]['Q']:.3f}")
                else:
                    tqdm.write(f"  {layer}: vis={vis_scores[layer]['Q']:.3f}, lang={lang_scores[layer]['Q']:.3f}, both={both_scores[layer]['Q']:.3f}")
        
        result = {"vision": vis_scores, "language": lang_scores, "both": both_scores}
        if self.contrastive_both:
            result["both_text_partial"] = both_text_partial_scores
            result["both_image_partial"] = both_image_partial_scores
        return result

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
        self._verb_cache = {}  # Clear verbalization cache
        
        vis_layers = [l for l in layers if self._is_vision(l)]
        merger_layers = [l for l in layers if self._is_merger(l)]
        lang_layers = [l for l in layers if self._is_language(l)]
        
        # Forward count
        n_forwards = n_samples * (1 + n_aug) * 2  # vis + lang
        if self.blank_image_for_lang:
            n_forwards += n_samples
        if self.blank_text_for_vision:
            n_forwards += n_samples
        if self.contrastive_both:
            n_forwards += n_samples * 3 * (1 + n_aug)
        else:
            n_forwards += n_samples * (1 + n_aug)
        
        if verbose:
            print(f"[AutoLayer] {len(layers)} layers ({len(vis_layers)} vision, {len(merger_layers)} merger, {len(lang_layers)} language)")
            print(f"            {n_samples} samples × {n_aug} augs → {n_forwards} forwards")
            if self.verbalize_mode == "replace":
                print(f"            VERBALIZE MODE: replace → <blank, verb(I) + T>")
            elif self.verbalize_mode == "augment":
                print(f"            VERBALIZE MODE: augment → <I, verb(I) + T>")
            if self.blank_text_for_vision:
                if self.verbalize_mode == "replace":
                    print(f"            Vision robustness: <blank, verb(I)> vs <blank, verb(aug(I))>")
                elif self.verbalize_mode == "augment":
                    print(f"            Vision robustness: <I, verb(I)> vs <aug(I), verb(aug(I))>")
                else:
                    print(f"            Vision robustness: <I, \"\"> mode")
            if self.blank_image_for_lang:
                if self.verbalize_mode == "replace":
                    print(f"            Language robustness: <blank, verb(I) + T> vs <blank, verb(I) + aug(T)>")
                elif self.verbalize_mode == "augment":
                    print(f"            Language robustness: <I, verb(I) + T> vs <I, verb(I) + aug(T)>")
                else:
                    print(f"            Language robustness: <blank, T> mode")
            if self.contrastive_both:
                mode_str = "CONTRASTIVE"
                if self.weighted_contrastive:
                    mode_str += " + WEIGHTED [1, 0.5, 0]"
                if self.verbalize_mode != "none":
                    print(f"            Both robustness: {mode_str} (verb_mode={self.verbalize_mode})")
                else:
                    print(f"            Both robustness: {mode_str} (3 sub-clusters: A=<I,T>, B=<I,T'>, C=<I',T>)")
            else:
                print(f"            Both robustness: <I, T> vs <aug(I), aug(T)>")
        
        scores = self.score_layers(dataset, layers, n_samples, n_aug, verbose)
        
        best = {
            "vision_robustness": {
                "overall": self._find_best_in(scores["vision"], layers, metric),
                "vision_layer": self._find_best_in(scores["vision"], vis_layers, metric),
                "merger_layer": self._find_best_in(scores["vision"], merger_layers, metric),
                "language_layer": self._find_best_in(scores["vision"], lang_layers, metric),
            },
            "language_robustness": {
                "overall": self._find_best_in(scores["language"], layers, metric),
                "vision_layer": self._find_best_in(scores["language"], vis_layers, metric),
                "merger_layer": self._find_best_in(scores["language"], merger_layers, metric),
                "language_layer": self._find_best_in(scores["language"], lang_layers, metric),
            },
            "both_robustness": {
                "overall": self._find_best_in(scores["both"], layers, metric),
                "vision_layer": self._find_best_in(scores["both"], vis_layers, metric),
                "merger_layer": self._find_best_in(scores["both"], merger_layers, metric),
                "language_layer": self._find_best_in(scores["both"], lang_layers, metric),
            },
        }
        
        # Add partial sensitivity best layers if contrastive mode
        if self.contrastive_both and "both_text_partial" in scores:
            best["both_text_partial"] = {
                "overall": self._find_best_in(scores["both_text_partial"], layers, metric),
                "vision_layer": self._find_best_in(scores["both_text_partial"], vis_layers, metric),
                "merger_layer": self._find_best_in(scores["both_text_partial"], merger_layers, metric),
                "language_layer": self._find_best_in(scores["both_text_partial"], lang_layers, metric),
            }
            best["both_image_partial"] = {
                "overall": self._find_best_in(scores["both_image_partial"], layers, metric),
                "vision_layer": self._find_best_in(scores["both_image_partial"], vis_layers, metric),
                "merger_layer": self._find_best_in(scores["both_image_partial"], merger_layers, metric),
                "language_layer": self._find_best_in(scores["both_image_partial"], lang_layers, metric),
            }
        
        if verbose:
            print(f"\n{'='*60}")
            for rob_type, bests in best.items():
                # Map rob_type to scores key
                if "vision_rob" in rob_type:
                    mode_key = "vision"
                elif "language_rob" in rob_type:
                    mode_key = "language"
                elif "text_partial" in rob_type:
                    mode_key = "both_text_partial"
                elif "image_partial" in rob_type:
                    mode_key = "both_image_partial"
                else:
                    mode_key = "both"
                print(f"{rob_type}:")
                for group, layer in bests.items():
                    if layer and mode_key in scores:
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
        merger_layers = [l for l in layers if self._is_merger(l)]
        lang_layers = [l for l in layers if self._is_language(l)]
        
        best = {
            "vision_robustness": {
                "overall": self._find_best_in(agg_scores["vision"], layers, metric, is_agg=True),
                "vision_layer": self._find_best_in(agg_scores["vision"], vis_layers, metric, is_agg=True),
                "merger_layer": self._find_best_in(agg_scores["vision"], merger_layers, metric, is_agg=True),
                "language_layer": self._find_best_in(agg_scores["vision"], lang_layers, metric, is_agg=True),
            },
            "language_robustness": {
                "overall": self._find_best_in(agg_scores["language"], layers, metric, is_agg=True),
                "vision_layer": self._find_best_in(agg_scores["language"], vis_layers, metric, is_agg=True),
                "merger_layer": self._find_best_in(agg_scores["language"], merger_layers, metric, is_agg=True),
                "language_layer": self._find_best_in(agg_scores["language"], lang_layers, metric, is_agg=True),
            },
        }
        
        if "both" in agg_scores:
            best["both_robustness"] = {
                "overall": self._find_best_in(agg_scores["both"], layers, metric, is_agg=True),
                "vision_layer": self._find_best_in(agg_scores["both"], vis_layers, metric, is_agg=True),
                "merger_layer": self._find_best_in(agg_scores["both"], merger_layers, metric, is_agg=True),
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

    def plot_scores(self, scores, metric="Q", normalize=False, figsize=None):
        """Line plot of scores with optional error bars (vision=green, merger=orange, language=blue)."""
        import matplotlib.pyplot as plt
        
        metrics = [metric] if isinstance(metric, str) else metric
        all_modes = ["vision", "language", "both", "both_text_partial", "both_image_partial"]
        modes = [m for m in all_modes if m in scores]
        fig, axes = plt.subplots(len(metrics), len(modes), figsize=figsize or (4 * len(modes), 3 * len(metrics)), squeeze=False)
        
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
                is_merger = np.array([self._is_merger(l) for l in layers])
                is_lang = np.array([self._is_language(l) for l in layers])
                
                if normalize:
                    vmin, vmax = vals.min(), vals.max()
                    if vmax - vmin > 1e-8:
                        vals = (vals - vmin) / (vmax - vmin)
                        if stds is not None:
                            stds = stds / (vmax - vmin)
                
                # Plot by group: vision=green, merger=orange, language=blue
                for mask, color, label in [(is_vis, 'green', 'vision'), (is_merger, 'orange', 'merger'), (is_lang, 'blue', 'language')]:
                    idx = indices[mask]
                    if len(idx) > 0:
                        if stds is not None:
                            ax.errorbar(idx, vals[mask], yerr=stds[mask], fmt='o-', ms=3, lw=1, capsize=2, alpha=0.8, color=color, label=label)
                        else:
                            ax.plot(idx, vals[mask], 'o-', ms=3, lw=1, color=color, label=label)
                
                # Mark best overall (red star), best per group (triangles)
                best = np.argmax(vals)
                ax.scatter([indices[best]], [vals[best]], c='red', s=120, zorder=6, marker='*', edgecolors='black')
                
                for mask, color in [(is_vis, 'green'), (is_merger, 'orange'), (is_lang, 'blue')]:
                    idx = indices[mask]
                    if len(idx) > 0:
                        best_idx = idx[np.argmax(vals[mask])]
                        ax.scatter([best_idx], [vals[best_idx]], c=color, s=80, zorder=5, marker='^', edgecolors='black')
                
                ax.set_xlabel("Layer" if row == len(metrics) - 1 else "")
                ax.set_ylabel(f"{m} (↑)" if col == 0 else "")
                # Shorter titles for partial modes
                title_map = {"both_text_partial": "Text Sens.", "both_image_partial": "Image Sens."}
                title = title_map.get(mode, mode.capitalize())
                ax.set_title(f"{title} - {m}" if row == 0 else "")
                ax.legend(fontsize=7, loc='lower right')
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_all_metrics(self, scores, normalize=False, figsize=None):
        """Plot all metrics in a grid with vision=green, merger=orange, language=blue."""
        import matplotlib.pyplot as plt
        
        metrics = ["mse", "dot", "ll", "Q"]
        all_modes = ["vision", "language", "both", "both_text_partial", "both_image_partial"]
        modes = [m for m in all_modes if m in scores]
        title_map = {"both_text_partial": "Text Sens.", "both_image_partial": "Image Sens."}
        
        figsize = figsize or (14, 2.5 * len(modes))
        fig, axes = plt.subplots(len(modes), 4, figsize=figsize, squeeze=False)
        
        for col, metric in enumerate(metrics):
            for row, mode in enumerate(modes):
                mode_scores = scores[mode]
                ax = axes[row, col]
                layers = list(mode_scores.keys())
                
                sample_val = mode_scores[layers[0]][metric]
                is_agg = isinstance(sample_val, dict) and "mean" in sample_val
                
                if is_agg:
                    vals = np.array([mode_scores[l][metric]["mean"] for l in layers])
                    stds = np.array([mode_scores[l][metric]["std"] for l in layers])
                else:
                    vals = np.array([mode_scores[l][metric] for l in layers])
                    stds = None
                
                is_vis = np.array([self._is_vision(l) for l in layers])
                is_merger = np.array([self._is_merger(l) for l in layers])
                is_lang = np.array([self._is_language(l) for l in layers])
                indices = np.arange(len(layers))
                
                if normalize:
                    vmin, vmax = vals.min(), vals.max()
                    if vmax - vmin > 1e-8:
                        vals = (vals - vmin) / (vmax - vmin)
                        if stds is not None:
                            stds = stds / (vmax - vmin)
                
                # Plot by group
                for mask, color, label in [(is_vis, 'green', 'vision'), (is_merger, 'orange', 'merger'), (is_lang, 'blue', 'language')]:
                    idx = indices[mask]
                    if len(idx) > 0:
                        if stds is not None:
                            ax.errorbar(idx, vals[mask], yerr=stds[mask], fmt='o-', ms=3, lw=1, capsize=2, alpha=0.8, color=color, label=label)
                        else:
                            ax.plot(idx, vals[mask], 'o-', ms=3, lw=1, color=color, label=label)
                
                # Mark best overall and per group
                best = np.argmax(vals)
                ax.scatter([indices[best]], [vals[best]], c='red', s=120, zorder=6, marker='*')
                
                for mask, color in [(is_vis, 'green'), (is_merger, 'orange'), (is_lang, 'blue')]:
                    idx = indices[mask]
                    if len(idx) > 0:
                        best_idx = idx[np.argmax(vals[mask])]
                        ax.scatter([best_idx], [vals[best_idx]], c=color, s=80, zorder=5, marker='^')
                
                ax.set_title(f"{metric} (↑)" if row == 0 else "")
                ax.set_xlabel("Layer" if row == len(modes) - 1 else "")
                ax.grid(alpha=0.3)
                if col == 0:
                    ax.set_ylabel(title_map.get(mode, mode.capitalize()))
                if col == 3 and row == 0:
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
        self._verb_cache = {}
