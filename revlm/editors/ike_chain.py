"""IKE_CHAIN: Sentence-Specific Patch-Aware In-Context Knowledge Editing

Expands retrieval surface by creating patch-level keys from images.
Each rationale sentence gets its own patches selected by that sentence.

Key structure: [<image/patch, text>, value]
- Original image: (1 + n) keys for question + n sentences
- Sentence-specific patches: n × (up to k) keys
- Total per edit: (1 + n) + n × k keys max
"""

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Tuple, Optional

from .utils import brackets_to_periods, parent_module, Augmenter, ImagePatchifier


class IKE_CHAIN(nn.Module):
    """Sentence-specific patch-aware codebook for VLM editing.
    
    Codebook entry: [key_emb, value, radius]
    - key_emb: vision_layer(<image/patch, text>) embedding
    - value: sentence to retrieve
    - radius: percentile of augmented distances
    
    Edit structure (for n sentences, k patches):
    - (1+n) keys from original image: <orig, question> + <orig, si> for each si
    - n×k keys from sentence-specific patches: <patches_si, si> for each si
    
    Query: patchify query image, check query embeddings against codebook.
    """

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        cfg = getattr(config, "editor", config)

        # ==================== Model References ====================
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))

        # ==================== Core Retrieval ====================
        self.cap_k = int(getattr(cfg, "cap_k", 3))                              # max keys to retrieve
        self.retrieve_single_edit = getattr(cfg, "retrieve_single_edit", False) # only retrieve from winning edit
        self.prefix = getattr(cfg, "cot_prefix", "")                            # prefix for retrieved facts
        self.seed = getattr(cfg, "seed", None)

        # ==================== Embedding Config ====================
        self.distance = getattr(cfg, "distance", "l2")              # "l2" or "cosine"
        self.dual_layer = getattr(cfg, "dual_layer", True)          # concat vision + lang embeddings
        self.pool_method = getattr(cfg, "pool_method", "mean")      # "mean" or "last"
        self.lang_encoder = getattr(cfg, "lang_encoder", "sbert")   # "internal" or "sbert"
        self.lang_scaler = float(getattr(config.model, "lang_scaler_sbert", 30.0)) if self.lang_encoder == "sbert" else float(getattr(config.model, "lang_scaler", 30.0))
        
        # ==================== Radius Estimation ====================
        self.radius_method = getattr(cfg, "radius_method", "augment")  # "fixed", "augment", or "balance"
        self.fixed_radius = float(getattr(cfg, "fixed_radius", 1000.0))
        # augment method
        self.radius_area_pct = float(getattr(cfg, "radius_area_pct", 0.5))
        self.radius_scaler = float(getattr(cfg, "radius_scaler", 1.0))
        self.n_radius_samples = int(getattr(cfg, "n_radius_samples", 3))
        self.radius_percentile = float(getattr(cfg, "radius_percentile", 50))
        # balance method
        self.n_positive_samples = int(getattr(cfg, "n_positive_samples", 5))
        self.balance_alpha = float(getattr(cfg, "balance_alpha", 0.5))

        # ==================== Patchification ====================
        self.top_k_patches = int(getattr(cfg, "top_k_patches", 3))  # patches to select per edit
        self.query_kernels = getattr(cfg, "query_kernels", ['1x1', '2x2', '3x3'])
        self.patch_select_prompt = getattr(cfg, "patch_select_prompt", "Describe this image.")

        # ==================== Key Management ====================
        self.merge_keys = getattr(cfg, "merge_keys", False)  # enable both merge methods
        self.merge_ioa_threshold = float(getattr(cfg, "merge_ioa_threshold", 0.9))  # IoA merge threshold
        self.merge_dist_pct = float(getattr(cfg, "merge_dist_pct", 0.1))  # distance merge: if dist < pct * both radii

        # ==================== Image-Only Fallback ====================
        self.image_only_retrieval = getattr(cfg, "image_only_retrieval", False)
        self.top_i_image_only = int(getattr(cfg, "top_i_image_only_entry", 1))

        # ==================== Internal State ====================
        self._added_uids = set()
        self._edit_count = 0
        self._vision_act = None
        self._lang_act = None
        self._sbert = None
        self._blank_image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        # Codebook storage
        self.codebook = []
        self.key_embs = None   # [N, hidden]
        self.key_radii = None  # [N]
        
        # Logging
        self.last_retrieval_log = None
        self.plot_codebook_pct_threshold = 85

        # ==================== Setup Hooks & Tools ====================
        self.patchifier = ImagePatchifier()
        dataset_name = getattr(getattr(config, "experiment", None), "dataset_name", None)
        self.augmenter = Augmenter(self.wrapper, seed=self.seed, mosaic_prob=1.0, dataset_name=dataset_name)
        
        # VLM activation hooks
        model_cfg = getattr(config, "model", config)
        inner_params_vision = getattr(model_cfg, "inner_params_vision", [])
        inner_params_lang = getattr(model_cfg, "inner_params_lang", [])
        if not inner_params_vision:
            raise ValueError("Requires config.model.inner_params_vision")
        if self.dual_layer and self.lang_encoder == "internal" and not inner_params_lang:
            raise ValueError("dual_layer=True with lang_encoder='internal' requires config.model.inner_params_lang")
        
        def _setup_hook(param_name, attr_name):
            name = param_name.rsplit(".", 1)[0] if param_name.endswith((".weight", ".bias")) else param_name
            mod = parent_module(self.model, brackets_to_periods(name))
            layer = getattr(mod, name.rsplit(".", 1)[-1])
            return layer.register_forward_hook(
                lambda m, i, o, an=attr_name: setattr(self, an, i[0].detach() if isinstance(i[0], torch.Tensor) else None)
            )
        
        self._vision_hook = _setup_hook(inner_params_vision[0], "_vision_act")
        self._lang_hook = None
        if self.dual_layer and self.lang_encoder == "internal":
            self._lang_hook = _setup_hook(inner_params_lang[0], "_lang_act")

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        return (self.model if hasattr(self.model, "generate") else self.wrapper).generate(*a, **kw)

    def _pool_act(self, act, batch_size):
        """Pool activation to [B, hidden] shape.
        
        pool_method:
        - "mean": average over sequence dimension (default)
        - "last": use last token position
        """
        if act is None:
            raise RuntimeError("Hook failed to capture activation")
        act = act.to(self.device, torch.float32)
        if act.dim() == 3:
            if self.pool_method == "last":
                return act[:, -1, :]
            else:
                return act.mean(dim=1)
        elif act.dim() == 2:
            if act.shape[0] == batch_size:
                return act
            elif act.shape[0] % batch_size == 0:
                patches = act.shape[0] // batch_size
                if self.pool_method == "last":
                    return act.view(batch_size, patches, -1)[:, -1, :]
                else:
                    return act.view(batch_size, patches, -1).mean(dim=1)
            else:
                if self.pool_method == "last":
                    return act[-1:].expand(batch_size, -1)
                else:
                    return act.mean(dim=0, keepdim=True).expand(batch_size, -1)
        else:
            raise RuntimeError(f"Expected 2D or 3D activation, got {act.shape}")

    def _get_sbert(self):
        """Lazy load sentence-transformers model."""
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
            self._sbert.to(self.device)
        return self._sbert

    def _encode_sbert(self, texts: List[str]) -> torch.Tensor:
        """Get sentence embeddings from SBERT. Returns [B, 768]."""
        sbert = self._get_sbert()
        emb = sbert.encode(texts, convert_to_tensor=True)
        return emb.to(self.device, torch.float32)

    @torch.no_grad()
    def _encode_vlm(self, images: List, texts: List[str]) -> torch.Tensor:
        """Get VLM embedding for <image, text> pairs.
        
        If dual_layer=True:
          - internal: concat(vision(<img,text>), lang(<blank,text>))
          - sbert: concat(vision(<img,text>), sbert(text))
        Returns: [B, hidden] or [B, hidden+lang_dim] tensor
        """
        self.model.eval()
        batch_size = len(images) if isinstance(images, list) else 1
        
        # Pass 1: <image, text> -> vision embedding
        self._vision_act = None
        inputs = self.wrapper.encode(images, texts, tokenize=False)
        self.model(**inputs)
        vision_emb = self._pool_act(self._vision_act, batch_size)
        self._vision_act = None
        
        if not self.dual_layer:
            return vision_emb
        
        # Pass 2: language embedding (internal VLM layer or SBERT)
        if self.lang_encoder == "sbert":
            lang_emb = self._encode_sbert(texts) * self.lang_scaler
        else:
            self._lang_act = None
            blank_imgs = [self._blank_image] * batch_size
            inputs = self.wrapper.encode(blank_imgs, texts, tokenize=False)
            self.model(**inputs)
            lang_emb = self._pool_act(self._lang_act, batch_size) * self.lang_scaler
            self._lang_act = None
        
        return torch.cat([vision_emb, lang_emb], dim=-1)

    @torch.no_grad()
    def _get_nll(self, image, prompt: str, label: str) -> float:
        """Get negative log-likelihood of label given <image, prompt>."""
        if hasattr(self.wrapper, 'get_loss_y'):
            avg_nll, _, _ = self.wrapper.get_loss_y(image, prompt, label)
            return avg_nll
        
        # Fallback: manual NLL computation
        inputs = self.wrapper.encode([image], [prompt], tokenize=False)
        label_ids = self.wrapper.tokenizer(
            label, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        
        input_ids = inputs["input_ids"]
        full_ids = torch.cat([input_ids, label_ids], dim=1)
        labels = torch.full_like(full_ids, -100)
        labels[:, input_ids.size(1):] = full_ids[:, input_ids.size(1):]
        
        inputs["input_ids"] = full_ids
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat([
                inputs["attention_mask"], torch.ones_like(label_ids)
            ], dim=1)
        inputs["labels"] = labels
        
        out = self.model(**inputs)
        return float(out.loss.item())

    @torch.no_grad()
    def _select_top_k_patches(self, image, sentence: str, is_first_sentence: bool = True) -> List[Image.Image]:
        """Select up to k patches by log-likelihood of sentence, then VQA-verify.
        
        Args:
            sentence: The rationale sentence to select patches for
            is_first_sentence: If True, VQA prompt is "Does {s}?"; else "Does the image show {s}?"
        """
        patches = self.patchifier.patchify_exclude_full(image)  # 35 patches
        
        # Stage 1: Rank by NLL of sentence
        nlls = []
        for patch in patches:
            nll = self._get_nll(patch, self.patch_select_prompt, sentence)
            nlls.append(nll)
        
        nlls = np.array(nlls)
        top_k_idx = np.argsort(nlls)[:self.top_k_patches]
        candidates = [patches[i] for i in top_k_idx]
        
        # Stage 2: VQA verification
        if is_first_sentence:
            vqa_question = f"Does {sentence.lower().replace('.', '?')}"
        else:
            vqa_question = f"Does the image show {sentence.lower().replace('.', '?')}"
        
        verified = []
        yes_probs = []
        
        for patch in candidates:
            nll_yes = self._get_nll(patch, vqa_question, "Yes")
            nll_no = self._get_nll(patch, vqa_question, "No")
            p_yes = 1.0 / (1.0 + np.exp(nll_yes - nll_no))
            yes_probs.append(p_yes)
            if p_yes > 0.5:
                verified.append(patch)
        
        # Fallback: if none passed, keep highest P("yes")
        if not verified:
            best_idx = int(np.argmax(yes_probs))
            verified = [candidates[best_idx]]
        
        # Clear cache after many forward passes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return verified

    @torch.no_grad()
    def _estimate_radius_balance(self, key_emb: torch.Tensor, img, text: str, is_question: bool = True) -> float:
        """Estimate radius via positive (augmented image+text) and negative (blank image) samples."""
        # Positive: augmented image + augmented text
        pos_dists = []
        for _ in range(self.n_positive_samples):
            aug_img = self.augmenter.image(img)
            aug_text = self.augmenter.question(text) if is_question else self.augmenter.rationale(text) if text else ""
            pos_emb = self._encode_vlm([aug_img], [aug_text])
            pos_dists.append(float(torch.norm(pos_emb.cpu() - key_emb.cpu())))
        d_pos = float(np.median(pos_dists)) if pos_dists else 0.0
        # Negative: blank image, same text
        neg_emb = self._encode_vlm([self._blank_image], [text])
        d_neg = float(torch.norm(neg_emb.cpu() - key_emb.cpu()))
        # Combined: ε = (1 - α) * d(Pos, k) + α * d(Neg, k)
        return (1 - self.balance_alpha) * d_pos + self.balance_alpha * d_neg

    @torch.no_grad()
    def _estimate_radius(self, key_emb: torch.Tensor, img, text: str, is_question: bool = True) -> float:
        """Estimate radius.
        
        Methods:
        - 'fixed': constant radius (fastest, no forward pass)
        - 'augment': percentile of n augmented samples (n forward passes)
        - 'balance': positive (augmented) and negative (blank image) samples
        
        Args:
            is_question: True for question text (rephrase), False for rationale (turn into question)
        """
        if self.radius_method == "fixed":
            return self.fixed_radius * self.radius_scaler
        
        if self.radius_method == "balance":
            return self._estimate_radius_balance(key_emb, img, text, is_question) * self.radius_scaler
        
        # Default: multiple augmentations, take percentile
        aug_dists = []
        for _ in range(self.n_radius_samples):
            aug_img = self.augmenter.image(img, area_pct=self.radius_area_pct)
            aug_text = self.augmenter.question(text) if is_question else self.augmenter.rationale(text) if text else ""
            aug_emb = self._encode_vlm([aug_img], [aug_text])
            dist = float(torch.norm(aug_emb.cpu() - key_emb.cpu()))
            aug_dists.append(dist)
        
        return float(np.percentile(aug_dists, self.radius_percentile)) * self.radius_scaler

    def _circle_intersection(self, d: float, r1: float, r2: float) -> float:
        """Compute intersection area of two circles."""
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return np.pi * min(r1, r2) ** 2
        # Lens formula
        part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        part3 = 0.5 * np.sqrt((r1+r2-d) * (d+r1-r2) * (d-r1+r2) * (d+r1+r2))
        return part1 + part2 - part3

    def _circle_ioa_pair(self, d: float, r1: float, r2: float) -> Tuple[float, float]:
        """Compute IoA pair: (intersection/area1, intersection/area2)."""
        intersection = self._circle_intersection(d, r1, r2)
        area1 = np.pi * r1 ** 2
        area2 = np.pi * r2 ** 2
        return (intersection / area1 if area1 > 0 else 0.0,
                intersection / area2 if area2 > 0 else 0.0)

    def _effective_distances(self, indices: List[int] = None) -> np.ndarray:
        """Compute effective pairwise distances: max(0, d(ki,kj) - (ri + rj)).
        
        Returns [N, N] matrix where 0 means circles overlap/touch.
        """
        if indices is None:
            embs = self.key_embs.float().cpu().numpy()
            radii = self.key_radii.cpu().numpy()
        else:
            embs = self.key_embs[indices].float().cpu().numpy()
            radii = self.key_radii[indices].cpu().numpy()
        
        dists = np.linalg.norm(embs[:, None] - embs[None, :], axis=-1)
        radii_sum = radii[:, None] + radii[None, :]
        return np.maximum(0, dists - radii_sum)

    def _manage_new_key(self, emb: torch.Tensor, radius: float, value: str) -> Tuple[bool, float]:
        """Manage new key: merge or add as-is.
        
        Returns: (merged: bool, final_radius: float)
        - merged=True: key was merged, don't add
        - merged=False: add key with final_radius
        """
        if self.key_embs is None or len(self.codebook) == 0:
            return False, radius
        
        emb_cpu = emb.cpu().squeeze(0) if emb.dim() > 1 else emb.cpu()
        dists = torch.norm(self.key_embs - emb_cpu, dim=1).numpy()
        radii = self.key_radii.numpy()
        
        # Find overlapping keys
        overlapping = np.where(dists < radius + radii)[0]
        if len(overlapping) == 0:
            return False, radius
        
        # Step 1a: Distance-based merge (centers very close)
        if self.merge_keys:
            for idx in overlapping:
                d = dists[idx]
                if d < self.merge_dist_pct * radius and d < self.merge_dist_pct * radii[idx]:
                    self.key_radii[idx] = max(float(self.key_radii[idx]), d + radius)
                    existing_value = self.codebook[idx].get("value", "")
                    if value and value != existing_value and value not in existing_value:
                        self.codebook[idx]["value"] = f"{existing_value} {value}".strip()
                    self.codebook[idx]["is_merged"] = True
                    self.codebook[idx]["merge_count"] = self.codebook[idx].get("merge_count", 1) + 1
                    return True, radius
        
        # Step 1b: IoA-based merge (both IoA > threshold)
        if self.merge_keys:
            for idx in overlapping:
                ioa_new, ioa_old = self._circle_ioa_pair(dists[idx], radius, radii[idx])
                if ioa_new > self.merge_ioa_threshold and ioa_old > self.merge_ioa_threshold:
                    dist = float(dists[idx])
                    self.key_radii[idx] = max(float(self.key_radii[idx]), dist + radius)
                    existing_value = self.codebook[idx].get("value", "")
                    if value and value != existing_value and value not in existing_value:
                        self.codebook[idx]["value"] = f"{existing_value} {value}".strip()
                    self.codebook[idx]["is_merged"] = True
                    self.codebook[idx]["merge_count"] = self.codebook[idx].get("merge_count", 1) + 1
                    return True, radius
        
        return False, radius

    @torch.no_grad()
    def _add_edit(self, img, question: str, answer: str, rationale_sents: List[str]):
        """Add keys for one edit.
        
        Creates:
        - (1+n) keys from original image: <orig, question> + <orig, si>
        - n×k keys from sentence-specific patches: <patches_si, si>
        """
        answer_value = f"The answer to '{question}' is {answer}." if answer else ""
        
        new_entries = []
        new_imgs = []
        new_texts = []
        new_is_question = []
        
        # 1. Original image keys
        # <orig, question> -> answer
        new_entries.append({
            "value": answer_value, "is_patch": False, "edit_idx": self._edit_count,
            "key_text": question, "is_image_only": False, "is_question": True
        })
        new_imgs.append(img)
        new_texts.append(question)
        new_is_question.append(True)
        
        # <orig, si> -> si for each sentence
        for sent in rationale_sents:
            new_entries.append({
                "value": sent, "is_patch": False, "edit_idx": self._edit_count,
                "key_text": sent, "is_image_only": False, "is_question": False
            })
            new_imgs.append(img)
            new_texts.append(sent)
            new_is_question.append(False)
        
        # Optional: <orig, ""> for image-only retrieval
        if self.image_only_retrieval:
            new_entries.append({
                "value": "", "is_patch": False, "edit_idx": self._edit_count,
                "key_text": "", "is_image_only": True, "is_question": False
            })
            new_imgs.append(img)
            new_texts.append("")
            new_is_question.append(False)
        
        # 2. Sentence-specific patch keys: <patches_si, si> -> si
        for i, sent in enumerate(rationale_sents):
            is_first = (i == 0)
            patches = self._select_top_k_patches(img, sent, is_first_sentence=is_first)
            for patch in patches:
                new_entries.append({
                    "value": sent, "is_patch": True, "edit_idx": self._edit_count,
                    "key_text": sent, "is_image_only": False, "is_question": False
                })
                new_imgs.append(patch)
                new_texts.append(sent)
                new_is_question.append(False)
        
        self._edit_count += 1
        
        # Compute embeddings
        new_embs = self._encode_vlm(new_imgs, new_texts)
        if self.distance == "cosine":
            new_embs = F.normalize(new_embs, dim=-1)
        
        # Compute radii per key
        new_radii = []
        for i, (src_img, text, is_q) in enumerate(zip(new_imgs, new_texts, new_is_question)):
            if not text:  # image-only key
                r = self._estimate_radius(new_embs[i:i+1], src_img, question, is_question=True)
            else:
                r = self._estimate_radius(new_embs[i:i+1], src_img, text, is_question=is_q)
            new_radii.append(r)
        new_radii = torch.tensor(new_radii, dtype=torch.float32)
        
        # Move to CPU to save GPU memory
        new_embs = new_embs.cpu()
        
        # Add keys to codebook (with optional key management)
        n_merged = 0
        n_added = 0
        for i, entry in enumerate(new_entries):
            emb, radius = new_embs[i:i+1], float(new_radii[i])
            value = entry.get("value", "")
            
            if self.merge_keys:
                merged, radius = self._manage_new_key(emb, radius, value)
                if merged:
                    n_merged += 1
                    continue
            
            n_added += 1
            entry["original_radius"] = radius  # track original for hard budget
            self.codebook.append(entry)
            if self.key_embs is None:
                self.key_embs = emb
                self.key_radii = torch.tensor([radius])
            else:
                self.key_embs = torch.cat([self.key_embs, emb], dim=0)
                self.key_radii = torch.cat([self.key_radii, torch.tensor([radius])])
        
        print(f"[Keys] +{n_added} added, {n_merged} merged (from {len(new_entries)} candidates)")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _retrieve(self, image, question: str) -> List[str]:
        """Retrieve values for a query <image, question>.
        
        Stage 1: Text-aware matching against <image, text> keys
        Stage 2 (if enabled): Image-only fallback with <image, ""> keys  
        Stage 3: Re-match with candidate edit's texts
        """
        if self.key_embs is None or len(self.codebook) == 0:
            return []
        
        # Patchify query image
        query_patches = self.patchifier.patchify(image, kernels=self.query_kernels)
        
        # Stage 1: Text-aware matching
        results = self._retrieve_text_aware(query_patches, question)
        if results:
            return results
        
        # Stage 2 & 3: Image-only fallback (only if enabled)
        if self.image_only_retrieval:
            return self._retrieve_image_only_fallback(query_patches, question)
        
        return []

    @torch.no_grad()
    def _retrieve_text_aware(self, query_patches: List, question: str, key_indices: List[int] = None) -> List[str]:
        """Stage 1: Retrieve against text keys (non-image-only).
        
        Args:
            query_patches: Patchified query images
            question: Query text
            key_indices: Optional subset of key indices to check (None = all non-image-only)
        """
        # Build query embeddings
        q_embs = self._encode_vlm(query_patches, [question] * len(query_patches))
        if self.distance == "cosine":
            q_embs = F.normalize(q_embs, dim=-1)
        q_embs = q_embs.cpu()
        
        # Filter to text keys only (or specified subset)
        if key_indices is None:
            key_indices = [i for i, e in enumerate(self.codebook) if not e.get("is_image_only", False)]
        if not key_indices:
            return []
        
        key_idx_t = torch.tensor(key_indices)
        key_embs = self.key_embs[key_idx_t]
        key_radii = self.key_radii[key_idx_t]
        
        # Compute distances
        if self.distance == "cosine":
            dist_matrix = 1 - (q_embs @ key_embs.t())
        else:
            dist_matrix = torch.cdist(q_embs.float(), key_embs.float(), p=2)
        
        # Step 1: Find all keys within radius
        in_radius = dist_matrix <= key_radii
        matched_mask = in_radius.any(dim=0)
        
        if not matched_mask.any():
            return []
        
        # Get matched indices and their min distances
        matched_local = torch.where(matched_mask)[0]
        min_dists = dist_matrix[:, matched_local].min(dim=0).values
        
        # Step 2 & 3: Single-edit mode - only keep keys from winning edit
        if self.retrieve_single_edit and len(matched_local) > 0:
            # Group matched keys by edit_idx
            edit_keys = {}       # edit_idx -> list of (local_idx, dist)
            for local_i, dist in zip(matched_local.tolist(), min_dists.tolist()):
                global_i = key_indices[local_i]
                edit_idx = self.codebook[global_i]["edit_idx"]
                edit_keys.setdefault(edit_idx, []).append((local_i, dist))
            
            # Find winning edit: most keys, tie-break by closest key
            winning_edit = max(
                edit_keys.keys(),
                key=lambda e: (len(edit_keys[e]), -min(d for _, d in edit_keys[e]))
            )
            
            # Filter to only winning edit's keys
            winning_pairs = edit_keys[winning_edit]
            matched_local = torch.tensor([p[0] for p in winning_pairs])
            min_dists = torch.tensor([p[1] for p in winning_pairs])
        
        # Sort by distance, take top cap_k
        sorted_order = min_dists.argsort()
        selected_local = matched_local[sorted_order].tolist()[:self.cap_k]
        
        # Collect values
        retrieved = set()
        for local_i in selected_local:
            global_i = key_indices[local_i]
            value = self.codebook[global_i]["value"]
            if value:
                retrieved.add(value)
        
        return list(retrieved)

    @torch.no_grad()
    def _retrieve_image_only_fallback(self, query_patches: List, question: str) -> List[str]:
        """Stage 2 & 3: Image-only gate then text re-matching."""
        # Stage 2: Query <patches, ""> against <src, ""> keys
        img_only_indices = [i for i, e in enumerate(self.codebook) if e.get("is_image_only", False)]
        if not img_only_indices:
            return []
        
        # Encode query with empty text
        q_embs = self._encode_vlm(query_patches, [""] * len(query_patches))
        if self.distance == "cosine":
            q_embs = F.normalize(q_embs, dim=-1)
        q_embs = q_embs.cpu()
        
        # Get image-only key embeddings
        img_idx_t = torch.tensor(img_only_indices)
        img_key_embs = self.key_embs[img_idx_t]
        img_key_radii = self.key_radii[img_idx_t]
        
        # Compute distances to image-only keys
        if self.distance == "cosine":
            dist_matrix = 1 - (q_embs @ img_key_embs.t())
        else:
            dist_matrix = torch.cdist(q_embs.float(), img_key_embs.float(), p=2)
        
        # Check which image-only keys match
        in_radius = dist_matrix <= img_key_radii
        matched_mask = in_radius.any(dim=0)
        
        if not matched_mask.any():
            return []
        
        # Get top_i closest image-only matches
        matched_local = torch.where(matched_mask)[0]
        min_dists = dist_matrix[:, matched_local].min(dim=0).values
        sorted_order = min_dists.argsort()
        top_i_local = matched_local[sorted_order][:self.top_i_image_only].tolist()
        
        # Get edit_idx for each matched image-only key
        matched_edit_ids = set()
        for local_i in top_i_local:
            global_i = img_only_indices[local_i]
            matched_edit_ids.add(self.codebook[global_i]["edit_idx"])
        
        # Stage 3: For each matched edit, try text re-matching
        retrieved = set()
        for edit_idx in matched_edit_ids:
            # Get all text keys from this edit
            edit_text_keys = [i for i, e in enumerate(self.codebook) 
                            if e.get("edit_idx") == edit_idx and not e.get("is_image_only", False)]
            if not edit_text_keys:
                continue
            
            # Get the texts from this edit
            edit_texts = list(set(self.codebook[i]["key_text"] for i in edit_text_keys))
            
            # Try each text from the edit against its keys
            for text in edit_texts:
                # Get keys for this specific text
                text_key_indices = [i for i in edit_text_keys if self.codebook[i]["key_text"] == text]
                results = self._retrieve_text_aware(query_patches, text, key_indices=text_key_indices)
                retrieved.update(results)
        
        return list(retrieved)

    @torch.no_grad()
    def _get_matched_indices(self, image, question: str, apply_cap_k: bool = True) -> Tuple[set, set]:
        """Get matched key indices for plotting. Returns (text_matched, img_fallback_matched)."""
        text_matched, img_matched = set(), set()
        if self.key_embs is None or len(self.codebook) == 0:
            return text_matched, img_matched
        
        query_patches = self.patchifier.patchify(image, kernels=self.query_kernels)
        
        # Stage 1: Text-aware
        text_indices = [i for i, e in enumerate(self.codebook) if not e.get("is_image_only", False)]
        if text_indices:
            q_embs = self._encode_vlm(query_patches, [question] * len(query_patches)).cpu()
            if self.distance == "cosine":
                q_embs = F.normalize(q_embs, dim=-1)
            idx_t = torch.tensor(text_indices)
            if self.distance == "cosine":
                dm = 1 - (q_embs @ self.key_embs[idx_t].t())
            else:
                dm = torch.cdist(q_embs.float(), self.key_embs[idx_t].float(), p=2)
            matched = dm <= self.key_radii[idx_t]
            matched_local = torch.where(matched.any(dim=0))[0]
            if matched_local.numel() > 0:
                min_dists = dm[:, matched_local].min(dim=0).values
                
                # Single-edit mode: filter to winning edit
                if self.retrieve_single_edit:
                    edit_keys = {}
                    for local_i, dist in zip(matched_local.tolist(), min_dists.tolist()):
                        global_i = text_indices[local_i]
                        edit_idx = self.codebook[global_i]["edit_idx"]
                        edit_keys.setdefault(edit_idx, []).append((local_i, dist))
                    if edit_keys:
                        winning_edit = max(edit_keys.keys(), key=lambda e: (len(edit_keys[e]), -min(d for _, d in edit_keys[e])))
                        winning_pairs = edit_keys[winning_edit]
                        matched_local = torch.tensor([p[0] for p in winning_pairs])
                        min_dists = torch.tensor([p[1] for p in winning_pairs])
                
                top_k = matched_local[min_dists.argsort()]
                if apply_cap_k:
                    top_k = top_k[:self.cap_k]
                text_matched = set(text_indices[i.item()] for i in top_k)
        
        if text_matched or not self.image_only_retrieval:
            return text_matched, img_matched
        
        # Stage 2+3: Image-only fallback
        img_only_idx = [i for i, e in enumerate(self.codebook) if e.get("is_image_only", False)]
        if not img_only_idx:
            return text_matched, img_matched
        
        q_empty = self._encode_vlm(query_patches, [""] * len(query_patches)).cpu()
        if self.distance == "cosine":
            q_empty = F.normalize(q_empty, dim=-1)
        idx_t = torch.tensor(img_only_idx)
        if self.distance == "cosine":
            dm = 1 - (q_empty @ self.key_embs[idx_t].t())
        else:
            dm = torch.cdist(q_empty.float(), self.key_embs[idx_t].float(), p=2)
        matched = dm <= self.key_radii[idx_t]
        if not matched.any():
            return text_matched, img_matched
        
        # Top-i edits
        matched_local = torch.where(matched.any(dim=0))[0]
        min_d = dm[:, matched_local].min(dim=0).values
        top_local = matched_local[min_d.argsort()][:self.top_i_image_only].tolist()
        edit_ids = set(self.codebook[img_only_idx[i]]["edit_idx"] for i in top_local)
        
        # Stage 3: Re-match
        for eid in edit_ids:
            edit_keys = [i for i, e in enumerate(self.codebook) if e.get("edit_idx") == eid and not e.get("is_image_only", False)]
            for text in set(self.codebook[i]["key_text"] for i in edit_keys):
                t_idx = [i for i in edit_keys if self.codebook[i]["key_text"] == text]
                q_t = self._encode_vlm(query_patches, [text] * len(query_patches)).cpu()
                if self.distance == "cosine":
                    q_t = F.normalize(q_t, dim=-1)
                idx_t = torch.tensor(t_idx)
                if self.distance == "cosine":
                    dm = 1 - (q_t @ self.key_embs[idx_t].t())
                else:
                    dm = torch.cdist(q_t.float(), self.key_embs[idx_t].float(), p=2)
                matched = dm <= self.key_radii[idx_t]
                matched_local = torch.where(matched.any(dim=0))[0]
                if matched_local.numel() > 0:
                    min_dists = dm[:, matched_local].min(dim=0).values
                    top_k = matched_local[min_dists.argsort()]
                    if apply_cap_k:
                        top_k = top_k[:self.cap_k]
                    img_matched.update(t_idx[i.item()] for i in top_k)
        
        return text_matched, img_matched

    def apply_to_dataset(self, dataset):
        """Apply retrieved facts to dataset prompts."""
        applied = 0
        log = []
        data = getattr(dataset, "data", [])
        
        for ex in data:
            prompt_orig = ex.get("prompt_orig") or ex.get("prompt", "")
            q, img = ex.get("question", ""), ex.get("image")
            if not prompt_orig or img is None:
                continue
            
            facts = self._retrieve(img, q) if q else []
            
            if facts:
                ex["prompt_orig"] = prompt_orig  # Save original
                ex["prompt"] = f"{self.prefix}{' '.join(facts)} {prompt_orig}"
                applied += 1
            else:
                ex["prompt"] = prompt_orig  # Reset to original if no facts
            
            log.append({"uid": ex.get("uid"), "n_facts": len(facts), "facts": facts})
        
        self.last_retrieval_log = log
        print(f"[IKE_PATCH] applied facts to {applied}/{len(data)} examples", flush=True)

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add edits to codebook."""
        if edit_ds is None:
            return self.model
        
        n_before = len(self.codebook)
        added = 0
        
        # Filter valid examples first
        valid_exs = []
        for ex in getattr(edit_ds, "data", []):
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            if uid in self._added_uids:
                continue
            rat = ex.get("cot") or ex.get("rationale") or ""
            if not rat or ex.get("image") is None:
                continue
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", rat.strip()) if s.strip()]
            if sents:
                valid_exs.append((ex, sents, uid))
        
        total = len(valid_exs)
        for i, (ex, sents, uid) in enumerate(valid_exs):
            print(f"\r[IKE_PATCH] edit {i+1}/{total}...", end="", flush=True)
            self._add_edit(ex.get("image"), ex.get("question", ""), 
                          ex.get("answer") or ex.get("target") or "", sents)
            self._added_uids.add(uid)
            added += 1
        
        n_after = len(self.codebook)
        mem_mb = self.key_embs.numel() * 2 / 1024 / 1024 if self.key_embs is not None else 0
        if self.radius_method == "fixed":
            r_info = f"fixed={self.fixed_radius}"
        elif self.radius_method == "balance":
            r_info = f"balance(n={self.n_positive_samples},α={self.balance_alpha})"
        else:
            r_info = f"augment(n={self.n_radius_samples})"
        print(f"[IKE_PATCH] +{added} edits (k={self.top_k_patches}, r={r_info}), {n_before}->{n_after} keys, {mem_mb:.1f} MB", flush=True)
        
        self.apply_to_dataset(edit_ds)
        return self.model
    
    def save_index(self, path):
        """Save codebook, embeddings, and radii to disk."""
        data = {
            "codebook": self.codebook,
            "key_embs": self.key_embs,
            "key_radii": self.key_radii,
            "top_k_patches": self.top_k_patches,
        }
        torch.save(data, path)
        print(f"[IKE_PATCH] saved {len(self.codebook)} keys to {path}", flush=True)
    
    def load_index(self, path):
        """Load codebook, embeddings, and radii from disk."""
        data = torch.load(path, map_location=self.device)
        self.codebook = data["codebook"]
        self.key_embs = data["key_embs"].to(self.device)
        self.key_radii = data["key_radii"].to(self.device)
        print(f"[IKE_PATCH] loaded {len(self.codebook)} keys from {path}", flush=True)

    def get_stats(self) -> Dict:
        """Return statistics about stored keys."""
        n_patch_keys = sum(1 for e in self.codebook if e.get("is_patch", False))
        n_orig_keys = len(self.codebook) - n_patch_keys
        n_image_only = sum(1 for e in self.codebook if e.get("is_image_only", False))
        n_question_keys = sum(1 for e in self.codebook if e.get("is_question", False))
        n_rationale_keys = sum(1 for e in self.codebook if not e.get("is_question", True) and not e.get("is_image_only", False))
        n_merged_keys = sum(1 for e in self.codebook if e.get("is_merged", False))
        
        stats = {
            "num_keys": len(self.codebook),
            "num_orig_keys": n_orig_keys,
            "num_patch_keys": n_patch_keys,
            "num_merged_keys": n_merged_keys,
            "num_question_keys": n_question_keys,
            "num_rationale_keys": n_rationale_keys,
            "num_image_only_keys": n_image_only,
            "num_edits": len(self._added_uids),
            "top_k_patches": self.top_k_patches,
            "merge_keys": self.merge_keys,
            "retrieve_single_edit": self.retrieve_single_edit,
            "image_only_retrieval": self.image_only_retrieval,
            "emb_size_mb": self.key_embs.numel() * 2 / 1024 / 1024 if self.key_embs is not None else 0,
        }
        if self.key_radii is not None:
            stats["avg_radius"] = float(self.key_radii.mean())
            stats["min_radius"] = float(self.key_radii.min())
            stats["max_radius"] = float(self.key_radii.max())
        return stats

    @torch.no_grad()
    def visualize_patches(self, image, sentence: str = None, is_first_sentence: bool = True, 
                          figsize=(16, 10), score_type="softmax"):
        """Visualize patchification with scores and top-k highlighted.
        
        Args:
            image: Input image
            sentence: Sentence for patch selection (if None, shows all patches without scores)
            is_first_sentence: If True, VQA prompt is "Does {s}?"; else "Does the image show {s}?"
            figsize: Figure size
            score_type: "softmax" (default, probabilities sum to 1) or "ll" (raw log-likelihood)
        
        Green border = passed VQA verification (p_yes > 0.9)
        Red border = in top-k but failed VQA
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from scipy.special import softmax
        
        patches = self.patchifier.patchify(image)
        patch_names = self.patchifier.get_patch_names()
        n_patches = len(patches)
        
        n_cols = 6
        n_rows = (n_patches + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        scores = None
        top_k_idx = []
        vqa_passed = set()
        score_label = ""
        if sentence:
            nlls = []
            for patch in patches[:-1]:
                nll = self._get_nll(patch, self.patch_select_prompt, sentence)
                nlls.append(nll)
            nlls = np.array(nlls)
            
            if score_type == "softmax":
                scores = softmax(-nlls)
                score_label = "P"
                top_k_idx = np.argsort(scores)[::-1][:self.top_k_patches].tolist()
            else:
                scores = -nlls
                score_label = "LL"
                top_k_idx = np.argsort(scores)[::-1][:self.top_k_patches].tolist()
            
            # VQA verification with sentence-appropriate prompt
            if is_first_sentence:
                vqa_question = f"Does {sentence.lower().replace('.', '?')}"
            else:
                vqa_question = f"Does the image show {sentence.lower().replace('.', '?')}"
            print(f"[VQA] Q: {vqa_question}")
            for i, idx in enumerate(top_k_idx):
                patch = patches[idx]
                nll_yes = self._get_nll(patch, vqa_question, "Yes")
                nll_no = self._get_nll(patch, vqa_question, "No")
                p_yes = 1.0 / (1.0 + np.exp(nll_yes - nll_no))
                status = "✓" if p_yes > 0.9 else "✗"
                s_score = scores[idx] if score_type == "softmax" else -nlls[idx]
                print(f"  [{i}] {patch_names[idx]}: score={s_score:.1%}, p_yes={p_yes:.3f} {status}")
                if p_yes > 0.9:
                    vqa_passed.add(idx)
        
        for idx in range(len(axes)):
            ax = axes[idx]
            if idx < n_patches:
                patch = patches[idx]
                ax.imshow(patch)
                
                if scores is not None and idx < len(scores):
                    if score_type == "softmax":
                        title = f"{patch_names[idx]}\n{score_label}={scores[idx]:.1%}"
                    else:
                        title = f"{patch_names[idx]}\n{score_label}={scores[idx]:.1f}"
                else:
                    title = patch_names[idx]
                ax.set_title(title, fontsize=6)
                
                if idx in top_k_idx:
                    color = 'limegreen' if idx in vqa_passed else 'red'
                    rect = Rectangle((0, 0), patch.width-1, patch.height-1, 
                                      linewidth=8, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
            ax.axis('off')
        
        score_info = "softmax prob" if score_type == "softmax" else "log-likelihood"
        n_passed = len(vqa_passed) if sentence else 0
        prompt_type = "s1" if is_first_sentence else "s2+"
        title = f"Patches ({n_patches} total, top-k={self.top_k_patches}, VQA passed={n_passed}, {score_info}, {prompt_type})"
        if sentence:
            title += f"\nsentence: {sentence}"
        plt.suptitle(title, fontsize=10)
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def plot_codebook(self, max_edits=20, figsize=(6, 4), query_img=None, query_text=None, 
                      apply_cap_k=True, use_effective_dist=False):
        """Plot force-directed network of keys.
        
        Args:
            use_effective_dist: If True, use d_eff = max(0, d - (r1+r2)) instead of raw distance.
                               Edges connect overlapping keys (d_eff=0).
        
        Color by edit_idx, size by is_patch (original=large, patch=small).
        Black circle: text retrieval, Red circle: image-only fallback.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if self.key_embs is None or len(self.codebook) == 0:
            print("[IKE_CHAIN] No keys to plot")
            return
        
        # Sample edits if too many
        all_edits = sorted(set(e.get("edit_idx", 0) for e in self.codebook))
        if len(all_edits) > max_edits:
            import random
            selected_edits = set(random.sample(all_edits, max_edits))
        else:
            selected_edits = set(all_edits)
        
        # Get indices of selected edits
        indices = [i for i, e in enumerate(self.codebook) if e.get("edit_idx", 0) in selected_edits]
        idx_to_local = {g: l for l, g in enumerate(indices)}
        embs = self.key_embs[indices].float().cpu().numpy()
        n_keys = len(indices)
        
        # Get matched indices using helper
        text_global, img_global = set(), set()
        q_emb = None
        if query_img is not None and query_text is not None:
            text_global, img_global = self._get_matched_indices(query_img, query_text, apply_cap_k=apply_cap_k)
            query_patches = self.patchifier.patchify(query_img, kernels=self.query_kernels)
            q_embs = self._encode_vlm(query_patches, [query_text] * len(query_patches)).cpu()
            if self.distance == "cosine":
                q_embs = F.normalize(q_embs, dim=-1)
            q_emb = q_embs[0].numpy()
        text_local = {idx_to_local[g] for g in text_global if g in idx_to_local}
        img_local = {idx_to_local[g] for g in img_global if g in idx_to_local}
        
        # Build graph from similarity
        if use_effective_dist:
            eff_dists = self._effective_distances(indices)
            sims = 1 / (1 + eff_dists)  # overlapping keys have eff_dist=0 -> sim=1
        else:
            dists = np.linalg.norm(embs[:, None] - embs[None, :], axis=-1)
            sims = 1 / (1 + dists)
        
        G = nx.Graph()
        G.add_nodes_from(range(n_keys))
        thresh = np.percentile(sims[np.triu_indices(n_keys, k=1)], self.plot_codebook_pct_threshold) if n_keys > 1 else 0
        for i in range(n_keys):
            for j in range(i + 1, n_keys):
                if sims[i, j] > thresh:
                    G.add_edge(i, j, weight=sims[i, j])
        
        # Add query node if provided
        q_node = None
        if q_emb is not None:
            q_dists = np.linalg.norm(embs - q_emb, axis=-1)
            q_sims = 1 / (1 + q_dists)
            q_node = n_keys
            G.add_node(q_node)
            for i in range(n_keys):
                if q_sims[i] > thresh:
                    G.add_edge(q_node, i, weight=q_sims[i])
        
        # Layout and plot
        pos = nx.spring_layout(G, weight='weight', seed=42, k=2/np.sqrt(len(G.nodes())))
        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_edges(G, pos, alpha=0.08, width=0.2, ax=ax)
        
        # Color map
        edit_list = sorted(selected_edits)
        edit_to_color = {e: i for i, e in enumerate(edit_list)}
        cmap = plt.cm.get_cmap('tab20', max(len(edit_list), 1))
        
        # Draw nodes: original (large circle), patch (small circle), merged (small square)
        for node_type, size, marker in [("original", 60, 'o'), ("patch", 15, 'o'), ("merged", 20, 's')]:
            if node_type == "merged":
                nodelist = [i for i in range(n_keys) if self.codebook[indices[i]].get("is_merged", False)]
            elif node_type == "original":
                nodelist = [i for i in range(n_keys) if not self.codebook[indices[i]].get("is_patch", False) 
                           and not self.codebook[indices[i]].get("is_merged", False)]
            else:  # patch
                nodelist = [i for i in range(n_keys) if self.codebook[indices[i]].get("is_patch", False)
                           and not self.codebook[indices[i]].get("is_merged", False)]
            if not nodelist:
                continue
            colors = [cmap(edit_to_color[self.codebook[indices[i]].get("edit_idx", 0)]) for i in nodelist]
            edgecolors = ['black' if i in text_local else 'red' if i in img_local else 'none' for i in nodelist]
            linewidths = [1.5 if i in text_local or i in img_local else 0 for i in nodelist]
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colors, node_shape=marker,
                                   node_size=size, alpha=0.8, ax=ax, edgecolors=edgecolors, linewidths=linewidths)
        
        # Draw query as black star
        if q_node is not None:
            ax.scatter(pos[q_node][0], pos[q_node][1], c='black', s=80, marker='*', zorder=10)
        
        # Legend
        ax.scatter([], [], c='gray', s=40, marker='o', label='original')
        ax.scatter([], [], c='gray', s=12, marker='o', label='patch')
        ax.scatter([], [], c='gray', s=15, marker='s', label='merged')
        if q_node is not None:
            ax.scatter([], [], c='black', s=40, marker='*', label='query')
        if text_local:
            ax.scatter([], [], c='gray', s=30, marker='o', edgecolors='black', linewidths=1.5, label='text retrieval')
        if img_local:
            ax.scatter([], [], c='gray', s=30, marker='o', edgecolors='red', linewidths=1.5, label='image fallback')
        ax.legend(loc='lower left', fontsize=6, frameon=False, handletextpad=0.1)
        
        dist_mode = "eff" if use_effective_dist else "raw"
        ax.set_title(f'Codebook ({len(edit_list)} edits, {n_keys} keys, {len(text_local)}+{len(img_local)} retrieved, {dist_mode})', fontsize=8)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

