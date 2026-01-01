"""IKE_PATCH: Patch-Aware In-Context Knowledge Editing

Expands retrieval surface by creating patch-level keys from images.
Uses log-likelihood scoring to select informative patches.

Key structure: [<image/patch, text>, value]
- Original image always included (4 keys)
- Top-k patches selected by log-likelihood of s1 (4k keys)
- Total per edit: 4 + 4k keys
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
    """Patch-aware codebook for VLM editing.
    
    Codebook entry: [key_emb, value, radius]
    - key_emb: vision_layer(<image/patch, text>) embedding
    - value: sentence to retrieve
    - radius: 99th percentile of augmented distances
    
    Edit structure:
    - 4 keys from original image: (question, s1, s2, s3) × original
    - 4k keys from top-k patches: (question, s1, s2, s3) × each patch
    
    Query: patchify query image, check all 14 query embeddings against codebook.
    """

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        cfg = getattr(config, "editor", config)

        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))

        # Hyperparams
        self.top_k_patches = int(getattr(cfg, "top_k_patches", 3))  # patches to select per edit
        self.cap_k = int(getattr(cfg, "cap_k", 3))  # k closest keys to retrieve among all matching keys
        self.prefix = getattr(cfg, "cot_prefix", "")
        self.distance = getattr(cfg, "distance", "l2")
        self.dual_layer = getattr(cfg, "dual_layer", True)  # concat lang_scaler*lang_layer(<blank, text>) with vision_layer(<img, text>)
        self.lang_encoder = getattr(cfg, "lang_encoder", "sbert")  # "internal" or "sbert"
        # Use lang_scaler_sbert if sbert, else lang_scaler
        if self.lang_encoder == "sbert":
            self.lang_scaler = float(getattr(config.model, "lang_scaler_sbert", 30.0))
        else:
            self.lang_scaler = float(getattr(config.model, "lang_scaler", 30.0))
        
        # Radius estimation config
        self.radius_method = getattr(cfg, "radius_method", "augment")  # "fixed", "augment", or "balance"
        self.fixed_radius = float(getattr(cfg, "fixed_radius", 100.0))
        # "augment": radius based on percentile of augmented image distances
        self.n_radius_samples = int(getattr(cfg, "n_radius_samples", 1))
        self.radius_percentile = float(getattr(cfg, "radius_percentile", 50)) # 99
        # "balance": radius based on positive (augmented image+text) and negative (blank image) samples
        self.n_positive_samples = int(getattr(cfg, "n_positive_samples", 5))
        self.balance_alpha = float(getattr(cfg, "balance_alpha", 0.5))
        
        # Query kernels: which patches to use at retrieval. None = all 36, ["3x3"] = full image only
        self.query_kernels = getattr(cfg, "query_kernels", None)
        
        # Image-only fallback retrieval
        self.image_only_retrieval = getattr(cfg, "image_only_retrieval", False)
        self.top_i_image_only = int(getattr(cfg, "top_i_image_only_entry", 1))
        
        # Seed for reproducibility
        self.seed = getattr(cfg, "seed", None)
        
        # Key merging: merge keys with same key_text if candidate falls within radius
        self.merge_keys = getattr(cfg, "merge_keys", False)
        self.merge_radius_factor = float(getattr(cfg, "merge_radius_factor", 0.25))  # merge if dist < factor * radius
        
        # Patchifier and Augmenter
        self.patchifier = ImagePatchifier()
        dataset_name = getattr(getattr(config, "experiment", None), "dataset_name", None)
        self.augmenter = Augmenter(self.wrapper, seed=self.seed, mosaic_prob=1.0, dataset_name=dataset_name)
        
        # Prompt for patch selection
        self.patch_select_prompt = getattr(cfg, "patch_select_prompt", "Describe this image.")

        # Hook for VLM activations
        model_cfg = getattr(config, "model", config)
        inner_params_vision = getattr(model_cfg, "inner_params_vision", [])
        inner_params_lang = getattr(model_cfg, "inner_params_lang", [])
        if not inner_params_vision:
            raise ValueError("Requires config.model.inner_params_vision")
        # inner_params_lang only required for dual_layer with internal encoder
        if self.dual_layer and self.lang_encoder == "internal" and not inner_params_lang:
            raise ValueError("dual_layer=True with lang_encoder='internal' requires config.model.inner_params_lang")
        
        self._vision_act = None
        self._lang_act = None
        self._blank_image = Image.new('RGB', (224, 224), (128, 128, 128))  # used by dual_layer internal and balance radius
        self._sbert = None  # lazy loaded
        
        def _setup_hook(param_name, attr_name):
            name = param_name.rsplit(".", 1)[0] if param_name.endswith((".weight", ".bias")) else param_name
            mod = parent_module(self.model, brackets_to_periods(name))
            layer = getattr(mod, name.rsplit(".", 1)[-1])
            return layer.register_forward_hook(
                lambda m, i, o, an=attr_name: setattr(self, an, i[0].detach() if isinstance(i[0], torch.Tensor) else None)
            )
        
        self._vision_hook = _setup_hook(inner_params_vision[0], "_vision_act")
        # Lang hook only needed for internal encoder
        self._lang_hook = None
        if self.dual_layer and self.lang_encoder == "internal":
            self._lang_hook = _setup_hook(inner_params_lang[0], "_lang_act")

        # Codebook: list of {key_idx, value, edit_idx, is_patch}
        self.codebook = []
        
        # Embeddings and radii
        self.key_embs = None   # [N, hidden]
        self.key_radii = None  # [N]
        
        # Tracking
        self._added_uids = set()
        self._edit_count = 0
        self.last_retrieval_log = None

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        return (self.model if hasattr(self.model, "generate") else self.wrapper).generate(*a, **kw)

    def _pool_act(self, act, batch_size):
        """Pool activation to [B, hidden] shape."""
        if act is None:
            raise RuntimeError("Hook failed to capture activation")
        act = act.to(self.device, torch.float32)
        if act.dim() == 3:
            return act.mean(dim=1)
        elif act.dim() == 2:
            if act.shape[0] == batch_size:
                return act
            elif act.shape[0] % batch_size == 0:
                patches = act.shape[0] // batch_size
                return act.view(batch_size, patches, -1).mean(dim=1)
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
    def _select_top_k_patches(self, image, s1: str) -> List[Image.Image]:
        """Select top-k patches by log-likelihood of s1."""
        patches = self.patchifier.patchify_exclude_full(image)  # 35 patches
        
        nlls = []
        for patch in patches:
            nll = self._get_nll(patch, self.patch_select_prompt, s1)
            nlls.append(nll)
        
        # Clear cache after many forward passes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        nlls = np.array(nlls)
        top_k_idx = np.argsort(nlls)[:self.top_k_patches]
        return [patches[i] for i in top_k_idx]

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
            return self.fixed_radius
        
        if self.radius_method == "balance":
            return self._estimate_radius_balance(key_emb, img, text, is_question)
        
        # Default: multiple augmentations, take percentile
        aug_dists = []
        for _ in range(self.n_radius_samples):
            aug_img = self.augmenter.image(img)
            aug_text = self.augmenter.question(text) if is_question else self.augmenter.rationale(text) if text else ""
            aug_emb = self._encode_vlm([aug_img], [aug_text])
            dist = float(torch.norm(aug_emb.cpu() - key_emb.cpu()))
            aug_dists.append(dist)
        
        return float(np.percentile(aug_dists, self.radius_percentile))

    def _try_merge_key(self, emb: torch.Tensor, radius: float, key_text: str) -> bool:
        """Try to merge a new key into an existing one with same key_text.
        
        Merge condition: same key_text AND dist < existing_r * merge_radius_factor.
        Merge result: existing key stays, radius extends if candidate sticks out.
        
        Returns True if merged, False if should add as new key.
        """
        if self.key_embs is None or not key_text:
            return False
        
        # Find candidates with same key_text
        candidates = [i for i, e in enumerate(self.codebook) if e.get("key_text") == key_text]
        if not candidates:
            return False
        
        # Check if new key falls within any existing key's radius
        emb_cpu = emb.cpu().squeeze(0) if emb.dim() > 1 else emb.cpu()
        for idx in candidates:
            existing_emb = self.key_embs[idx]
            existing_r = float(self.key_radii[idx])
            dist = float(torch.norm(emb_cpu - existing_emb))
            
            if dist < existing_r * self.merge_radius_factor:  # Candidate must be deep inside
                # Extend radius if candidate's coverage sticks out
                new_r = max(existing_r, dist + radius)
                
                # Update radius only (embedding stays)
                self.key_radii[idx] = new_r
                self.codebook[idx]["is_merged"] = True
                return True
        
        return False

    @torch.no_grad()
    def _add_edit(self, img, question: str, answer: str, rationale_sents: List[str]):
        """Add keys for one edit.
        
        Creates:
        - 4 keys from original image
        - 4k keys from top-k patches
        """
        # Select top-k patches based on s1 likelihood
        s1 = rationale_sents[0] if rationale_sents else ""
        top_patches = self._select_top_k_patches(img, s1) if s1 else []
        
        # Prepare all image sources: original + top-k patches
        image_sources = [img] + top_patches
        
        # Build 4 key-value pairs per image source
        answer_value = f"The answer to '{question}' is {answer}." if answer else ""
        
        new_entries = []
        new_imgs = []
        new_texts = []
        new_is_question = []  # Track if each key is question (True) or rationale (False)
        
        for src_idx, src_img in enumerate(image_sources):
            is_patch = (src_idx > 0)
            
            # Key 1: <image, question> -> answer
            new_entries.append({
                "value": answer_value,
                "is_patch": is_patch,
                "edit_idx": self._edit_count,
                "key_text": question,
                "is_image_only": False,
                "is_question": True
            })
            new_imgs.append(src_img)
            new_texts.append(question)
            new_is_question.append(True)
            
            # Keys 2-4: <image, si> -> si for each sentence (rationale)
            for sent in rationale_sents:
                new_entries.append({
                    "value": sent,
                    "is_patch": is_patch,
                    "edit_idx": self._edit_count,
                    "key_text": sent,
                    "is_image_only": False,
                    "is_question": False
                })
                new_imgs.append(src_img)
                new_texts.append(sent)
                new_is_question.append(False)
            
            # Key 5: <image, ""> -> "" (image-only gate key)
            if self.image_only_retrieval:
                new_entries.append({
                    "value": "",
                    "is_patch": is_patch,
                    "edit_idx": self._edit_count,
                    "key_text": "",
                    "is_image_only": True,
                    "is_question": False
                })
                new_imgs.append(src_img)
                new_texts.append("")
                new_is_question.append(False)
        
        self._edit_count += 1
        
        # Compute embeddings
        new_embs = self._encode_vlm(new_imgs, new_texts)
        if self.distance == "cosine":
            new_embs = F.normalize(new_embs, dim=-1)
        
        # Compute radii per key, using appropriate augmentation
        new_radii = []
        for i, (src_img, text, is_q) in enumerate(zip(new_imgs, new_texts, new_is_question)):
            if not text:  # image-only key
                # Use question radius for image-only keys
                r = self._estimate_radius(new_embs[i:i+1], src_img, question, is_question=True)
            else:
                r = self._estimate_radius(new_embs[i:i+1], src_img, text, is_question=is_q)
            new_radii.append(r)
        new_radii = torch.tensor(new_radii, dtype=torch.float32)
        
        # Move to CPU to save GPU memory (only used for retrieval)
        new_embs = new_embs.cpu()
        
        # Add keys to codebook (with optional merging)
        for i, entry in enumerate(new_entries):
            emb, radius = new_embs[i:i+1], float(new_radii[i])
            key_text = entry.get("key_text", "")
            
            # Try merge if enabled, otherwise add as new
            if self.merge_keys and self._try_merge_key(emb, radius, key_text):
                continue  # Merged into existing key
            
            # Add as new key
            self.codebook.append(entry)
            if self.key_embs is None:
                self.key_embs = emb
                self.key_radii = torch.tensor([radius])
            else:
                self.key_embs = torch.cat([self.key_embs, emb], dim=0)
                self.key_radii = torch.cat([self.key_radii, torch.tensor([radius])])
        
        # Clear CUDA cache
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
        
        # Check matches
        in_radius = dist_matrix <= key_radii
        matched_mask = in_radius.any(dim=0)
        
        if not matched_mask.any():
            return []
        
        # Get matched, sorted by min distance
        matched_local = torch.where(matched_mask)[0]
        min_dists = dist_matrix[:, matched_local].min(dim=0).values
        sorted_order = min_dists.argsort()
        matched_local = matched_local[sorted_order].tolist()
        
        # Collect values
        retrieved = set()
        for local_i in matched_local[:self.cap_k]:
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
    def _get_matched_indices(self, image, question: str) -> Tuple[set, set]:
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
                top_k = matched_local[min_dists.argsort()][:self.cap_k]
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
                    top_k = matched_local[min_dists.argsort()][:self.cap_k]
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
        torch.save({
            "codebook": self.codebook,
            "key_embs": self.key_embs,
            "key_radii": self.key_radii,
            "top_k_patches": self.top_k_patches,
        }, path)
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
            "image_only_retrieval": self.image_only_retrieval,
            "emb_size_mb": self.key_embs.numel() * 2 / 1024 / 1024 if self.key_embs is not None else 0,
        }
        if self.key_radii is not None:
            stats["avg_radius"] = float(self.key_radii.mean())
            stats["min_radius"] = float(self.key_radii.min())
            stats["max_radius"] = float(self.key_radii.max())
        return stats

    @torch.no_grad()
    def visualize_patches(self, image, s1: str = None, figsize=(16, 10), score_type="softmax"):
        """Visualize patchification with scores and top-k highlighted.
        
        Args:
            image: Input image
            s1: First sentence for patch selection (if None, shows all patches without scores)
            figsize: Figure size
            score_type: "softmax" (default, probabilities sum to 1) or "ll" (raw log-likelihood)
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from scipy.special import softmax
        
        patches = self.patchifier.patchify(image)
        patch_names = self.patchifier.get_patch_names()
        n_patches = len(patches)
        
        # Grid layout: 6x6 for 36 patches
        n_cols = 6
        n_rows = (n_patches + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        # Compute scores for each patch (excluding full image)
        scores = None
        top_k_idx = []
        score_label = ""
        if s1:
            nlls = []
            for patch in patches[:-1]:  # exclude 3x3
                nll = self._get_nll(patch, self.patch_select_prompt, s1)
                nlls.append(nll)
            nlls = np.array(nlls)
            
            if score_type == "softmax":
                scores = softmax(-nlls)  # softmax over -NLL (higher prob = better)
                score_label = "P"
                top_k_idx = np.argsort(scores)[::-1][:self.top_k_patches].tolist()
            else:  # "ll"
                scores = -nlls
                score_label = "LL"
                top_k_idx = np.argsort(scores)[::-1][:self.top_k_patches].tolist()
        
        for idx in range(len(axes)):
            ax = axes[idx]
            if idx < n_patches:
                patch = patches[idx]
                ax.imshow(patch)
                
                # Build title with score if available
                if scores is not None and idx < len(scores):
                    if score_type == "softmax":
                        title = f"{patch_names[idx]}\n{score_label}={scores[idx]:.1%}"
                    else:
                        title = f"{patch_names[idx]}\n{score_label}={scores[idx]:.1f}"
                else:
                    title = patch_names[idx]
                ax.set_title(title, fontsize=6)
                
                # Highlight top-k with green box
                if idx in top_k_idx:
                    rect = Rectangle((0, 0), patch.width-1, patch.height-1, 
                                      linewidth=4, edgecolor='limegreen', facecolor='none')
                    ax.add_patch(rect)
            ax.axis('off')
        
        # Title with s1 preview
        score_info = "softmax prob" if score_type == "softmax" else "log-likelihood"
        title = f"Patches ({n_patches} total, top-k={self.top_k_patches}, {score_info})"
        if s1:
            title += f"\ns1: {s1}"
        plt.suptitle(title, fontsize=10)
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def plot_codebook(self, max_edits=20, figsize=(6, 4), query_img=None, query_text=None):
        """Plot force-directed network of keys.
        
        Color by edit_idx, size by is_patch (original=large, patch=small).
        Black circle: text retrieval, Red circle: image-only fallback.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if self.key_embs is None or len(self.codebook) == 0:
            print("[IKE_PATCH] No keys to plot")
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
            text_global, img_global = self._get_matched_indices(query_img, query_text)
            # Get query embedding for plotting
            query_patches = self.patchifier.patchify(query_img, kernels=self.query_kernels)
            q_embs = self._encode_vlm(query_patches, [query_text] * len(query_patches)).cpu()
            if self.distance == "cosine":
                q_embs = F.normalize(q_embs, dim=-1)
            q_emb = q_embs[0].numpy()
        text_local = {idx_to_local[g] for g in text_global if g in idx_to_local}
        img_local = {idx_to_local[g] for g in img_global if g in idx_to_local}
        
        # Build graph from similarity
        dists = np.linalg.norm(embs[:, None] - embs[None, :], axis=-1)
        sims = 1 / (1 + dists)
        G = nx.Graph()
        G.add_nodes_from(range(n_keys))
        thresh = np.percentile(sims[np.triu_indices(n_keys, k=1)], 90) if n_keys > 1 else 0
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
        
        ax.set_title(f'Codebook ({len(edit_list)} edits, {n_keys} keys, {len(text_local)}+{len(img_local)} retrieved)', fontsize=8)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

