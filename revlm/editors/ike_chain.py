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
        self.cap_k = int(getattr(cfg, "cap_k", 10))  # max entries to retrieve
        self.prefix = getattr(cfg, "cot_prefix", "Context: ")
        self.distance = getattr(cfg, "distance", "l2")
        
        # Radius estimation config
        self.radius_method = getattr(cfg, "radius_method", "single_aug")  # "fixed", "single_aug", or "augment"
        self.fixed_radius = float(getattr(cfg, "fixed_radius", 100.0))
        self.single_aug_scale = float(getattr(cfg, "single_aug_scale", 1.0))  # scale factor for single_aug
        self.n_radius_samples = int(getattr(cfg, "n_radius_samples", 5))
        self.radius_percentile = float(getattr(cfg, "radius_percentile", 99))
        
        # Query kernels: which patches to use at retrieval. None = all 36, ["3x3"] = full only
        self.query_kernels = getattr(cfg, "query_kernels", ["3x3"])
        
        # Patchifier and Augmenter
        self.patchifier = ImagePatchifier()
        self.augmenter = Augmenter(self.wrapper)
        
        # Prompt for patch selection
        self.patch_select_prompt = getattr(cfg, "patch_select_prompt", "Describe this image.")

        # Hook for VLM activations (vision-only)
        model_cfg = getattr(config, "model", config)
        inner_params_vision = getattr(model_cfg, "inner_params_vision", [])
        if not inner_params_vision:
            raise ValueError("Requires config.model.inner_params_vision")
        
        self._vision_act = None
        
        def _setup_hook(param_name, attr_name):
            name = param_name.rsplit(".", 1)[0] if param_name.endswith((".weight", ".bias")) else param_name
            mod = parent_module(self.model, brackets_to_periods(name))
            layer = getattr(mod, name.rsplit(".", 1)[-1])
            return layer.register_forward_hook(
                lambda m, i, o, an=attr_name: setattr(self, an, i[0].detach() if isinstance(i[0], torch.Tensor) else None)
            )
        
        self._vision_hook = _setup_hook(inner_params_vision[0], "_vision_act")

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

    @torch.no_grad()
    def _encode_vlm(self, images: List, texts: List[str]) -> torch.Tensor:
        """Get VLM vision embedding for <image, text> pairs.
        
        Returns: [B, hidden] tensor
        """
        self.model.eval()
        batch_size = len(images) if isinstance(images, list) else 1
        
        self._vision_act = None
        inputs = self.wrapper.encode(images, texts, tokenize=False)
        self.model(**inputs)
        return self._pool_act(self._vision_act, batch_size)

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
        
        nlls = np.array(nlls)
        top_k_idx = np.argsort(nlls)[:self.top_k_patches]
        return [patches[i] for i in top_k_idx]

    @torch.no_grad()
    def _estimate_radius(self, key_emb: torch.Tensor, img, text: str) -> float:
        """Estimate radius.
        
        Methods:
        - 'fixed': constant radius (fastest, no forward pass)
        - 'single_aug': one aggressive augmentation × scale factor (1 forward pass)
        - 'augment': 99th percentile of n augmented samples (n forward passes)
        """
        if self.radius_method == "fixed":
            return self.fixed_radius
        
        if self.radius_method == "single_aug":
            # One augmentation (image + text), scaled up. Always use mosaic for aggressive aug.
            aug_img = self.augmenter.image(img, use_mosaic=True)
            aug_text = self.augmenter.question(text) if text else ""
            aug_emb = self._encode_vlm([aug_img], [aug_text])
            dist = float(torch.norm(aug_emb - key_emb))
            return dist * self.single_aug_scale
        
        # Default: multiple augmentations, take percentile
        aug_dists = []
        for _ in range(self.n_radius_samples):
            aug_img = self.augmenter.image(img)
            aug_text = self.augmenter.question(text) if text else ""
            aug_emb = self._encode_vlm([aug_img], [aug_text])
            dist = float(torch.norm(aug_emb - key_emb))
            aug_dists.append(dist)
        
        return float(np.percentile(aug_dists, self.radius_percentile))

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
        
        for src_idx, src_img in enumerate(image_sources):
            is_patch = (src_idx > 0)
            
            # Key 1: <image, question> -> answer
            new_entries.append({
                "value": answer_value,
                "is_patch": is_patch,
                "edit_idx": self._edit_count,
                "key_text": question
            })
            new_imgs.append(src_img)
            new_texts.append(question)
            
            # Keys 2-4: <image, si> -> si for each sentence
            for sent in rationale_sents:
                new_entries.append({
                    "value": sent,
                    "is_patch": is_patch,
                    "edit_idx": self._edit_count,
                    "key_text": sent
                })
                new_imgs.append(src_img)
                new_texts.append(sent)
        
        self._edit_count += 1
        
        # Compute embeddings
        new_embs = self._encode_vlm(new_imgs, new_texts)
        if self.distance == "cosine":
            new_embs = F.normalize(new_embs, dim=-1)
        
        # Compute radii
        new_radii = []
        for i, (im, tx) in enumerate(zip(new_imgs, new_texts)):
            r = self._estimate_radius(new_embs[i:i+1], im, tx)
            new_radii.append(r)
        new_radii = torch.tensor(new_radii, dtype=torch.float32)
        
        # Append to codebook
        self.codebook.extend(new_entries)
        
        if self.key_embs is None:
            self.key_embs = new_embs
            self.key_radii = new_radii
        else:
            self.key_embs = torch.cat([self.key_embs, new_embs], dim=0)
            self.key_radii = torch.cat([self.key_radii, new_radii])

    @torch.no_grad()
    def _retrieve(self, image, question: str) -> List[str]:
        """Retrieve values for a query <image, question>.
        
        Patchifies query image using query_kernels (None=all 36, ["3x3"]=full only).
        Returns deduplicated set of retrieved values.
        """
        if self.key_embs is None or len(self.codebook) == 0:
            return []
        
        # Patchify query image with specified kernels
        query_patches = self.patchifier.patchify(image, kernels=self.query_kernels)
        
        # Build query embeddings
        q_embs = self._encode_vlm(query_patches, [question] * len(query_patches))
        if self.distance == "cosine":
            q_embs = F.normalize(q_embs, dim=-1)
        
        # Check each query against all keys
        radii = self.key_radii.cpu().numpy()
        retrieved = set()
        matched_indices = []
        
        for q_idx in range(len(query_patches)):
            q_emb = q_embs[q_idx:q_idx+1]
            
            if self.distance == "cosine":
                dists = 1 - (q_emb @ self.key_embs.t()).squeeze(0).cpu().numpy()
            else:
                dists = torch.norm(self.key_embs.float() - q_emb.float(), dim=-1).cpu().numpy()
            
            # Find keys within radius
            in_radius = dists <= radii
            if in_radius.any():
                # Get indices sorted by distance
                valid_idx = np.where(in_radius)[0]
                valid_idx = valid_idx[np.argsort(dists[valid_idx])]
                matched_indices.extend(valid_idx.tolist())
        
        # Deduplicate and collect values (preserve order by first match)
        seen_idx = set()
        for idx in matched_indices:
            if idx not in seen_idx and len(retrieved) < self.cap_k:
                seen_idx.add(idx)
                value = self.codebook[idx]["value"]
                if value:
                    retrieved.add(value)
        
        return list(retrieved)

    def apply_to_dataset(self, dataset):
        """Apply retrieved facts to dataset prompts."""
        applied = 0
        log = []
        data = getattr(dataset, "data", [])
        
        for ex in data:
            prompt, q, img = ex.get("prompt", ""), ex.get("question", ""), ex.get("image")
            if not prompt or img is None:
                continue
            
            facts = self._retrieve(img, q) if q else []
            
            if facts:
                ex.setdefault("prompt_orig", prompt)
                ex["prompt"] = f"{self.prefix}{' '.join(facts)} {prompt}"
                applied += 1
            
            log.append({"uid": ex.get("uid"), "n_facts": len(facts), "facts": facts})
        
        self.last_retrieval_log = log
        print(f"\n[IKE_PATCH] applied facts to {applied}/{len(data)} examples", flush=True)

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
        elif self.radius_method == "single_aug":
            r_info = f"single_aug(×{self.single_aug_scale})"
        else:
            r_info = f"augment(n={self.n_radius_samples})"
        print(f"\n[IKE_PATCH] +{added} edits (k={self.top_k_patches}, r={r_info}), {n_before}->{n_after} keys, {mem_mb:.1f} MB", flush=True)
        
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
        
        stats = {
            "num_keys": len(self.codebook),
            "num_orig_keys": n_orig_keys,
            "num_patch_keys": n_patch_keys,
            "num_edits": len(self._added_uids),
            "top_k_patches": self.top_k_patches,
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
        Optional: add query point as black star.
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
        embs = self.key_embs[indices].float().cpu().numpy()
        n_keys = len(indices)
        
        # Pairwise distances -> similarity
        dists = np.linalg.norm(embs[:, None] - embs[None, :], axis=-1)
        sims = 1 / (1 + dists)
        
        # Build graph
        G = nx.Graph()
        for i in range(n_keys):
            G.add_node(i)
        
        # Add edges (top 10% similarities)
        thresh = np.percentile(sims[np.triu_indices(n_keys, k=1)], 75) if n_keys > 1 else 0
        for i in range(n_keys):
            for j in range(i + 1, n_keys):
                if sims[i, j] > thresh:
                    G.add_edge(i, j, weight=sims[i, j])
        
        # Add query node if provided
        q_node = None
        if query_img is not None and query_text is not None:
            q_emb = self._encode_vlm([query_img], [query_text]).cpu().numpy()
            q_dists = np.linalg.norm(embs - q_emb, axis=-1)
            q_sims = 1 / (1 + q_dists)
            
            q_node = n_keys
            G.add_node(q_node)
            for i in range(n_keys):
                if q_sims[i] > thresh:
                    G.add_edge(q_node, i, weight=q_sims[i])
        
        # Layout
        pos = nx.spring_layout(G, weight='weight', seed=42, k=2/np.sqrt(len(G.nodes())))
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_edges(G, pos, alpha=0.08, width=0.2, ax=ax)
        
        # Color map by edit
        edit_list = sorted(selected_edits)
        edit_to_color = {e: i for i, e in enumerate(edit_list)}
        cmap = plt.cm.get_cmap('tab20', max(len(edit_list), 1))
        
        # Draw original (large) and patch (small) nodes separately
        for is_patch, size in [(False, 60), (True, 15)]:
            nodelist = [i for i in range(n_keys) 
                       if self.codebook[indices[i]].get("is_patch", False) == is_patch]
            if not nodelist:
                continue
            colors = [cmap(edit_to_color[self.codebook[indices[i]].get("edit_idx", 0)]) for i in nodelist]
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colors,
                                   node_size=size, alpha=0.8, ax=ax)
        
        # Draw query as black star
        if q_node is not None:
            ax.scatter(pos[q_node][0], pos[q_node][1], c='black', s=80, marker='*', zorder=10)
        
        # Legend
        ax.scatter([], [], c='gray', s=40, marker='o', label='original')
        ax.scatter([], [], c='gray', s=12, marker='o', label='patch')
        if q_node is not None:
            ax.scatter([], [], c='black', s=40, marker='*', label='query')
        ax.legend(loc='lower left', fontsize=6, frameon=False, handletextpad=0.1)
        
        ax.set_title(f'Codebook ({len(edit_list)} edits, {n_keys} keys)', fontsize=8)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

