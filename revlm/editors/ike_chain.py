import re
import numpy as np
from scipy.stats import t as t_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import brackets_to_periods, parent_module, Augmenter


class IKE_CHAIN(nn.Module):
    """Non-parametric chain-of-keys codebook for VLM editing.
    
    Entry structure: [key_emb, downstream_keys, retrieve_value]
    - key_emb: <img, text> VLM embedding (computed once at edit time)
    - downstream_keys: indices of valid next keys in chain
    - retrieve_value: sentence to return when matched
    
    Memory efficient: images not stored, incremental indexing.
    """

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        cfg = getattr(config, "editor", config)

        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))

        # Hyperparams
        self.top_n = int(getattr(cfg, "top_n", 50))
        self.cap_k = int(getattr(cfg, "cap_k", 1))  # Cap entry points: 1=top-1, >1=multiple entries
        self.prefix = getattr(cfg, "cot_prefix", "New Fact: ")
        self.distance = getattr(cfg, "distance", "l2")  # "l2" (default) or "cosine"
        self.neighbor_window = int(getattr(cfg, "neighbor_window", 0))  # 0=exact, 1=[prev,curr,next], etc.
        self.auto_k_method = getattr(cfg, "auto_k_method", "ensemble")  # "grubbs", "otsu", or "ensemble"
        
        # Augmentation config (0 = disabled)
        self.n_aug_entry = int(getattr(cfg, "n_aug_entry", 0))  # Augmented entry points per edit
        self.n_aug_sent = int(getattr(cfg, "n_aug_sent", 0))    # Augmented sentence keys per edit
        self.augmenter = Augmenter(self.wrapper) if (self.n_aug_entry > 0 or self.n_aug_sent > 0) else None

        # Hook for VLM activations
        inner_params = getattr(getattr(config, "model", config), "inner_params", [])
        if not inner_params:
            raise ValueError("Requires config.model.inner_params")
        raw = inner_params[0]
        self.inner_param_name = raw.rsplit(".", 1)[0] if raw.endswith((".weight", ".bias")) else raw
        edit_mod = parent_module(self.model, brackets_to_periods(self.inner_param_name))
        self.target_layer = getattr(edit_mod, self.inner_param_name.rsplit(".", 1)[-1])
        self._last_act = None
        self._hook = self.target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_last_act", i[0].detach() if isinstance(i[0], torch.Tensor) else None)
        )

        # Codebook: stores only downstream indices and retrieve values (no images!)
        # Each entry: {"downstream_idx": list[int], "retrieve": str}
        self.codebook = []
        
        # Embeddings stored separately, built incrementally
        self.key_embs = None  # [N, hidden]
        
        # Track added edits to avoid duplicates in sequential mode
        self._added_uids = set()
        
        # Logging (API compat with other editors)
        self.last_retrieval_log = None

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        return (self.model if hasattr(self.model, "generate") else self.wrapper).generate(*a, **kw)

    @torch.no_grad()
    def _encode_vlm(self, images, texts):
        """Get VLM layer activation for <image, text> pairs."""
        self.model.eval()
        self._last_act = None
        inputs = self.wrapper.encode(images, texts, tokenize=False)
        self.model(**inputs)
        act = self._last_act
        if act is None:
            raise RuntimeError("Hook failed")
        act = act.to(self.device, torch.float32)
        
        if act.dim() != 3:
            raise RuntimeError(f"Expected 3D activation (B, seq, hidden), got {act.shape}")
        
        # Mean pool over sequence
        return act.mean(dim=1)  # [B, hidden]

    @torch.no_grad()
    def _add_edit(self, img, question, cot_sents, answer):
        """Add chain entries for one edit. Computes embeddings immediately.
        
        Builds entries and embeddings incrementally - no rebuild needed.
        Images are NOT stored, only embeddings.
        
        With augmentation enabled:
        - n_aug_entry: adds augmented entry points (img+question variants)
        - n_aug_sent: adds augmented sentence keys (img variants)
        """
        base_idx = len(self.codebook)
        n = len(cot_sents)
        sent_indices = list(range(base_idx, base_idx + n))
        
        # Collect all <img, text> pairs for this edit
        imgs = []
        texts = []
        
        # Sentence entries: s1 -> s2 -> ... -> sn -> end
        # With neighbor_window, each key retrieves [s_{i-w}, ..., s_i, ..., s_{i+w}]
        for i, s in enumerate(cot_sents):
            # Compute neighbor window
            w = self.neighbor_window
            start = max(0, i - w)
            end = min(n, i + w + 1)
            neighbors = cot_sents[start:end]
            
            self.codebook.append({
                "downstream_idx": sent_indices[i+1:],
                "retrieve": neighbors  # List of sentences (window around s_i)
            })
            imgs.append(img)
            texts.append(s)
        
        # Entry point 1: <img, ""> -> all sentences
        self.codebook.append({
            "downstream_idx": sent_indices,
            "retrieve": []  # Entry point retrieves nothing directly # list(cot_sents) 
        })
        imgs.append(img)
        texts.append("")
        
        # Entry point 2: <img, question> -> all sentences
        answer_text = f"The answer to '{question}' is {answer}." if answer else ""
        self.codebook.append({
            "downstream_idx": sent_indices,
            "retrieve": [answer_text] if answer_text else []  # List format # [answer_text] + list(cot_sents) if answer_text else list(cot_sents)
        })
        imgs.append(img)
        texts.append(question)
        
        # Augmented entry points (for text/image generality)
        if self.augmenter and self.n_aug_entry > 0:
            for _ in range(self.n_aug_entry):
                aug_img = self.augmenter.image(img)
                # Augmented <img, ""> entry
                self.codebook.append({
                    "downstream_idx": sent_indices,
                    "retrieve": []
                })
                imgs.append(aug_img)
                texts.append("")
                # Augmented <img, question> entry (with text augmentation)
                aug_q = self.augmenter.question(question) if question else ""
                self.codebook.append({
                    "downstream_idx": sent_indices,
                    "retrieve": [answer_text] if answer_text else []
                })
                imgs.append(aug_img)
                texts.append(aug_q)
        
        # Augmented sentence keys (for image generality on chain following)
        if self.augmenter and self.n_aug_sent > 0:
            for _ in range(self.n_aug_sent):
                aug_img = self.augmenter.image(img)
                for i, s in enumerate(cot_sents):
                    # Same neighbor window as non-augmented
                    w = self.neighbor_window
                    start = max(0, i - w)
                    end = min(n, i + w + 1)
                    neighbors = cot_sents[start:end]
                    
                    self.codebook.append({
                        "downstream_idx": sent_indices[i+1:],
                        "retrieve": neighbors
                    })
                    imgs.append(aug_img)
                    texts.append(s)
        
        # Compute embeddings for this edit ONLY (incremental)
        new_embs = self._encode_vlm(imgs, texts)
        if self.distance == "cosine":
            new_embs = F.normalize(new_embs, dim=-1)
        
        # Append to existing index
        if self.key_embs is None:
            self.key_embs = new_embs
        else:
            self.key_embs = torch.cat([self.key_embs, new_embs], dim=0)

    @staticmethod
    def _grubbs_k(sims, top_n=50, alpha=0.05):
        """Use Grubbs' test on similarity gaps to find natural cutoff.
        
        Returns 0 if no significant outlier found (same as IKE_PROTO).
        """
        sims = np.asarray(sims, dtype=float)
        if sims.size < 4:
            return 0  # Not enough data for reliable test
        vals = np.sort(sims)[::-1][:top_n]
        spread = vals[0] - vals[-1]
        if spread <= 0:
            return 0  # No spread = no clear outlier
        d = (vals[:-1] - vals[1:]) / spread
        n = d.size
        if n < 3:
            return 0
        mean, std = d.mean(), d.std(ddof=1)
        if std <= 1e-12:
            return 0
        i = int(np.argmax(d))
        G = abs(d[i] - mean) / std
        p = alpha / (2 * n)
        tcrit = t_dist.ppf(1 - p, df=n - 2)
        Gcrit = ((n - 1) / np.sqrt(n)) * np.sqrt(tcrit**2 / (n - 2 + tcrit**2))
        return (i + 1) if G > Gcrit else 0  # Return 0 if no significant gap

    @staticmethod
    def _otsu_k(sims, n_bins=50):
        """Find threshold that minimizes intra-class variance (good vs bad matches)."""
        sims = np.asarray(sims, dtype=float)
        if sims.size < 2:
            return 0
        hist, bin_edges = np.histogram(sims, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total = hist.sum()
        if total == 0:
            return 0
        
        best_thresh, best_var = 0, -1
        for i in range(1, len(hist)):
            w0, w1 = hist[:i].sum(), hist[i:].sum()
            if w0 == 0 or w1 == 0:
                continue
            m0 = (hist[:i] * bin_centers[:i]).sum() / w0
            m1 = (hist[i:] * bin_centers[i:]).sum() / w1
            var = w0 * w1 * (m0 - m1) ** 2
            if var > best_var:
                best_var, best_thresh = var, bin_centers[i]
        
        return int((sims > best_thresh).sum()) if best_var > 0 else 0

    def _auto_k(self, scores):
        """Dispatch to the configured auto-k method.
        
        Methods:
        - "grubbs": Grubbs' test on similarity gaps (conservative, statistical)
        - "otsu": Otsu's method for bimodal split (good for clear separation)
        - "ensemble": Both must agree (most conservative, highest precision)
        """
        if self.auto_k_method == "grubbs":
            return self._grubbs_k(scores, top_n=self.top_n)
        elif self.auto_k_method == "otsu":
            return self._otsu_k(scores)
        elif self.auto_k_method == "ensemble":
            k_grubbs = self._grubbs_k(scores, top_n=self.top_n)
            k_otsu = self._otsu_k(scores)
            # Both must agree there are outliers; take the more conservative (smaller k)
            if k_grubbs == 0 or k_otsu == 0:
                return 0
            return min(k_grubbs, k_otsu)
        else:
            raise ValueError(f"Unknown auto_k_method: {self.auto_k_method}")

    @torch.no_grad()
    def _retrieve_chain(self, image, start_text=""):
        """Retrieve chain starting from <image, start_text>.
        
        1. Encode query
        2. Grubbs test to find k significant outliers
        3. If k=0, return [] (no match)
        4. Else enter min(k, cap_k) chains, follow downstream picking closest to query
        
        Supports L2 distance (default) or cosine similarity.
        """
        if self.key_embs is None or len(self.codebook) == 0:
            return []

        # Encode query
        q_emb = self._encode_vlm([image], [start_text])
        
        if self.distance == "cosine":
            # Cosine similarity: normalize and dot product
            q_emb = F.normalize(q_emb, dim=-1)
            sims = (q_emb @ self.key_embs.t()).squeeze(0).float().cpu().numpy()
            # Higher is better for cosine
            scores = sims
        else:
            # L2 distance: lower is better, convert to similarity for Grubbs
            q_emb = q_emb
            dists = torch.norm(self.key_embs.float() - q_emb.float(), dim=-1).cpu().numpy()
            # Convert to "similarity" (negative distance) so higher = closer
            scores = -dists
        
        # Run auto-k detection (grubbs, otsu, or ensemble)
        k = self._auto_k(scores)
        if k == 0:
            return []  # No significant outlier, don't enter any chain
        
        # Cap k at configured value
        k = min(k, self.cap_k)
        
        # Get top-k entry points (highest scores = best matches)
        entry_indices = np.argsort(scores)[::-1][:k].tolist()
        
        # Collect from all entry points
        collected = []
        seen = set()
        
        for entry_idx in entry_indices:
            # Follow this chain
            idx = entry_idx
            while True:
                entry = self.codebook[idx]
                # retrieve is now a list of sentences
                for sent in entry["retrieve"]:
                    if sent and sent not in seen:
                        seen.add(sent)
                        collected.append(sent)
                
                downstream = entry["downstream_idx"]
                if not downstream:
                    break
                
                # Pick downstream key closest to original query (highest score)
                sub_scores = scores[downstream]
                best_sub = int(np.argmax(sub_scores))
                idx = downstream[best_sub]
        
        return collected

    def apply_to_dataset(self, dataset):
        """Apply retrieved facts to dataset prompts (two routes)."""
        applied = 0
        log = []
        for ex in getattr(dataset, "data", []):
            prompt, q, img = ex.get("prompt", ""), ex.get("question", ""), ex.get("image")
            if not prompt or img is None:
                continue
            
            # Route 1: <img, "">
            facts1 = self._retrieve_chain(img, "")
            # Route 2: <img, question>
            facts2 = self._retrieve_chain(img, q) if q else []
            
            # Merge unique, route 1 first
            seen = set(facts1)
            facts = facts1 + [f for f in facts2 if f not in seen]
            
            if facts:
                ex.setdefault("prompt_orig", prompt)
                ex["prompt"] = f"{self.prefix}{' '.join(facts)} {prompt}"
                applied += 1
            
            log.append({"uid": ex.get("uid"), "n_facts": len(facts), "facts": facts})
        
        self.last_retrieval_log = log
        print(f"[IKE_CHAIN] applied facts to {applied} examples", flush=True)

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add edits to codebook (no training, incremental indexing)."""
        if edit_ds is None:
            return self.model
        
        n_before = len(self.codebook)
        added = 0
        
        for ex in getattr(edit_ds, "data", []):
            # Skip already added edits (for sequential mode)
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            if uid in self._added_uids:
                continue
            
            rat = ex.get("cot") or ex.get("rationale") or ""
            q = ex.get("question", "")
            img = ex.get("image")
            ans = ex.get("answer") or ex.get("target") or ""
            
            if not rat or img is None:
                continue
            
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", rat.strip()) if s.strip()]
            if sents:
                self._add_edit(img, q, sents, ans)
                self._added_uids.add(uid)
                added += 1
        
        n_after = len(self.codebook)
        mem_mb = self.key_embs.numel() * 2 / 1024 / 1024 if self.key_embs is not None else 0
        print(f"[IKE_CHAIN] +{added} edits, {n_before}->{n_after} keys, {mem_mb:.1f} MB", flush=True)
        
        # Apply to dataset
        self.apply_to_dataset(edit_ds)
        
        return self.model
    
    def save_index(self, path):
        """Save codebook and embeddings to disk."""
        torch.save({
            "codebook": self.codebook,
            "key_embs": self.key_embs
        }, path)
        print(f"[IKE_CHAIN] saved {len(self.codebook)} keys to {path}", flush=True)
    
    def load_index(self, path):
        """Load codebook and embeddings from disk."""
        data = torch.load(path, map_location=self.device)
        self.codebook = data["codebook"]
        self.key_embs = data["key_embs"].to(self.device)
        print(f"[IKE_CHAIN] loaded {len(self.codebook)} keys from {path}", flush=True)
