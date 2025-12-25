import re
import numpy as np
from scipy.stats import t as t_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
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
        self.top_n = int(getattr(cfg, "top_n", 30))
        self.cap_k = int(getattr(cfg, "cap_k", 3))  # Cap entry points: 1=top-1, >1=multiple entries
        self.prefix = getattr(cfg, "cot_prefix", "New Fact: ")
        self.distance = getattr(cfg, "distance", "l2")  # "l2" (default) or "cosine"
        self.neighbor_window = int(getattr(cfg, "neighbor_window", 0))  # 0=exact, 1=[prev,curr,next], etc.
        self.auto_k_method = getattr(cfg, "auto_k_method", "radius")  # "grubbs", "otsu", "ensemble", or "radius"
        
        # Radius estimation config (for auto_k_method="radius")
        self.radius_method = getattr(cfg, "radius_method", "balance")  # "balance", "augment", or "balancekey" (need more than 2 edits)
        # "augment": radius based on percentile of augmented image distances
        self.n_radiusaug_samples = int(getattr(cfg, "n_radiusaug_samples", 10))
        self.radius_percentile = float(getattr(cfg, "radius_percentile", 99))
        # "balance": radius based on positive (augmented image+text) and negative (blank image) samples
        self.n_positive_samples = int(getattr(cfg, "n_positive_samples", 5))
        self.balance_alpha = float(getattr(cfg, "balance_alpha", 0.3))
        # "balancekey": radius based on positive (augmented image+text) and other keys in codebook
        
        # Augmentation config (0 = disabled)
        self.n_aug_entry = int(getattr(cfg, "n_aug_entry", 1))  # Augmented entry points per edit # 1 
        self.n_aug_sent = int(getattr(cfg, "n_aug_sent", 1))    # Augmented sentence keys per edit # 1
        needs_aug = self.n_aug_entry > 0 or self.n_aug_sent > 0 or self.auto_k_method == "radius"
        self.augmenter = Augmenter(self.wrapper) if needs_aug else None
        
        # Switch: set to True to plot score distributions in apply_to_dataset
        self.plot_k_dist = False

        # Hook for VLM activations (supports dual-layer: vision + language)
        model_cfg = getattr(config, "model", config)
        inner_params = getattr(model_cfg, "inner_params", [])
        inner_params_vision = getattr(model_cfg, "inner_params_vision", [])
        if not inner_params:
            raise ValueError("Requires config.model.inner_params")
        
        # Dual-layer mode: inner_params (language) + inner_params_vision (vision)
        self._dual_layer = len(inner_params_vision) > 0
        self._vision_act = None
        self._lang_act = None
        
        def _setup_hook(param_name, attr_name):
            name = param_name.rsplit(".", 1)[0] if param_name.endswith((".weight", ".bias")) else param_name
            mod = parent_module(self.model, brackets_to_periods(name))
            layer = getattr(mod, name.rsplit(".", 1)[-1])
            return layer.register_forward_hook(
                lambda m, i, o, an=attr_name: setattr(self, an, i[0].detach() if isinstance(i[0], torch.Tensor) else None)
            )
        
        # Language layer from inner_params, vision layer from inner_params_vision
        self._lang_hook = _setup_hook(inner_params[0], "_lang_act")
        self._vision_hook = _setup_hook(inner_params_vision[0], "_vision_act") if self._dual_layer else None
        
        # Blank image for language-only embedding (gray 224x224)
        self._blank_image = Image.new('RGB', (224, 224), (128, 128, 128)) if self._dual_layer else None

        # Codebook: stores only downstream indices and retrieve values (no images!)
        # Each entry: {"downstream_idx": list[int], "retrieve": str}
        self.codebook = []
        
        # Embeddings and radii stored separately, built incrementally
        self.key_embs = None  # [N, hidden]
        self.key_radii = None  # [N] - radius per key (for radius-based retrieval)
        
        # Track added edits to avoid duplicates in sequential mode
        self._added_uids = set()
        self._edit_count = 0  # Counter for edit index
        
        # Logging (API compat with other editors)
        self.last_retrieval_log = None

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        return (self.model if hasattr(self.model, "generate") else self.wrapper).generate(*a, **kw)

    def _pool_act(self, act, batch_size):
        """Pool activation to [B, hidden] shape."""
        if act is None:
            raise RuntimeError("Hook failed")
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
    def _encode_vlm(self, images, texts):
        """Get VLM embedding for <image, text> pairs.
        
        Dual-layer mode: concat(vision(<image,text>), language(<blank,text>))
        Single-layer mode: language(<image,text>) only
        """
        self.model.eval()
        batch_size = len(images) if isinstance(images, list) else 1
        
        if not self._dual_layer:
            # Single-layer: language only
            self._lang_act = None
            inputs = self.wrapper.encode(images, texts, tokenize=False)
            self.model(**inputs)
            return self._pool_act(self._lang_act, batch_size)
        
        # Dual-layer mode
        # Pass 1: <image, text> -> vision embedding
        self._vision_act = None
        inputs = self.wrapper.encode(images, texts, tokenize=False)
        self.model(**inputs)
        vision_emb = self._pool_act(self._vision_act, batch_size)
        
        # Pass 2: <blank_image, text> -> language embedding
        self._lang_act = None
        blank_imgs = [self._blank_image] * batch_size
        inputs = self.wrapper.encode(blank_imgs, texts, tokenize=False)
        self.model(**inputs)
        lang_emb = self._pool_act(self._lang_act, batch_size)
        
        return torch.cat([vision_emb, lang_emb], dim=-1)

    @torch.no_grad()
    def _estimate_radius_augment(self, key, img, text):
        """Estimate radius via percentile of augmented image distances."""
        if self.n_radiusaug_samples <= 0:
            return 0.0
        aug_keys = []
        for _ in range(self.n_radiusaug_samples):
            aug_img = self.augmenter.image(img)
            aug_key = self._encode_vlm([aug_img], [text])
            aug_keys.append(aug_key)
        aug_keys = torch.cat(aug_keys, dim=0)  # [N, D]
        dists = torch.norm(aug_keys - key, dim=-1).cpu().numpy()
        return float(np.percentile(dists, self.radius_percentile))

    @torch.no_grad()
    def _estimate_radius_balance(self, key, img, text):
        """Estimate radius via positive (augmented image+text) and negative (blank image) samples."""
        # Positive: augmented image + augmented text
        pos_dists = []
        for _ in range(self.n_positive_samples):
            aug_img = self.augmenter.image(img)
            aug_text = self.augmenter.question(text) if text else ""
            pos_key = self._encode_vlm([aug_img], [aug_text])
            pos_dists.append(float(torch.norm(pos_key - key)))
        d_pos = float(np.median(pos_dists)) if pos_dists else 0.0
        # Negative: blank image, same text
        neg_key = self._encode_vlm([self._blank_image or self.augmenter._blank], [text])
        d_neg = float(torch.norm(neg_key - key))
        # Combined: ε = (1 - α) * d(Pos, k) + α * d(Neg, k)
        return (1 - self.balance_alpha) * d_pos + self.balance_alpha * d_neg

    @torch.no_grad()
    def _estimate_radius_balancekey(self, key, img, text):
        """Estimate radius via positive samples (short) and other keys in codebook (long).
        
        Short radius: median distance to augmented (image, text) pairs
        Long radius: median distance to all other existing keys
        Final: midpoint of short and long
        """
        # Short: median dist to positive samples (augmented image + augmented text)
        pos_dists = []
        for _ in range(self.n_positive_samples):
            aug_img = self.augmenter.image(img)
            aug_text = self.augmenter.question(text) if text else ""
            pos_key = self._encode_vlm([aug_img], [aug_text])
            pos_dists.append(float(torch.norm(pos_key - key)))
        d_short = float(np.median(pos_dists)) if pos_dists else 0.0
        
        # Long: median dist to all other keys in codebook
        if self.key_embs is not None and len(self.key_embs) > 0:
            all_dists = torch.norm(self.key_embs.float() - key.float(), dim=-1).cpu().numpy()
            d_long = float(np.median(all_dists))
        else:
            d_long = d_short * 2  # fallback for first key: double the short radius
        
        # Final: alpha-weighted mean (same as balance)
        return (1 - self.balance_alpha) * d_short + self.balance_alpha * d_long

    def _estimate_radius(self, key, img, text):
        """Estimate radius for a key using configured method."""
        if self.radius_method == "balance":
            return self._estimate_radius_balance(key, img, text)
        elif self.radius_method == "balancekey":
            return self._estimate_radius_balancekey(key, img, text)
        return self._estimate_radius_augment(key, img, text)

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
                "retrieve": neighbors,
                "is_aug": False,
                "edit_idx": self._edit_count
            })
            imgs.append(img)
            texts.append(s)
        
        # Entry point 1: <img, ""> -> all sentences
        self.codebook.append({
            "downstream_idx": sent_indices,
            "retrieve": [],
            "is_aug": False,
            "edit_idx": self._edit_count
        })
        imgs.append(img)
        texts.append("")
        
        # Entry point 2: <img, question> -> all sentences
        answer_text = f"The answer to '{question}' is {answer}." if answer else ""
        self.codebook.append({
            "downstream_idx": sent_indices,
            "retrieve": [answer_text] if answer_text else [],
            "is_aug": False,
            "edit_idx": self._edit_count
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
                    "retrieve": [],
                    "is_aug": True,
                    "edit_idx": self._edit_count
                })
                imgs.append(aug_img)
                texts.append("")
                # Augmented <img, question> entry (with text augmentation)
                aug_q = self.augmenter.question(question) if question else ""
                self.codebook.append({
                    "downstream_idx": sent_indices,
                    "retrieve": [answer_text] if answer_text else [],
                    "is_aug": True,
                    "edit_idx": self._edit_count
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
                        "retrieve": neighbors,
                        "is_aug": True,
                        "edit_idx": self._edit_count
                    })
                    imgs.append(aug_img)
                    texts.append(s)
        
        # Increment edit counter
        self._edit_count += 1
        
        # Compute embeddings for this edit ONLY (incremental)
        new_embs = self._encode_vlm(imgs, texts)
        if self.distance == "cosine":
            new_embs = F.normalize(new_embs, dim=-1)
        
        # Compute radii if using radius-based retrieval
        if self.auto_k_method == "radius":
            new_radii = []
            for i, (im, tx) in enumerate(zip(imgs, texts)):
                r = self._estimate_radius(new_embs[i:i+1], im, tx)
                new_radii.append(r)
            new_radii = torch.tensor(new_radii, dtype=torch.float32)
            if self.key_radii is None:
                self.key_radii = new_radii
            else:
                self.key_radii = torch.cat([self.key_radii, new_radii])
        
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
    def _otsu_k(sims, top_n=50, n_bins=50):
        """Find threshold that minimizes intra-class variance (good vs bad matches)."""
        sims = np.asarray(sims, dtype=float)
        if sims.size < 2:
            return 0
        # Only analyze top-N scores (same as Grubbs)
        sims = np.sort(sims)[::-1][:top_n]
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
        - "radius": Handled separately in _retrieve_chain (returns None here)
        """
        if self.auto_k_method == "radius":
            return None  # Handled separately
        elif self.auto_k_method == "grubbs":
            return self._grubbs_k(scores, top_n=self.top_n)
        elif self.auto_k_method == "otsu":
            return self._otsu_k(scores, top_n=self.top_n)
        elif self.auto_k_method == "ensemble":
            k_grubbs = self._grubbs_k(scores, top_n=self.top_n)
            k_otsu = self._otsu_k(scores, top_n=self.top_n)
            # Both must agree there are outliers; take the more conservative (smaller k)
            if k_grubbs == 0 or k_otsu == 0:
                return 0
            return min(k_grubbs, k_otsu)
        else:
            raise ValueError(f"Unknown auto_k_method: {self.auto_k_method}")

    @torch.no_grad()
    def _retrieve_chain(self, image, start_text=""):
        """Retrieve chain starting from <image, start_text>.
        
        Entry selection methods:
        - grubbs/otsu/ensemble: statistical detection of significant outliers
        - radius: enter chains where query distance ≤ key's radius
        
        Supports L2 distance (default) or cosine similarity.
        """
        if self.key_embs is None or len(self.codebook) == 0:
            return []

        # Encode query
        q_emb = self._encode_vlm([image], [start_text])
        
        # Compute distances/scores
        if self.distance == "cosine":
            q_emb = F.normalize(q_emb, dim=-1)
            sims = (q_emb @ self.key_embs.t()).squeeze(0).float().cpu().numpy()
            scores = sims  # Higher is better
            dists = 1 - sims  # For radius comparison
        else:
            dists = torch.norm(self.key_embs.float() - q_emb.float(), dim=-1).cpu().numpy()
            scores = -dists  # Higher is better (negative distance)
        
        # Select entry points based on method
        if self.auto_k_method == "radius":
            # Radius-based: enter chains where dist <= radius
            if self.key_radii is None:
                return []
            radii = self.key_radii.cpu().numpy()
            in_radius = dists <= radii
            if not in_radius.any():
                return []
            # Get indices within radius, sorted by distance (closest first)
            entry_indices = np.where(in_radius)[0]
            entry_indices = entry_indices[np.argsort(dists[entry_indices])][:self.cap_k].tolist()
        else:
            # Auto-k detection (grubbs, otsu, ensemble)
            k = self._auto_k(scores)
            if k == 0:
                return []
            k = min(k, self.cap_k)
            entry_indices = np.argsort(scores)[::-1][:k].tolist()
        
        # Collect from all entry points
        collected = []
        seen = set()
        
        for entry_idx in entry_indices:
            idx = entry_idx
            while True:
                entry = self.codebook[idx]
                for sent in entry["retrieve"]:
                    if sent and sent not in seen:
                        seen.add(sent)
                        collected.append(sent)
                
                downstream = entry["downstream_idx"]
                if not downstream:
                    break
                
                # Pick downstream key closest to query
                sub_scores = scores[downstream]
                best_sub = int(np.argmax(sub_scores))
                idx = downstream[best_sub]
        
        return collected

    def _retrieve(self, image, question=""):
        """Single-call retrieval for API compatibility (two routes merged)."""
        facts1 = self._retrieve_chain(image, "")
        facts2 = self._retrieve_chain(image, question) if question else []
        seen = set(facts1)
        return facts1 + [f for f in facts2 if f not in seen]

    @torch.no_grad()
    def plot_score_distribution(self, image, question="", ax=None):
        """Plot histogram of scores with Grubbs/Otsu k thresholds as vertical lines.
        
        Raw keys in blue, augmented keys in orange.
        """
        if self.key_embs is None:
            return
        
        # Compute scores
        q_emb = self._encode_vlm([image], [question])
        if self.distance == "cosine":
            q_emb = F.normalize(q_emb, dim=-1)
            scores = (q_emb @ self.key_embs.t()).squeeze(0).float().cpu().numpy()
        else:
            scores = -torch.norm(self.key_embs.float() - q_emb.float(), dim=-1).cpu().numpy()
        
        k_grubbs = self._grubbs_k(scores, top_n=self.top_n)
        k_otsu = self._otsu_k(scores, top_n=self.top_n)
        sorted_scores = np.sort(scores)[::-1]
        
        # Split by augmentation
        is_aug = np.array([e.get("is_aug", False) for e in self.codebook])
        raw_scores = scores[~is_aug]
        aug_scores = scores[is_aug]
        
        show = ax is None
        if show:
            _, ax = plt.subplots(figsize=(6, 4))
        
        # Side-by-side histograms using seaborn
        import seaborn as sns
        import pandas as pd
        df = pd.DataFrame({
            'score': np.concatenate([raw_scores, aug_scores]),
            'type': ['raw'] * len(raw_scores) + ['aug'] * len(aug_scores)
        })
        sns.histplot(data=df, x='score', hue='type', multiple='dodge', bins=50, shrink=0.9,
                     palette={'raw': 'lightblue', 'aug': 'orange'}, ax=ax, edgecolor='black')
        
        if k_grubbs > 0:
            ax.axvline(sorted_scores[k_grubbs-1], color='red', linestyle='--', lw=2, label=f'Grubbs k={k_grubbs}')
        if k_otsu > 0:
            ax.axvline(sorted_scores[k_otsu-1], color='green', linestyle=':', lw=2, label=f'Otsu k={k_otsu}')
        ax.set_title(f'n_keys={len(self.codebook)}')
        ax.legend(fontsize=7)
        
        if show:
            plt.tight_layout()
            plt.show()

    def apply_to_dataset(self, dataset):
        """Apply retrieved facts to dataset prompts (two routes)."""
        applied = 0
        log = []
        data = getattr(dataset, "data", [])
        
        for ex in data:
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
        
        # Plot random 10 samples in 2x5 grids (both routes) if switch is on
        if self.plot_k_dist and data:
            import random
            samples = random.sample(data, min(10, len(data)))
            
            # Route 1: <img, "">
            fig1, axes1 = plt.subplots(2, 5, figsize=(20, 5))
            fig1.suptitle('Route: <img, "">', fontsize=14)
            for i, ax in enumerate(axes1.flatten()):
                if i < len(samples) and samples[i].get("image"):
                    self.plot_score_distribution(samples[i]["image"], "", ax=ax)
                else:
                    ax.axis('off')
            plt.tight_layout()
            plt.show()
            
            # Route 2: <img, question>
            fig2, axes2 = plt.subplots(2, 5, figsize=(20, 5))
            fig2.suptitle('Route: <img, question>', fontsize=14)
            for i, ax in enumerate(axes2.flatten()):
                if i < len(samples) and samples[i].get("image"):
                    self.plot_score_distribution(samples[i]["image"], samples[i].get("question", ""), ax=ax)
                else:
                    ax.axis('off')
            plt.tight_layout()
            plt.show()

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
        """Save codebook, embeddings, and radii to disk."""
        torch.save({
            "codebook": self.codebook,
            "key_embs": self.key_embs,
            "key_radii": self.key_radii
        }, path)
        print(f"[IKE_CHAIN] saved {len(self.codebook)} keys to {path}", flush=True)
    
    def load_index(self, path):
        """Load codebook, embeddings, and radii from disk."""
        data = torch.load(path, map_location=self.device)
        self.codebook = data["codebook"]
        self.key_embs = data["key_embs"].to(self.device)
        self.key_radii = data.get("key_radii")
        if self.key_radii is not None:
            self.key_radii = self.key_radii.to(self.device)
        print(f"[IKE_CHAIN] loaded {len(self.codebook)} keys from {path}", flush=True)

    def get_stats(self):
        """Return statistics about stored keys."""
        stats = {
            "num_keys": len(self.codebook),
            "num_edits": len(self._added_uids),
            "emb_size_mb": self.key_embs.numel() * 2 / 1024 / 1024 if self.key_embs is not None else 0,
            "auto_k_method": self.auto_k_method,
        }
        if self.key_radii is not None:
            stats["avg_radius"] = float(self.key_radii.mean())
            stats["radius_method"] = self.radius_method
        return stats

    def plot_codebook(self, max_edits=20, figsize=(5, 3)):
        """Plot force-directed network of keys based on pairwise L2 distance.
        
        Args:
            max_edits: Maximum number of edits to include (samples random edits if more)
        
        Colors by edit index, shapes: circle=raw, triangle=augmented.
        """
        import networkx as nx
        
        if self.key_embs is None or len(self.codebook) == 0:
            print("[IKE_CHAIN] No keys to plot")
            return
        
        # Get unique edit indices and sample if needed
        all_edit_indices = set(e.get("edit_idx", 0) for e in self.codebook)
        if len(all_edit_indices) > max_edits:
            import random
            selected_edits = set(random.sample(list(all_edit_indices), max_edits))
        else:
            selected_edits = all_edit_indices
        
        # Get all keys belonging to selected edits
        indices = np.array([i for i, e in enumerate(self.codebook) if e.get("edit_idx", 0) in selected_edits])
        embs = self.key_embs[indices].float().cpu().numpy()
        
        # Pairwise L2 distances -> similarity weights
        dists = np.linalg.norm(embs[:, None] - embs[None, :], axis=-1)
        sims = 1 / (1 + dists)
        
        # Build graph
        G = nx.Graph()
        for i, idx in enumerate(indices):
            G.add_node(i, 
                       is_aug=self.codebook[idx].get("is_aug", False),
                       edit_idx=self.codebook[idx].get("edit_idx", 0))
        
        # Add edges (only keep stronger connections for cleaner layout)
        thresh = np.percentile(sims[np.triu_indices(len(indices), k=1)], 50)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if sims[i, j] > thresh:
                    G.add_edge(i, j, weight=sims[i, j])
        
        # Spring layout
        pos = nx.spring_layout(G, weight='weight', seed=42, k=2/np.sqrt(len(indices)))
        
        # Split nodes by aug status
        raw_nodes = [i for i in G.nodes if not G.nodes[i]['is_aug']]
        aug_nodes = [i for i in G.nodes if G.nodes[i]['is_aug']]
        
        # Colors by edit index
        edit_indices = {i: G.nodes[i]['edit_idx'] for i in G.nodes}
        n_edits = len(selected_edits)
        cmap = plt.cm.get_cmap('tab20', n_edits)
        raw_colors = [cmap(edit_indices[i] % 20) for i in raw_nodes]
        aug_colors = [cmap(edit_indices[i] % 20) for i in aug_nodes]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.1, ax=ax)
        # Raw nodes: circles
        nx.draw_networkx_nodes(G, pos, nodelist=raw_nodes, node_color=raw_colors, node_size=20, alpha=0.8, node_shape='o', ax=ax)
        # Aug nodes: triangles
        nx.draw_networkx_nodes(G, pos, nodelist=aug_nodes, node_color=aug_colors, node_size=20, alpha=0.8, node_shape='^', ax=ax)
        
        # Legend
        ax.scatter([], [], c='gray', s=15, marker='o', label=f'raw ({len(raw_nodes)})')
        ax.scatter([], [], c='gray', s=15, marker='^', label=f'aug ({len(aug_nodes)})')
        ax.legend(loc='lower left', fontsize=5, markerscale=0.7)
        ax.set_title(f'Codebook Space ({n_edits} edits)', fontsize=10)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
