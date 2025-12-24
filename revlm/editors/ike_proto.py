import re
from itertools import combinations
import numpy as np
from scipy.stats import t as t_dist
import torch
import torch.nn.functional as F
from .utils import brackets_to_periods, parent_module, Augmenter


class IKE_PROTO:
    """Prototype-based retriever for sequential editing without catastrophic forgetting.
    
    Instead of training projectors, stores frozen VLM embeddings as prototypes.
    Uses multi-prototype storage (3 query variants + augmentation) for generality.
    """

    def __init__(self, config, model):
        self.config = config
        cfg = getattr(config, "editor", config)

        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))

        # Hyperparams
        self.k = int(getattr(cfg, "k", -1))  # negative = auto
        self.prefix = getattr(cfg, "cot_prefix", "")
        self.sim_threshold = float(getattr(cfg, "sim_threshold", 0.0))  # min sim to retrieve
        self.max_subset_size = int(getattr(cfg, "max_subset_size", 1))  # max sentences per k3 subset
        self.augment_keys = dict(getattr(cfg, "augment_keys", [("k1", 0), ("k2", 0), ("k3", 0)]))
        self.subset_sentences = getattr(cfg, "subset_sentences", False)  # True=subset-specific, False=all sentences
        self.distance = getattr(cfg, "distance", "l2")  # "cosine" or "l2"

        # Radius estimation config
        self.radius_method = getattr(cfg, "radius_method", "balancedit")  # "none", "augment", "balancedit"
        self.use_radius_in_auto_k = getattr(cfg, "use_radius_in_auto_k", False)  # adjust sims by radius
        # For "augment" method
        self.n_radius_samples = int(getattr(cfg, "n_radius_samples", 50))
        self.radius_percentile = float(getattr(cfg, "radius_percentile", 99))
        # For "balancedit" method
        self.balancedit_alpha = float(getattr(cfg, "balancedit_alpha", 0.5))
        self.n_positive_samples = int(getattr(cfg, "n_positive_samples", 10))

        # Two-route retrieval mode
        self.two_route_mode = getattr(cfg, "two_route_mode", "intersect")  # "union" or "intersect"

        # Augmenter
        self.augmenter = Augmenter(self.wrapper)

        # Hook setup for VLM hidden states
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

        # Prototype storage
        self._proto_keys = []       # List of [1, D] tensors
        self._proto_radii = []      # List of floats
        self._proto_sentences = []  # List of sentence lists
        self._keys_stacked = None   # [N, D] cached tensor
        self._radii_stacked = None  # [N] cached tensor
        self._added_uids = set()    # Track added edits

    def generate(self, *a, **kw):
        if hasattr(self.model, "generate"):
            return self.model.generate(*a, **kw)
        return self.wrapper.generate(*a, **kw)

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def __call__(self, *a, **kw):
        return self.forward(*a, **kw)

    @staticmethod
    def _auto_k(sims, top_k=100, alpha=0.05):
        """Use Grubbs' test on similarity gaps to find natural cutoff."""
        sims = np.asarray(sims, dtype=float)
        if sims.size < 4:
            return 0
        vals = np.sort(sims)[::-1][:top_k]
        spread = vals[0] - vals[-1]
        if spread <= 0:
            return 0
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
        return (i + 1) if G > Gcrit else 0

    @torch.no_grad()
    def _encode_vlm(self, images, questions):
        """Encode (image, question) pairs using VLM hidden states."""
        self.model.eval()
        self._last_act = None
        inputs = self.wrapper.encode(images, questions, tokenize=False)
        self.model(**inputs)
        act = self._last_act
        if act is None:
            raise RuntimeError("Hook failed to capture activations")
        act = (act.unsqueeze(0) if act.dim() == 2 else act).to(self.device, dtype=torch.float32)
        return act.mean(dim=1)  # [B, D]

    def _parse_rationale(self, rat):
        """Split rationale into sentences."""
        if not rat:
            return []
        return [p.strip() for p in re.split(r"(?<=[.!?])\s+", rat.strip()) if p.strip()]

    def _estimate_radius(self, key, img, text):
        """Estimate radius for a key using configured method."""
        if self.radius_method == "none":
            return 0.0
        elif self.radius_method == "augment":
            return self._estimate_radius_augment(key, img, text)
        elif self.radius_method == "balancedit":
            return self._estimate_radius_balancedit(key, img, text)
        return 0.0

    def _estimate_radius_augment(self, key, img, text):
        """Estimate radius via percentile of augmented image distances."""
        if self.n_radius_samples <= 0:
            return 0.0
        aug_keys = []
        for _ in range(self.n_radius_samples):
            aug_img = self.augmenter.image(img)
            aug_key = self._encode_vlm([aug_img], [text]).cpu()
            aug_keys.append(aug_key)
        aug_keys = torch.cat(aug_keys, dim=0)  # [N, D]
        dists = torch.norm(aug_keys - key, dim=-1).numpy()
        return float(np.percentile(dists, self.radius_percentile))

    def _estimate_radius_balancedit(self, key, img, text):
        """Estimate radius via positive (rephrased text) and negative (black image) samples."""
        # Positive: same image, rephrased text(s)
        pos_dists = []
        for _ in range(self.n_positive_samples):
            reph_text = self.augmenter.question(text) if text else ""
            pos_key = self._encode_vlm([img], [reph_text]).cpu()
            pos_dists.append(float(torch.norm(pos_key - key)))
        d_pos = float(np.median(pos_dists)) if pos_dists else 0.0

        # Negative: black image, same text
        neg_key = self._encode_vlm([self.augmenter._blank], [text]).cpu()
        d_neg = float(torch.norm(neg_key - key))

        # Combined: ε = (1 - α) * d_pos + α * d_neg
        return (1 - self.balancedit_alpha) * d_pos + self.balancedit_alpha * d_neg

    @torch.no_grad()
    def _add_prototype(self, image, question, rationale, sentences=None):
        """Add prototypes for a single edit with query fusion + subset expansion + augmentation."""
        if sentences is None:
            sentences = self._parse_rationale(rationale)
        if not sentences:
            return

        def add_key(img, text, sents):
            key = self._encode_vlm([img], [text]).cpu()
            radius = self._estimate_radius(key, img, text)
            self._proto_keys.append(key)
            self._proto_radii.append(radius)
            self._proto_sentences.append(sents)

        # k1: <img, question> → all sentences
        add_key(image, question, sentences)

        # k2: <img, ""> → all sentences
        add_key(image, "", sentences)

        # k3: <img, rationale_subset> → subset or all sentences based on config
        k3_subsets = []
        n = len(sentences)
        for size in range(1, min(n, self.max_subset_size) + 1):
            for subset in combinations(range(n), size):
                subset_sents = [sentences[i] for i in subset]
                text = " ".join(subset_sents)
                k3_subsets.append((text, subset_sents))
                add_key(image, text, subset_sents if self.subset_sentences else sentences)

        # Augmented keys
        if self.augmenter:
            for _ in range(self.augment_keys.get("k1", 0)):
                add_key(self.augmenter.image(image), self.augmenter.question(question), sentences)
            for _ in range(self.augment_keys.get("k2", 0)):
                add_key(self.augmenter.image(image), "", sentences)
            for _ in range(self.augment_keys.get("k3", 0)):
                aug_img = self.augmenter.image(image)
                for text, subset_sents in k3_subsets:
                    add_key(aug_img, text, subset_sents if self.subset_sentences else sentences)

        # Invalidate stacked cache
        self._keys_stacked = None
        self._radii_stacked = None

    def _get_stacked(self):
        """Lazily build stacked tensors for efficient retrieval."""
        if self._keys_stacked is None and self._proto_keys:
            self._keys_stacked = torch.cat(self._proto_keys, dim=0)  # [N, D]
            self._radii_stacked = torch.tensor(self._proto_radii)    # [N]
        return self._keys_stacked, self._radii_stacked

    def _select_by_radius(self, dists, radii, max_k):
        """Select keys where query is within radius, sorted by distance.
        Returns: indices array or None if no matches.
        """
        within = (dists <= radii).numpy()
        if not within.any():
            return None
        indices = np.where(within)[0]
        indices = indices[np.argsort(dists[indices].numpy())]
        return indices[:max_k] if max_k > 0 else indices

    @torch.no_grad()
    def _retrieve(self, image, question, k=None):
        """Two-stage retrieval: radius-based first, then auto_k fallback."""
        if not self._proto_keys:
            return []

        if k is None:
            k = self.k
        max_k = abs(k) if k != 0 else len(self._proto_keys)

        # Encode query
        query = self._encode_vlm([image], [question]).cpu()
        keys, radii = self._get_stacked()

        # Compute distances/similarities
        if self.distance == "cosine":
            q, kn = F.normalize(query, dim=-1), F.normalize(keys, dim=-1)
            sims = (q @ kn.t()).squeeze(0).numpy()
            dists = None
        else:
            dists = torch.norm(keys - query, dim=-1)
            sims = -dists.numpy()

        # Stage 1: Radius-based selection
        top_idx = None
        if self.distance == "l2" and self.radius_method != "none":
            top_idx = self._select_by_radius(dists, radii, max_k)

        # Stage 2: Auto_k fallback
        if top_idx is None:
            if self.distance == "cosine" and self.sim_threshold > 0 and not (sims > self.sim_threshold).any():
                return []
            
            sims_adj = (-dists + radii).numpy() if (self.use_radius_in_auto_k and dists is not None) else sims
            
            if k < 0:
                k = self._auto_k(sims_adj)
                if k == 0:
                    return []
                k = min(k, max_k)
            
            top_idx = np.argsort(sims_adj)[::-1][:k]

        # Collect unique sentences
        seen, results = set(), []
        for idx in top_idx:
            if self.distance == "cosine" and sims[idx] < self.sim_threshold:
                continue
            for sent in self._proto_sentences[idx]:
                if sent not in seen:
                    seen.add(sent)
                    results.append(sent)
        return results

    def apply_to_dataset(self, dataset):
        """Apply retrieval to prepend facts to prompts in dataset (two-route: image-only + image-question)."""
        for ex in getattr(dataset, "data", []):
            if "_original_prompt" not in ex:
                ex["_original_prompt"] = ex.get("prompt", "")
            
            prompt = ex["_original_prompt"]
            q = ex.get("question", "")
            img = ex.get("image")
            if not prompt or img is None:
                continue
            
            # Route 1: <image, ""> query
            facts1 = self._retrieve(img, "", self.k)
            # Route 2: <image, question> query
            facts2 = self._retrieve(img, q, self.k) if q else []
            
            # Combine based on mode
            if self.two_route_mode == "intersect":
                # Only facts in both routes
                facts = [f for f in facts1 if f in set(facts2)]
            else:  # union (default)
                # Merge unique, preserving order from route 1 first
                seen = set(facts1)
                facts = facts1 + [f for f in facts2 if f not in seen]
            
            if facts:
                ex["prompt"] = f"{self.prefix}{' '.join(facts)} {prompt}"
            else:
                ex["prompt"] = prompt

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add new edits as prototypes (no training required)."""
        if edit_ds is None:
            return self.model

        data = getattr(edit_ds, "data", [])
        added = 0
        for ex in data:
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            if uid in self._added_uids:
                continue
            
            rat = ex.get("cot") or ex.get("rationale") or ""
            q = ex.get("question", "")
            img = ex.get("image")
            if not rat or img is None or not q:
                continue
            self._add_prototype(img, q, rat)
            self._added_uids.add(uid)
            added += 1

        print(f"[IKE_PROTO] Added {added} new edits, total prototypes: {len(self._proto_keys)}")
        self.apply_to_dataset(edit_ds)
        return self.model

    def get_stats(self):
        """Return statistics about stored prototypes."""
        n_protos = len(self._proto_keys)
        n_edits = len(self._added_uids)
        dim = self._proto_keys[0].shape[-1] if self._proto_keys else 0
        mem_mb = (n_protos * dim * 4) / (1024 * 1024)
        avg_radius = float(np.mean(self._proto_radii)) if self._proto_radii else 0.0
        return {
            "num_prototypes": n_protos,
            "num_edits": n_edits,
            "protos_per_edit": n_protos / max(1, n_edits),
            "max_subset_size": self.max_subset_size,
            "radius_method": self.radius_method,
            "avg_radius": round(avg_radius, 4),
            "memory_mb": round(mem_mb, 2),
        }
