import re
from itertools import combinations
import numpy as np
import torch
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
        self.prefix = getattr(cfg, "cot_prefix", "")
        self.max_subset_size = int(getattr(cfg, "max_subset_size", 1))  # max sentences per k3 subset
        
        # Radius estimation method: "augment" (percentile of augmented images) or "balancedit" (pos/neg samples)
        self.radius_method = getattr(cfg, "radius_method", "balancedit") # "augment"
        # For "augment" method
        self.n_radius_samples = int(getattr(cfg, "n_radius_samples", 50))
        self.radius_percentile = float(getattr(cfg, "radius_percentile", 99))
        # For "balancedit" method
        self.balancedit_alpha = float(getattr(cfg, "balancedit_alpha", 0.5))
        self.n_positive_samples = int(getattr(cfg, "n_positive_samples", 10))

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

        # Prototype storage - optimized for large edit sets
        self._proto_keys = []      # List of [1, D] tensors (will stack for retrieval)
        self._proto_radii = []     # List of floats (radius per key)
        self._proto_keys_stacked = None  # [N, D] tensor for fast retrieval
        self._proto_radii_stacked = None  # [N] tensor for fast retrieval
        self._proto_sentences = []  # List of sentence lists
        self._added_uids = set()   # Track added edits to avoid duplicates

    def generate(self, *a, **kw):
        if hasattr(self.model, "generate"):
            return self.model.generate(*a, **kw)
        return self.wrapper.generate(*a, **kw)

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def __call__(self, *a, **kw):
        return self.forward(*a, **kw)

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
        # Mean pool over sequence dimension
        return act.mean(dim=1)  # [B, D]

    def _parse_rationale(self, rat):
        """Split rationale into sentences."""
        if not rat:
            return []
        return [p.strip() for p in re.split(r"(?<=[.!?])\s+", rat.strip()) if p.strip()]

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
        dists = torch.norm(aug_keys - key, dim=-1).numpy()  # [N]
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

        # Combined: ε = (1 - α) * d(Pos, k) + α * d(Neg, k)
        return (1 - self.balancedit_alpha) * d_pos + self.balancedit_alpha * d_neg

    @torch.no_grad()
    def _add_prototype(self, image, question, rationale, sentences=None):
        """Add prototypes for a single edit with radius estimation."""
        if sentences is None:
            sentences = self._parse_rationale(rationale)
        if not sentences:
            return

        def add_key_with_radius(img, text, sents):
            """Store key and estimate radius using configured method."""
            key = self._encode_vlm([img], [text]).cpu()  # [1, D]
            if self.radius_method == "balancedit":
                radius = self._estimate_radius_balancedit(key, img, text)
            else:
                radius = self._estimate_radius_augment(key, img, text)
            self._proto_keys.append(key)
            self._proto_radii.append(radius)
            self._proto_sentences.append(sents)

        # k1: <img, question> → all sentences
        add_key_with_radius(image, question, sentences)

        # k2: <img, ""> → all sentences
        add_key_with_radius(image, "", sentences)

        # k3: <img, rationale_subset> → subset sentences
        n = len(sentences)
        for size in range(1, min(n, self.max_subset_size) + 1):
            for subset in combinations(range(n), size):
                subset_sents = [sentences[i] for i in subset]
                text = " ".join(subset_sents)
                add_key_with_radius(image, text, subset_sents)

        # Invalidate stacked cache
        self._proto_keys_stacked = None
        self._proto_radii_stacked = None

    def _get_stacked(self):
        """Lazily build stacked key and radius tensors for efficient retrieval."""
        if self._proto_keys_stacked is None and self._proto_keys:
            self._proto_keys_stacked = torch.cat(self._proto_keys, dim=0)  # [N, D]
            self._proto_radii_stacked = torch.tensor(self._proto_radii)  # [N]
        return self._proto_keys_stacked, self._proto_radii_stacked

    @torch.no_grad()
    def _retrieve(self, image, question, top_k=5):
        """Retrieve facts using radius-based matching on top-k closest keys."""
        if not self._proto_keys:
            return []

        # Encode query
        query = self._encode_vlm([image], [question]).cpu()  # [1, D]
        keys, radii = self._get_stacked()  # [N, D], [N]

        # Batched L2 distance: [N]
        dists = torch.norm(keys - query, dim=-1)
        
        # Get top-k closest, then filter by radius (vectorized)
        top_k_actual = min(top_k, len(dists))
        top_indices = torch.topk(dists, k=top_k_actual, largest=False).indices
        mask = dists[top_indices] <= radii[top_indices]
        matched_indices = top_indices[mask].tolist()

        # Collect unique sentences from matched keys
        seen = set()
        results = []
        for idx in matched_indices:
            for sent in self._proto_sentences[idx]:
                if sent not in seen:
                    seen.add(sent)
                    results.append(sent)

        return results

    def apply_to_dataset(self, dataset):
        """Apply retrieval to prepend facts to prompts in dataset."""
        for ex in getattr(dataset, "data", []):
            # Store original prompt to avoid accumulation on repeated calls
            if "_original_prompt" not in ex:
                ex["_original_prompt"] = ex.get("prompt", "")
            
            prompt = ex["_original_prompt"]
            q = ex.get("question", "")
            img = ex.get("image")
            if not prompt or not q or img is None:
                continue
            facts = self._retrieve(img, q)
            if facts:
                ex["prompt"] = f"{self.prefix}{' '.join(facts)} {prompt}"
            else:
                ex["prompt"] = prompt  # Restore original if no facts

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add new edits as prototypes (no training required)."""
        if edit_ds is None:
            return self.model

        data = getattr(edit_ds, "data", [])
        added = 0
        for ex in data:
            # Use uid or (image, question) as unique key to avoid duplicates
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            if uid in self._added_uids:
                continue  # Skip already added edits
            
            rat = ex.get("cot") or ex.get("rationale") or ""
            q = ex.get("question", "")
            img = ex.get("image")
            if not rat or img is None or not q:
                continue
            self._add_prototype(img, q, rat)
            self._added_uids.add(uid)
            added += 1

        print(f"[IKE_PROTO] Added {added} new edits, total prototypes: {len(self._proto_keys)}")

        # Apply to dataset
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
            "radius_method": self.radius_method,
            "avg_radius": round(avg_radius, 4),
            "memory_mb": round(mem_mb, 2),
        }

