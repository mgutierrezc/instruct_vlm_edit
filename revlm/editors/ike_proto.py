import re
import random
from itertools import combinations
import numpy as np
from PIL import Image as PILImage
from scipy.stats import t as t_dist
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from .utils import brackets_to_periods, parent_module


class Augmenter:
    """Online augmentation for images and questions."""

    def __init__(self, wrapper=None):
        self.wrapper = wrapper
        self.img_aug = T.Compose([
            T.RandomResizedCrop(size=(384, 384), scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ])
        self._blank = PILImage.new("RGB", (364, 364), color="black")

    def image(self, img):
        """Apply random image augmentations."""
        if isinstance(img, str):
            img = PILImage.open(img).convert("RGB")
        elif hasattr(img, "convert"):
            img = img.convert("RGB")
        return self.img_aug(img)

    def question(self, q):
        """Rephrase question using VLM."""
        if not self.wrapper or not q:
            return q
        prompt = f"Rephrase this question differently while keeping the same meaning:\n\n{q}\n\nRephrased:"
        try:
            out = self.wrapper.generate([self._blank], [prompt], max_new_tokens=64, temperature=0.7)[0]
            out = str(out).strip()
            return out if out else q
        except Exception:
            return q


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
        self.k = int(getattr(cfg, "k", -3))  # negative = auto
        self.prefix = getattr(cfg, "cot_prefix", "")
        self.sim_threshold = float(getattr(cfg, "sim_threshold", 0.0))  # min sim to retrieve
        self.max_subset_size = int(getattr(cfg, "max_subset_size", 3))  # max sentences per k3 subset
        self.augment_keys = dict(getattr(cfg, "augment_keys", [("k1", 3), ("k2", 0), ("k3", 0)]))
        self.distance = getattr(cfg, "distance", "l2")  # "cosine" or "l2"

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
        self._proto_keys_stacked = None  # [N, D] tensor for fast retrieval
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

    @staticmethod
    def _auto_k(sims, top_k=20, alpha=0.05):
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
        # Mean pool over sequence dimension
        return act.mean(dim=1)  # [B, D]

    def _parse_rationale(self, rat):
        """Split rationale into sentences."""
        if not rat:
            return []
        return [p.strip() for p in re.split(r"(?<=[.!?])\s+", rat.strip()) if p.strip()]

    @torch.no_grad()
    def _add_prototype(self, image, question, rationale, sentences=None):
        """Add prototypes for a single edit with query fusion + subset expansion + augmentation."""
        if sentences is None:
            sentences = self._parse_rationale(rationale)
        if not sentences:
            return

        def add_key(img, text):
            k = self._encode_vlm([img], [text]).cpu()  # store raw
            self._proto_keys.append(k)
            self._proto_sentences.append(sentences)

        # k1: <img, question>
        add_key(image, question)

        # k2: <img, "">
        add_key(image, "")

        # k3: <img, rationale_subset> for subsets up to max_subset_size
        k3_texts = []
        n = len(sentences)
        for size in range(1, min(n, self.max_subset_size) + 1):
            for subset in combinations(range(n), size):
                text = " ".join(sentences[i] for i in subset)
                k3_texts.append(text)
                add_key(image, text)

        # Augmented keys: augment_keys = {"k1": n1, "k2": n2, "k3": n3}
        if self.augmenter:
            for _ in range(self.augment_keys.get("k1", 0)):
                add_key(self.augmenter.image(image), self.augmenter.question(question))
            for _ in range(self.augment_keys.get("k2", 0)):
                add_key(self.augmenter.image(image), "")
            for _ in range(self.augment_keys.get("k3", 0)):
                aug_img = self.augmenter.image(image)
                for text in k3_texts:
                    add_key(aug_img, text)

        # Invalidate stacked cache
        self._proto_keys_stacked = None

    def _get_stacked_keys(self):
        """Lazily build stacked key tensor for efficient retrieval."""
        if self._proto_keys_stacked is None and self._proto_keys:
            self._proto_keys_stacked = torch.cat(self._proto_keys, dim=0)  # [N, D]
        return self._proto_keys_stacked

    @torch.no_grad()
    def _retrieve(self, image, question, k=None):
        """Retrieve facts using nearest prototype search (batched for efficiency)."""
        if not self._proto_keys:
            return []

        if k is None:
            k = self.k

        # Encode query and get keys (both raw)
        query = self._encode_vlm([image], [question]).cpu()  # [1, D]
        keys = self._get_stacked_keys()  # [N, D]

        # Compute similarity/distance (normalize on-the-fly for cosine)
        if self.distance == "cosine":
            q = F.normalize(query, dim=-1)
            k = F.normalize(keys, dim=-1)
            sims_np = (q @ k.t()).squeeze(0).numpy()  # higher = better
        else:  # l2
            sims_np = -torch.norm(keys - query, dim=-1).numpy()  # negative L2, higher = better

        # Apply threshold filter (only for cosine)
        if self.distance == "cosine" and self.sim_threshold > 0:
            if not (sims_np > self.sim_threshold).any():
                return []

        # Auto k if k < 0
        if k < 0:
            k = self._auto_k(sims_np)
            if k == 0:
                return []
            k = min(k, abs(self.k))  # cap at |self.k|

        # Get top-k prototypes (highest sims)
        top_indices = np.argsort(sims_np)[::-1][:k]

        # Collect unique sentences from top prototypes
        seen = set()
        results = []
        for idx in top_indices:
            if self.distance == "cosine" and sims_np[idx] < self.sim_threshold:
                continue
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
            facts = self._retrieve(img, q, self.k)
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
        # Estimate memory: each key is [1, D] float32
        dim = self._proto_keys[0].shape[-1] if self._proto_keys else 0
        mem_mb = (n_protos * dim * 4) / (1024 * 1024)
        return {
            "num_prototypes": n_protos,
            "num_edits": n_edits,
            "protos_per_edit": n_protos / max(1, n_edits),
            "max_subset_size": self.max_subset_size,
            "memory_mb": round(mem_mb, 2),
        }

