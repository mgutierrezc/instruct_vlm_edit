import re
from itertools import combinations
import numpy as np
from PIL import Image as PILImage
from scipy.stats import t as t_dist
import torch
from torchvision import transforms as T
from .utils import brackets_to_periods, parent_module

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
        self.augment_keys = dict(getattr(cfg, "augment_keys", [("k1", 3), ("k2", 0), ("k3", 0)]))
        self.bl_size = int(getattr(cfg, "bl_size", 10))  # edits per block
        self.max_k = getattr(cfg, "max_k", 3)  # max keys to retrieve, None = no cap (ckpt aokvqa)

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

        # Block-based prototype storage
        self._blocks = []  # List of {"keys": [N, D], "sentences": [...]}
        self._cur_keys = []  # Current block being built (list of [1, D])
        self._cur_sentences = []  # Current block sentences
        self._cur_edits = 0  # Edits in current block
        self._added_uids = set()  # Track added edits to avoid duplicates

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

    def _finalize_block(self):
        """Finalize current block and start a new one."""
        if self._cur_keys:
            self._blocks.append({
                "keys": torch.cat(self._cur_keys, dim=0),  # [N, D]
                "sentences": self._cur_sentences.copy(),
            })
            self._cur_keys = []
            self._cur_sentences = []
            self._cur_edits = 0

    @torch.no_grad()
    def _add_prototype(self, image, question, rationale, sentences=None):
        """Add prototypes for a single edit with query fusion + subset expansion + augmentation."""
        if sentences is None:
            sentences = self._parse_rationale(rationale)
        if not sentences:
            return

        def add_key(img, text, sents):
            k = self._encode_vlm([img], [text]).cpu()
            self._cur_keys.append(k)
            self._cur_sentences.append(sents)

        # k1: <img, question> → all sentences
        add_key(image, question, sentences)

        # k2: <img, ""> → all sentences
        add_key(image, "", sentences)

        # k3: <img, rationale_subset> → only subset sentences
        k3_subsets = []  # list of (text, subset_sentences)
        n = len(sentences)
        for size in range(1, min(n, self.max_subset_size) + 1):
            for subset in combinations(range(n), size):
                subset_sents = [sentences[i] for i in subset]
                text = " ".join(subset_sents)
                k3_subsets.append((text, subset_sents))
                add_key(image, text, subset_sents)

        # Augmented keys
        if self.augmenter:
            for _ in range(self.augment_keys.get("k1", 0)):
                add_key(self.augmenter.image(image), self.augmenter.question(question), sentences)
            for _ in range(self.augment_keys.get("k2", 0)):
                add_key(self.augmenter.image(image), "", sentences)
            for _ in range(self.augment_keys.get("k3", 0)):
                aug_img = self.augmenter.image(image)
                for text, subset_sents in k3_subsets:
                    add_key(aug_img, text, subset_sents)

        # Check if block is full
        self._cur_edits += 1
        if self._cur_edits >= self.bl_size:
            self._finalize_block()

    @torch.no_grad()
    def _retrieve(self, image, question):
        """Block-by-block retrieval with batched distance computation."""
        # Include current (incomplete) block
        all_blocks = self._blocks.copy()
        if self._cur_keys:
            all_blocks.append({
                "keys": torch.cat(self._cur_keys, dim=0),
                "sentences": self._cur_sentences.copy(),
            })
        if not all_blocks:
            return []

        # Encode query once
        query = self._encode_vlm([image], [question]).cpu()  # [1, D]

        # Batched distance: stack all keys, compute once
        all_keys = torch.cat([b["keys"] for b in all_blocks], dim=0)  # [Total, D]
        all_sims = -torch.norm(all_keys - query, dim=-1).numpy()  # [Total], L2 distance, higher = better

        # Build block boundaries
        sizes = [b["keys"].shape[0] for b in all_blocks]
        offsets = np.cumsum([0] + sizes)

        # Process each block using pre-computed distances
        all_matches = []
        for i, block in enumerate(all_blocks):
            start, end = offsets[i], offsets[i + 1]
            block_sims = all_sims[start:end]

            # Run auto_k on this block
            block_k = self._auto_k(block_sims, top_k=len(block_sims))
            if block_k == 0:
                continue

            # Get top matches from this block
            top_indices = np.argsort(block_sims)[::-1][:block_k]
            for idx in top_indices:
                all_matches.append((block_sims[idx], block["sentences"][idx]))

        if not all_matches:
            return []

        # Sort by similarity and take top max_k (None = keep all)
        all_matches.sort(key=lambda x: x[0], reverse=True)
        top_matches = all_matches if self.max_k is None else all_matches[:self.max_k]

        # Collect unique sentences
        seen = set()
        results = []
        for _, sentences in top_matches:
            for sent in sentences:
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

        n_blocks = len(self._blocks) + (1 if self._cur_keys else 0)
        n_protos = sum(b["keys"].shape[0] for b in self._blocks) + len(self._cur_keys)
        print(f"[IKE_PROTO] Added {added} edits, blocks: {n_blocks}, protos: {n_protos}")

        # Apply to dataset
        self.apply_to_dataset(edit_ds)
        return self.model

    def get_stats(self):
        """Return statistics about stored prototypes."""
        n_blocks = len(self._blocks) + (1 if self._cur_keys else 0)
        n_protos = sum(b["keys"].shape[0] for b in self._blocks) + len(self._cur_keys)
        n_edits = len(self._added_uids)
        dim = self._blocks[0]["keys"].shape[-1] if self._blocks else (
            self._cur_keys[0].shape[-1] if self._cur_keys else 0)
        mem_mb = (n_protos * dim * 4) / (1024 * 1024)
        return {
            "num_blocks": n_blocks,
            "num_prototypes": n_protos,
            "num_edits": n_edits,
            "protos_per_edit": n_protos / max(1, n_edits),
            "bl_size": self.bl_size,
            "memory_mb": round(mem_mb, 2),
        }


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


