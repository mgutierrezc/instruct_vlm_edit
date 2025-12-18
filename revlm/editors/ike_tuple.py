import re
import random
import numpy as np
from PIL import Image as PILImage
from scipy.stats import t as t_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from sentence_transformers import SentenceTransformer
from .utils import brackets_to_periods, parent_module


class Augmenter:
    """Online augmentation for images, questions, and rationales."""

    def __init__(self, wrapper=None):
        self.wrapper = wrapper
        self.img_aug = T.Compose([
            T.RandomResizedCrop(size=(384, 384), scale=(0.7, 1.0)),  # random crop 70-100%
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ])
        self._blank = PILImage.new("RGB", (364, 364), color="black")

    def image(self, img):
        """Apply random image augmentations. Handles path strings or PIL Images."""
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

    def rationale(self, sent):
        """Rephrase rationale sentence using VLM."""
        if not self.wrapper or not sent:
            return sent
        prompt = f"Rephrase this fact differently while keeping the same meaning:\n\n{sent}\n\nRephrased:"
        try:
            out = self.wrapper.generate([self._blank], [prompt], max_new_tokens=64, temperature=0.7)[0]
            out = str(out).strip()
            return out if out else sent
        except Exception:
            return sent


class IKE_TUPLE(nn.Module):
    """Tuple retriever with multi-positive InfoNCE and global negatives."""

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        cfg = getattr(config, "editor", config)

        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))

        # Hyperparams
        self.k = int(getattr(cfg, "k", -3))
        self.clip_dim = int(getattr(cfg, "clip_dim", 512))
        self.num_epochs = int(getattr(cfg, "clip_epochs", 1000))
        self.batch_size = int(getattr(cfg, "clip_batch_size", 10))
        self.lr = float(getattr(cfg, "clip_lr", 1e-4))
        self.temperature = float(getattr(cfg, "clip_temperature", 1.0))
        self.early_stop_acc = float(getattr(cfg, "early_stop_acc", 0.975))
        self.early_stop_acc_last = float(getattr(cfg, "early_stop_acc_last", 0.99))
        self.prefix = getattr(cfg, "cot_prefix", "New Fact: ")
        self.use_augment = bool(getattr(cfg, "use_augment", True))
        self.use_counterfacts = bool(getattr(cfg, "use_counterfacts", False))

        # Augmenter (online, per-batch)
        self.augmenter = Augmenter(self.wrapper) if self.use_augment else None

        # Sentence model
        self.sentence_model = SentenceTransformer(
            getattr(cfg, "sentence_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        ).to(self.device).eval()
        self.txt_dim = self.sentence_model.get_sentence_embedding_dimension()

        # Hook setup
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

        # Heads and index
        self.image_proj = None
        self.text_proj = None
        self.rationale_texts = []
        self.rationale_emb = None

    def generate(self, *a, **kw):
        if hasattr(self.model, "generate"):
            return self.model.generate(*a, **kw)
        return self.wrapper.generate(*a, **kw)

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    @torch.no_grad()
    def _encode_texts(self, texts):
        return self.sentence_model.encode(texts, convert_to_tensor=True, show_progress_bar=False).to(self.device, dtype=torch.float32).clone()

    def _encode_vlm(self, images, questions):
        self.model.eval()
        self._last_act = None
        inputs = self.wrapper.encode(images, questions, tokenize=False)
        with torch.no_grad():
            self.model(**inputs)
        act = self._last_act
        if act is None:
            raise RuntimeError("Hook failed")
        return (act.unsqueeze(0) if act.dim() == 2 else act).mean(dim=1).to(self.device, dtype=torch.float32).clone()

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

    def _ensure_heads(self, img_dim):
        if self.image_proj is None:
            self.image_proj = nn.Sequential(
                nn.Linear(img_dim, self.clip_dim), nn.GELU(),
                nn.Linear(self.clip_dim, self.clip_dim), nn.LayerNorm(self.clip_dim)
            ).to(self.device)
        if self.text_proj is None:
            self.text_proj = nn.Sequential(
                nn.Linear(self.txt_dim, self.clip_dim), nn.GELU(),
                nn.Linear(self.clip_dim, self.clip_dim), nn.LayerNorm(self.clip_dim)
            ).to(self.device)

    def _gen_counterfact(self, sent):
        """Generate a counterfactual sentence using VLM."""
        if not self.wrapper:
            return None
        blank = PILImage.new("RGB", (364, 364), color="black")
        prompt = f"Rewrite this fact to state something different but plausible:\n\n{sent}\n\nRewritten:"
        try:
            out = self.wrapper.generate([blank], [prompt], max_new_tokens=64, temperature=0.7)[0]
            out = str(out).strip()
            return out if out and out.lower() != sent.lower() else None
        except Exception:
            return None

    def _build_samples(self, dataset):
        """Build samples: each has image, question, sentences, and counterfacts."""
        data = getattr(dataset, "data", [])
        samples = []
        for ex in data:
            rat = ex.get("cot") or ex.get("rationale") or ""
            q, img = ex.get("question", ""), ex.get("image")
            if not rat or img is None or not q:
                continue
            sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+", rat.strip()) if p.strip()]
            if not sentences:
                continue
            # Generate 1 counterfact per sentence
            counterfacts = []
            if self.use_counterfacts:
                for s in sentences:
                    cf = self._gen_counterfact(s)
                    if cf:
                        counterfacts.append(cf)
            samples.append({"image": img, "question": q, "sentences": sentences, "counterfacts": counterfacts})
        return samples

    @torch.no_grad()
    def _retrieval_acc(self, samples, last_only=False):
        """Compute retrieval accuracy: fraction of retrieved sentences that match sample's own."""
        if not samples or self.rationale_emb is None:
            return 0.0
        target = [samples[-1]] if last_only else samples
        hits, total = 0, 0
        for s in target:
            facts = self._retrieve(s["image"], s["question"], self.k)
            gt = set(s["sentences"])
            hits += sum(1 for f in facts if f in gt)
            total += len(facts)
        return hits / max(1, total)

    def _train(self, samples):
        """Train with multi-positive InfoNCE: 3 query types per sample vs global pool."""
        if not samples:
            return
        opt, sched = None, None
        n = len(samples)
        for ep in range(self.num_epochs):
            perm = torch.randperm(n)
            loss_sum, cnt = 0.0, 0
            for start in range(0, n, self.batch_size):
                batch = [samples[i] for i in perm[start:start + self.batch_size].tolist()]
                B = len(batch)
                if B < 2:
                    continue

                # 3 query types: <img,q>, <img>, <img,rationale>
                imgs = [b["image"] for b in batch]
                qs = [b["question"] for b in batch]
                rats = [" ".join(b["sentences"]) for b in batch]

                # Apply augmentations (online, per-batch)
                if self.augmenter:
                    imgs = [self.augmenter.image(img) for img in imgs]
                    qs = [self.augmenter.question(q) if random.random() < 0.5 else q for q in qs]
                    rats = [self.augmenter.rationale(r) if random.random() < 0.5 else r for r in rats]

                f1 = self._encode_vlm(imgs, qs)           # <image, question>
                f2 = self._encode_vlm(imgs, [""] * B)     # <image> only
                f3 = self._encode_vlm(imgs, rats)         # <image, rationale>

                self._ensure_heads(f1.shape[-1])
                q_emb = torch.cat([
                    F.normalize(self.image_proj(f1), dim=-1),
                    F.normalize(self.image_proj(f2), dim=-1),
                    F.normalize(self.image_proj(f3), dim=-1),
                ], dim=0)  # [3*B, D]
                q_owners = list(range(B)) * 3  # sample ownership for each query

                # Pool all sentences with owner ids (-1 for counterfacts = always negative)
                all_texts, owners = [], []
                for i, b in enumerate(batch):
                    for s in b["sentences"]:
                        all_texts.append(s)
                        owners.append(i)
                    for cf in b.get("counterfacts", []):
                        all_texts.append(cf)
                        owners.append(-1)

                if len(all_texts) < 2:
                    continue

                t_emb = F.normalize(self.text_proj(self._encode_texts(all_texts)), dim=-1)
                owners_t = torch.tensor(owners, device=self.device)

                logits = (q_emb @ t_emb.t()) / self.temperature  # [3*B, T]

                # Multi-positive InfoNCE for all 3*B queries
                loss = torch.tensor(0.0, device=self.device)
                valid = 0
                for i in range(3 * B):
                    pos_mask = (owners_t == q_owners[i])
                    if not pos_mask.any():
                        continue
                    loss_i = -torch.logsumexp(logits[i, pos_mask], dim=0) + torch.logsumexp(logits[i], dim=0)
                    loss = loss + loss_i
                    valid += 1

                if valid == 0:
                    continue
                loss = loss / valid

                if opt is None:
                    opt = torch.optim.Adam(
                        list(self.image_proj.parameters()) + list(self.text_proj.parameters()),
                        lr=self.lr
                    )
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.num_epochs))
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += loss.item()
                cnt += 1

            if sched:
                sched.step()
            self._build_index(samples)
            acc = self._retrieval_acc(samples)
            acc_last = self._retrieval_acc(samples, last_only=True)
            k_str = "auto" if self.k < 0 else str(self.k)
            print(f"[IKE_TUPLE] epoch {ep+1}/{self.num_epochs} loss: {loss_sum/max(1,cnt):.4f} acc@{k_str}: {acc:.3f} acc_last: {acc_last:.3f}")
            if acc >= self.early_stop_acc and acc_last >= self.early_stop_acc_last:
                print(f"[IKE_TUPLE] early stop at acc {acc:.3f}, acc_last {acc_last:.3f}")
                break
        self._build_index(samples)

    @torch.no_grad()
    def _build_index(self, samples):
        if not samples or self.text_proj is None:
            return
        # Collect unique sentences
        texts = []
        for s in samples:
            texts.extend(s["sentences"])
        texts = list(dict.fromkeys(texts))
        if texts:
            self.rationale_texts = texts
            self.rationale_emb = F.normalize(self.text_proj(self._encode_texts(texts)), dim=-1)

    @torch.no_grad()
    def _retrieve(self, image, question, k):
        if self.rationale_emb is None:
            return []
        img_e = F.normalize(self.image_proj(self._encode_vlm([image], [question])), dim=-1)
        sims = (self.rationale_emb @ img_e.t()).squeeze(-1)
        sims_np = sims.cpu().numpy()
        # Auto k if k < 0, capped at 3
        if k < 0:
            k = self._auto_k(sims_np)
            if k == 0:
                return []
            k = min(k, 3)
        topk = torch.topk(sims, k=min(k, len(sims)), largest=True)
        return [self.rationale_texts[i] for i in topk.indices.tolist()]

    def apply_to_dataset(self, dataset):
        for ex in getattr(dataset, "data", []):
            prompt, q, img = ex.get("prompt", ""), ex.get("question", ""), ex.get("image")
            if not prompt or not q or img is None:
                continue
            facts = self._retrieve(img, q, self.k)
            if facts:
                ex["prompt"] = f"{self.prefix}{' '.join(facts)} {prompt}"

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        if edit_ds is None:
            return self.model
        # Force counterfacts on for few edits
        if len(getattr(edit_ds, "data", [])) < 10:
            self.use_counterfacts = True
        samples = self._build_samples(edit_ds)
        self._train(samples)
        self.apply_to_dataset(edit_ds)
        return self.model
