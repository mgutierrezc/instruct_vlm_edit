import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from .utils import brackets_to_periods, parent_module


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
        self.k = int(getattr(cfg, "k", 1))
        self.clip_dim = int(getattr(cfg, "clip_dim", 512))
        self.num_epochs = int(getattr(cfg, "clip_epochs", 1000))
        self.batch_size = int(getattr(cfg, "clip_batch_size", 10))
        self.lr = float(getattr(cfg, "clip_lr", 1e-4))
        self.temperature = float(getattr(cfg, "clip_temperature", 1.0))
        self.early_stop_acc = float(getattr(cfg, "early_stop_acc", 0.9))
        self.prefix = getattr(cfg, "cot_prefix", "New Fact: ")

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

    def _build_samples(self, dataset):
        """Build samples: each has image, question, and list of rationale sentences."""
        data = getattr(dataset, "data", [])
        samples = []
        for ex in data:
            rat = ex.get("cot") or ex.get("rationale") or ""
            q, img = ex.get("question", ""), ex.get("image")
            if not rat or img is None or not q:
                continue
            sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+", rat.strip()) if p.strip()]
            if sentences:
                samples.append({"image": img, "question": q, "sentences": sentences})
        return samples

    @torch.no_grad()
    def _retrieval_acc(self, samples):
        """Compute retrieval accuracy: fraction of retrieved sentences that match sample's own."""
        if not samples or self.rationale_emb is None:
            return 0.0
        hits, total = 0, 0
        for s in samples:
            facts = self._retrieve(s["image"], s["question"], self.k)
            gt = set(s["sentences"])
            hits += sum(1 for f in facts if f in gt)
            total += len(facts)
        return hits / max(1, total)

    def _train(self, samples):
        """Train with multi-positive InfoNCE: each query's sentences vs global pool."""
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
                    continue  # need at least 2 samples for contrastive

                # Encode queries: one per sample
                img_f = self._encode_vlm([b["image"] for b in batch], [b["question"] for b in batch])
                self._ensure_heads(img_f.shape[-1])
                q_emb = F.normalize(self.image_proj(img_f), dim=-1)  # [B, D]

                # Pool all sentences with owner ids
                all_texts, owners = [], []
                for i, b in enumerate(batch):
                    for s in b["sentences"]:
                        all_texts.append(s)
                        owners.append(i)

                if len(all_texts) < 2:
                    continue

                # Encode all texts
                t_emb = F.normalize(self.text_proj(self._encode_texts(all_texts)), dim=-1)  # [T, D]
                owners_t = torch.tensor(owners, device=self.device)  # [T]

                # Similarity matrix: [B, T]
                logits = (q_emb @ t_emb.t()) / self.temperature

                # Multi-positive InfoNCE: L_i = -log(sum_pos exp) + log(sum_all exp)
                loss = torch.tensor(0.0, device=self.device)
                valid = 0
                for i in range(B):
                    pos_mask = (owners_t == i)
                    if not pos_mask.any():
                        continue
                    pos_logits = logits[i, pos_mask]
                    all_logits = logits[i]
                    loss_i = -torch.logsumexp(pos_logits, dim=0) + torch.logsumexp(all_logits, dim=0)
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
            print(f"[IKE_TUPLE] epoch {ep+1}/{self.num_epochs} loss: {loss_sum/max(1,cnt):.4f} acc@{self.k}: {acc:.3f}")
            if acc >= self.early_stop_acc:
                print(f"[IKE_TUPLE] early stop at acc {acc:.3f}")
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
        samples = self._build_samples(edit_ds)
        self._train(samples)
        self.apply_to_dataset(edit_ds)
        return self.model
