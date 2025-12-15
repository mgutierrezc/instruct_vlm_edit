import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from .utils import brackets_to_periods, parent_module


class IKE_TUPLE(nn.Module):
    """Minimal tuple retriever: (image, question) → sentence facts."""

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        cfg = getattr(config, "editor", config)

        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))

        # Hyperparams
        self.k = int(getattr(cfg, "k", 3))
        self.clip_dim = int(getattr(cfg, "clip_dim", 512))
        self.num_epochs = int(getattr(cfg, "clip_epochs", 100))
        self.batch_size = int(getattr(cfg, "clip_batch_size", 8))
        self.lr = float(getattr(cfg, "clip_lr", 1e-3))
        self.temperature = float(getattr(cfg, "clip_temperature", 1.0))
        self.max_pairs = int(getattr(cfg, "max_pairs", 512))
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
        return self.sentence_model.encode(texts, convert_to_tensor=True, show_progress_bar=False).to(self.device, dtype=torch.float32)

    def _encode_vlm(self, images, questions):
        self.model.eval()
        self._last_act = None
        inputs = self.wrapper.encode(images, questions, tokenize=False)
        with torch.no_grad():
            self.model(**inputs)
        act = self._last_act
        if act is None:
            raise RuntimeError("Hook failed")
        return (act.unsqueeze(0) if act.dim() == 2 else act).mean(dim=1).to(self.device, dtype=torch.float32)

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

    def _build_pairs(self, dataset):
        data = getattr(dataset, "data", [])
        pairs = []
        for ex in data:
            rat = ex.get("cot") or ex.get("rationale") or ""
            q, img = ex.get("question", ""), ex.get("image")
            if not rat or img is None or not q:
                continue
            for s in [p.strip() for p in re.split(r"(?<=[.!?])\s+", rat.strip()) if p.strip()]:
                neg = self._gen_neg(img, s)
                if neg:
                    pairs.append({"image": img, "question": q, "pos": s, "neg": neg})
                if len(pairs) >= self.max_pairs:
                    return pairs
        return pairs

    def _gen_neg(self, image, sentence):
        if not self.wrapper or not sentence:
            return ""
        inst = f"Rewrite to state a different plausible fact:\n\nOriginal: {sentence}\n\nRewritten:"
        try:
            out = self.wrapper.generate([image], [inst], max_new_tokens=64, temperature=0.0)[0]
        except Exception:
            return ""
        t = str(out).strip().splitlines()[0].strip()
        t = re.sub(r"^(rewritten sentence|rewrite|answer)\s*:\s*", "", t, flags=re.IGNORECASE).strip()
        return t if t and t.lower() != sentence.lower() else ""

    def _train(self, pairs):
        if not pairs:
            return
        opt = None
        n = len(pairs)
        for ep in range(self.num_epochs):
            perm = torch.randperm(n)
            loss_sum, cnt = 0.0, 0
            for start in range(0, n, self.batch_size):
                batch = [pairs[i] for i in perm[start:start + self.batch_size].tolist()]
                img_f = self._encode_vlm([b["image"] for b in batch], [b["question"] for b in batch])
                pos_f = self._encode_texts([b["pos"] for b in batch])
                neg_f = self._encode_texts([b["neg"] for b in batch])
                self._ensure_heads(img_f.shape[-1])
                img_e = F.normalize(self.image_proj(img_f), dim=-1)
                pos_e = F.normalize(self.text_proj(pos_f), dim=-1)
                neg_e = F.normalize(self.text_proj(neg_f), dim=-1)
                logits = torch.stack([
                    (img_e * pos_e).sum(-1) / self.temperature,
                    (img_e * neg_e).sum(-1) / self.temperature
                ], dim=1)
                loss = F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long, device=self.device))
                if opt is None:
                    opt = torch.optim.Adam(list(self.image_proj.parameters()) + list(self.text_proj.parameters()), lr=self.lr)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += loss.item()
                cnt += 1
            print(f"[IKE_TUPLE_SIMPLE] epoch {ep+1}/{self.num_epochs} loss: {loss_sum/max(1,cnt):.4f}")
        self._build_index(pairs)

    @torch.no_grad()
    def _build_index(self, pairs):
        if not pairs or self.text_proj is None:
            return
        texts = list(dict.fromkeys(p["pos"] for p in pairs if p.get("pos")))
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
                ex["prompt"] = f"{self.prefix}{' '.join(facts)}\n\n{prompt}"

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        if edit_ds is None:
            return self.model
        pairs = self._build_pairs(train_ds or edit_ds)
        if pairs:
            self._train(pairs)
        if self.rationale_emb is not None:
            self.apply_to_dataset(edit_ds)
        return self.model
