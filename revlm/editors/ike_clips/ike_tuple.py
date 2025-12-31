import re
import random
import numpy as np
from PIL import Image as PILImage
from scipy.stats import t as t_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from .utils import brackets_to_periods, parent_module, Augmenter


class AttentionPool(nn.Module):
    """Attention pooling over sequence dimension."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):  # x: [B, seq, D]
        w = F.softmax(self.attn(x), dim=1)
        return (w * x).sum(dim=1)  # [B, D]


class ResidualProj(nn.Module):
    """Residual MLP projector for better optimization."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.GELU(), 
            # nn.Linear(out_dim * 2, out_dim * 2), nn.GELU(), 
            nn.Linear(out_dim * 2, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.proj(x)
        return self.norm(x + self.mlp(x))


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
        self.lr = float(getattr(cfg, "clip_lr", 1e-3))
        self.fixed_temp = getattr(cfg, "clip_temperature", None)  # None = learned
        self.log_temp = nn.Parameter(torch.tensor(0.0)) if self.fixed_temp is None else None
        self.early_stop_acc = float(getattr(cfg, "early_stop_acc", 0.9))
        self.early_stop_acc_last = float(getattr(cfg, "early_stop_acc_last", 0.6)) # 2/3 correct on the last edit
        self.prefix = getattr(cfg, "cot_prefix", "New Fact: ")
        self.use_augment = bool(getattr(cfg, "use_augment", True))
        self.use_counterfacts = bool(getattr(cfg, "use_counterfacts", False))
        self.num_counterfacts = int(getattr(cfg, "num_counterfacts", 3))  # per sentence

        # Augmenter (online, per-batch)
        dataset_name = getattr(getattr(config, "experiment", None), "dataset_name", None)
        self.augmenter = Augmenter(self.wrapper, dataset_name=dataset_name) if self.use_augment else None

        # Sentence model
        self.sentence_model = SentenceTransformer(
            getattr(cfg, "sentence_model_name", "sentence-transformers/paraphrase-mpnet-base-v2")
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
        self.attn_pool = None
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
        act = (act.unsqueeze(0) if act.dim() == 2 else act).to(self.device, dtype=torch.float32).clone()
        # Lazy init attention pool
        if self.attn_pool is None:
            self.attn_pool = AttentionPool(act.shape[-1]).to(self.device)
        return self.attn_pool(act)  # [B, D]

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
            self.image_proj = ResidualProj(img_dim, self.clip_dim).to(self.device)
        if self.text_proj is None:
            self.text_proj = ResidualProj(self.txt_dim, self.clip_dim).to(self.device)

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
            # Generate multiple counterfacts per sentence
            counterfacts = []
            if self.use_counterfacts:
                for s in sentences:
                    seen = {s.lower()}
                    for _ in range(self.num_counterfacts):
                        cf = self._gen_counterfact(s)
                        if cf and cf.lower() not in seen:
                            counterfacts.append(cf)
                            seen.add(cf.lower())
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
        """Train with f1 + fr: f1=<img,q>->all sentences, fr=<img,sent>->that sent only."""
        if not samples:
            return
        opt, sched = None, None
        n = len(samples)
        best_loss, no_improve, total_ep = float('inf'), 0, 0
        while total_ep < self.num_epochs:
            perm = torch.randperm(n)
            loss_sum, cnt = 0.0, 0
            for start in range(0, n, self.batch_size):
                batch = [samples[i] for i in perm[start:start + self.batch_size].tolist()]
                B = len(batch)
                has_counterfacts = any(b.get("counterfacts") for b in batch)
                if B < 2 and not has_counterfacts:
                    continue

                # Build sentence pool with dual ownership: edit_id and sent_id
                all_texts, edit_ids, sent_ids = [], [], []
                sid = 0
                for i, b in enumerate(batch):
                    for s in b["sentences"]:
                        all_texts.append(s)
                        edit_ids.append(i)
                        sent_ids.append(sid)
                        sid += 1
                    for cf in b.get("counterfacts", []):
                        all_texts.append(cf)
                        edit_ids.append(-1)  # counterfacts always negative for f1
                        sent_ids.append(-1)  # counterfacts always negative for fr

                if len(all_texts) < 2:
                    continue

                # f1 queries: <img, question> → all sentences from same edit
                f1_imgs = [b["image"] for b in batch]
                f1_texts = [b["question"] for b in batch]
                if self.augmenter:
                    f1_imgs = [self.augmenter.image(img) for img in f1_imgs]
                    f1_texts = [self.augmenter.question(q) if random.random() < 0.5 else q for q in f1_texts]

                # f2 queries: <img, ""> → all sentences from same edit
                f2_imgs = [self.augmenter.image(b["image"]) if self.augmenter else b["image"] for b in batch]

                # fr queries: <img, sentence> → that sentence only
                fr_imgs, fr_texts, fr_sent_ids = [], [], []
                sid = 0
                for b in batch:
                    img = b["image"]
                    for s in b["sentences"]:
                        aug_img = self.augmenter.image(img) if self.augmenter else img
                        aug_s = self.augmenter.rationale(s) if self.augmenter and random.random() < 0.5 else s
                        fr_imgs.append(aug_img)
                        fr_texts.append(aug_s)
                        fr_sent_ids.append(sid)
                        sid += 1

                # Encode queries
                f1_emb = self._encode_vlm(f1_imgs, f1_texts)
                f2_emb = self._encode_vlm(f2_imgs, [""] * B)
                self._ensure_heads(f1_emb.shape[-1])
                q_embs = [
                    F.normalize(self.image_proj(f1_emb), dim=-1),
                    F.normalize(self.image_proj(f2_emb), dim=-1),
                ]
                if fr_imgs:
                    fr_emb = self._encode_vlm(fr_imgs, fr_texts)
                    q_embs.append(F.normalize(self.image_proj(fr_emb), dim=-1))
                q_emb = torch.cat(q_embs, dim=0)  # [2*B + num_fr, D]

                # Encode targets
                t_emb = F.normalize(self.text_proj(self._encode_texts(all_texts)), dim=-1)
                edit_ids_t = torch.tensor(edit_ids, device=self.device)
                sent_ids_t = torch.tensor(sent_ids, device=self.device)

                temp = self.fixed_temp if self.fixed_temp else self.log_temp.sigmoid().clamp(min=0.07)
                logits = (q_emb @ t_emb.t()) / temp

                # Loss: f1/f2 use edit-level positives, fr uses sentence-level positives
                loss = torch.tensor(0.0, device=self.device)
                valid = 0
                # f1 queries (0 to B) and f2 queries (B to 2B) - both use edit-level positives
                for i in range(2 * B):
                    edit_idx = i % B
                    pos_mask = (edit_ids_t == edit_idx)
                    if not pos_mask.any():
                        continue
                    loss_i = -torch.logsumexp(logits[i, pos_mask], dim=0) + torch.logsumexp(logits[i], dim=0)
                    loss = loss + loss_i
                    valid += 1
                # fr queries (after 2*B)
                for j, owner_sid in enumerate(fr_sent_ids):
                    qi = 2 * B + j
                    pos_mask = (sent_ids_t == owner_sid)
                    if not pos_mask.any():
                        continue
                    loss_i = -torch.logsumexp(logits[qi, pos_mask], dim=0) + torch.logsumexp(logits[qi], dim=0)
                    loss = loss + loss_i
                    valid += 1

                if valid == 0:
                    continue
                loss = loss / valid

                if opt is None:
                    params = list(self.attn_pool.parameters()) + list(self.image_proj.parameters()) + list(self.text_proj.parameters())
                    if self.log_temp is not None:
                        params.append(self.log_temp)
                    opt = torch.optim.Adam(params, lr=self.lr)
                    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt, mode='min', factor=0.9, patience=10, cooldown=5, min_lr=1e-6
                    )
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += loss.item()
                cnt += 1

            avg_loss = loss_sum / max(1, cnt)
            if sched and cnt > 0:
                sched.step(avg_loss)
            # Warm restart if stuck
            if avg_loss < best_loss - 0.001:
                best_loss, no_improve = avg_loss, 0
            else:
                no_improve += 1
            if no_improve >= 100 and opt:
                opt, sched = None, None  # reset on next batch
                no_improve, best_loss = 0, float('inf')
                print(f"[IKE_TUPLE] warm restart at epoch {total_ep+1}")
            self._build_index(samples)
            acc = self._retrieval_acc(samples)
            acc_last = self._retrieval_acc(samples, last_only=True)
            k_str = "auto" if self.k < 0 else str(self.k)
            lr_str = f"{opt.param_groups[0]['lr']:.2e}" if opt else f"{self.lr:.2e}"
            total_ep += 1
            if total_ep % 10 == 0 or total_ep == 1:
                print(f"[IKE_TUPLE] epoch {total_ep}/{self.num_epochs} loss: {avg_loss:.4f} lr: {lr_str} acc@{k_str}: {acc:.3f} acc_last: {acc_last:.3f}")
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
