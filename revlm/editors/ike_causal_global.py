"""IKE_CAUSAL with GLOBAL negatives (all keys from all edits).

Warning: May have memory issues with 7000+ edits.
Use ike_causal.py (in-batch negatives) for large-scale experiments.
"""
import re
import random
import numpy as np
from scipy.stats import t as t_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import brackets_to_periods, parent_module, Augmenter


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        w = F.softmax(self.attn(x), dim=1)
        return (w * x).sum(dim=1)


class Proj(nn.Module):
    """Residual MLP projector."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.GELU(),
            nn.Linear(out_dim * 2, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.proj(x)
        return self.norm(x + self.mlp(x))


class IKE_CAUSAL_GLOBAL(nn.Module):
    """Causal next-key prediction with GLOBAL negatives.
    
    Learns p(key<img, s_n> | key<img, s_{n-1}>) for sequential retrieval.
    All keys from all edits are used as negatives (memory intensive).
    """

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        cfg = getattr(config, "editor", config)

        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))

        # Hyperparams
        self.clip_dim = int(getattr(cfg, "clip_dim", 256))
        self.num_epochs = int(getattr(cfg, "clip_epochs", 500))
        self.batch_size = int(getattr(cfg, "clip_batch_size", 8))
        self.lr = float(getattr(cfg, "clip_lr", 1e-3))
        self.temperature = float(getattr(cfg, "clip_temperature", 0.07))
        self.max_retrieve_steps = int(getattr(cfg, "max_retrieve_steps", 5))
        self.early_stop_acc = float(getattr(cfg, "early_stop_acc", 0.9))
        self.prefix = getattr(cfg, "cot_prefix", "New Fact: ")
        self.use_augment = bool(getattr(cfg, "use_augment", True))

        self.augmenter = Augmenter(self.wrapper) if self.use_augment else None

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

        # Projectors (lazy init)
        self.attn_pool = None
        self.query_proj = None
        self.key_proj = None

        # Index
        self.key_emb = None
        self.key_values = []

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        return (self.model if hasattr(self.model, "generate") else self.wrapper).generate(*a, **kw)

    def _encode_vlm(self, images, texts):
        """Get VLM layer activation for <image, text> pairs."""
        self.model.eval()
        self._last_act = None
        inputs = self.wrapper.encode(images, texts, tokenize=False)
        with torch.no_grad():
            self.model(**inputs)
        act = self._last_act
        if act is None:
            raise RuntimeError("Hook failed")
        act = act.unsqueeze(0) if act.dim() == 2 else act
        act = act.to(self.device, torch.float32)
        if self.attn_pool is None:
            self.attn_pool = AttentionPool(act.shape[-1]).to(self.device)
        return self.attn_pool(act)

    def _ensure_proj(self, vlm_dim):
        if self.query_proj is None:
            self.query_proj = Proj(vlm_dim, self.clip_dim).to(self.device)
        if self.key_proj is None:
            self.key_proj = Proj(vlm_dim, self.clip_dim).to(self.device)

    def _build_chains(self, dataset):
        """Build chains: each is (image, question, [s1, s2, ...])."""
        chains = []
        for ex in getattr(dataset, "data", []):
            rat = ex.get("cot") or ex.get("rationale") or ""
            q, img = ex.get("question", ""), ex.get("image")
            if not rat or img is None:
                continue
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", rat.strip()) if s.strip()]
            if sents:
                chains.append({"image": img, "question": q, "sentences": sents})
        print(f"[IKE_CAUSAL_GLOBAL] built {len(chains)} chains", flush=True)
        return chains

    def _train(self, chains):
        """Train next-key prediction with GLOBAL negatives."""
        if not chains:
            print("[IKE_CAUSAL_GLOBAL] no chains to train", flush=True)
            return

        # Build key index: (chain_idx, sent_idx) -> global key index
        key_index = [(ci, si) for ci, chain in enumerate(chains) for si in range(len(chain["sentences"]))]
        key_to_idx = {k: i for i, k in enumerate(key_index)}

        # Build transitions: query_state -> target_key
        transitions = [(ci, si, key_to_idx[(ci, si)]) for ci, si in key_index]
        print(f"[IKE_CAUSAL_GLOBAL] {len(key_index)} keys, {len(transitions)} transitions", flush=True)

        if len(transitions) < 2:
            print("[IKE_CAUSAL_GLOBAL] too few transitions", flush=True)
            return

        # Pre-encode all keys ONCE (no augmentation for stable targets)
        print("[IKE_CAUSAL_GLOBAL] encoding keys...", flush=True)
        key_imgs = [chains[ci]["image"] for ci, si in key_index]
        key_texts = [chains[ci]["sentences"][si] for ci, si in key_index]
        
        all_key_emb = []
        batch_sz = 4
        for i in range(0, len(key_imgs), batch_sz):
            emb = self._encode_vlm(key_imgs[i:i+batch_sz], key_texts[i:i+batch_sz])
            all_key_emb.append(emb.detach())
        all_key_emb = torch.cat(all_key_emb, dim=0)  # [N_keys, vlm_dim]
        
        self._ensure_proj(all_key_emb.shape[-1])

        params = list(self.attn_pool.parameters()) + list(self.query_proj.parameters()) + list(self.key_proj.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20, min_lr=1e-6)

        n_trans = len(transitions)
        for epoch in range(self.num_epochs):
            perm = torch.randperm(n_trans)
            loss_sum, cnt = 0.0, 0

            for start in range(0, n_trans, self.batch_size):
                batch_trans = [transitions[i] for i in perm[start:start + self.batch_size].tolist()]
                if len(batch_trans) < 2:
                    continue

                # Encode query states (with augmentation)
                q_imgs, q_texts, targets = [], [], []
                for ci, state_idx, target_key_idx in batch_trans:
                    chain = chains[ci]
                    img = self.augmenter.image(chain["image"]) if self.augmenter else chain["image"]
                    if state_idx == 0:
                        q_text = ""  # key<image, "">
                    else:
                        prev_sent = chain["sentences"][state_idx - 1]
                        q_text = self.augmenter.rationale(prev_sent) if self.augmenter and random.random() < 0.5 else prev_sent
                    q_imgs.append(img)
                    q_texts.append(q_text)
                    targets.append(target_key_idx)

                # Forward: query against ALL keys (global negatives)
                q_emb = F.normalize(self.query_proj(self._encode_vlm(q_imgs, q_texts)), dim=-1)
                k_emb = F.normalize(self.key_proj(all_key_emb), dim=-1)
                logits = (q_emb @ k_emb.t()) / self.temperature
                loss = F.cross_entropy(logits, torch.tensor(targets, device=self.device))

                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += loss.item()
                cnt += 1

            avg_loss = loss_sum / max(1, cnt)
            sched.step(avg_loss)

            if (epoch + 1) % 20 == 0 or epoch == 0:
                acc = self._eval_acc(chains, transitions, all_key_emb)
                lr = opt.param_groups[0]['lr']
                print(f"[IKE_CAUSAL_GLOBAL] ep {epoch+1}/{self.num_epochs} loss:{avg_loss:.4f} acc:{acc:.3f} lr:{lr:.1e}", flush=True)
                if acc >= self.early_stop_acc:
                    print(f"[IKE_CAUSAL_GLOBAL] early stop at acc {acc:.3f}", flush=True)
                    break

        # Build final index
        self._build_index(chains, key_index, all_key_emb)

    @torch.no_grad()
    def _eval_acc(self, chains, transitions, all_key_emb):
        """Compute next-key prediction accuracy on global index."""
        if self.key_proj is None:
            return 0.0
        k_emb = F.normalize(self.key_proj(all_key_emb), dim=-1)
        hits = 0
        for ci, state_idx, target_key_idx in transitions:
            chain = chains[ci]
            q_text = "" if state_idx == 0 else chain["sentences"][state_idx - 1]
            q_emb = F.normalize(self.query_proj(self._encode_vlm([chain["image"]], [q_text])), dim=-1)
            sims = (q_emb @ k_emb.t()).squeeze(0)
            if sims.argmax().item() == target_key_idx:
                hits += 1
        return hits / max(1, len(transitions))

    @torch.no_grad()
    def _build_index(self, chains, key_index, all_key_emb):
        """Build retrieval index."""
        if self.key_proj is None:
            return
        self.key_emb = F.normalize(self.key_proj(all_key_emb), dim=-1)
        self.key_values = [chains[ci]["sentences"][si] for ci, si in key_index]
        print(f"[IKE_CAUSAL_GLOBAL] index: {len(self.key_values)} keys", flush=True)

    @staticmethod
    def _is_outlier(probs, alpha=0.05):
        """Grubbs' test: is max prob an outlier?"""
        p = np.asarray(probs, dtype=float)
        n = len(p)
        if n < 3:
            return p.max() > 0.3
        mean, std = p.mean(), p.std(ddof=1)
        if std < 1e-12:
            return False
        G = (p.max() - mean) / std
        t_p = alpha / (2 * n)
        tcrit = t_dist.ppf(1 - t_p, df=n - 2)
        Gcrit = ((n - 1) / np.sqrt(n)) * np.sqrt(tcrit**2 / (n - 2 + tcrit**2))
        return G > Gcrit

    @torch.no_grad()
    def _retrieve_chain(self, image, start_text=""):
        """Autoregressive retrieval starting from key<img, start_text>."""
        if self.key_emb is None or len(self.key_values) == 0:
            return []

        retrieved = []
        seen = set()
        q_text = start_text

        for step in range(self.max_retrieve_steps):
            q_emb = F.normalize(self.query_proj(self._encode_vlm([image], [q_text])), dim=-1)
            sims = (q_emb @ self.key_emb.t()).squeeze(0)
            probs = F.softmax(sims / self.temperature, dim=-1).cpu().numpy()

            if not self._is_outlier(probs):
                break

            best_idx = int(probs.argmax())
            next_sent = self.key_values[best_idx]

            if next_sent in seen:
                break
            seen.add(next_sent)
            retrieved.append(next_sent)
            q_text = next_sent

        return retrieved

    def apply_to_dataset(self, dataset):
        """Apply retrieved facts to dataset prompts (two routes: image-only + image-question)."""
        applied = 0
        for ex in getattr(dataset, "data", []):
            prompt, q, img = ex.get("prompt", ""), ex.get("question", ""), ex.get("image")
            if not prompt or img is None:
                continue
            facts1 = self._retrieve_chain(img, "")
            facts2 = self._retrieve_chain(img, q) if q else []
            seen = set(facts1)
            facts = facts1 + [f for f in facts2 if f not in seen]
            if facts:
                ex["prompt"] = f"{self.prefix}{' '.join(facts)} {prompt}"
                applied += 1
        print(f"[IKE_CAUSAL_GLOBAL] applied facts to {applied} examples", flush=True)

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        print(f"[IKE_CAUSAL_GLOBAL] edit called, edit_ds has {len(getattr(edit_ds, 'data', []))} examples", flush=True)
        if edit_ds is None:
            return self.model
        chains = self._build_chains(edit_ds)
        self._train(chains)
        self.apply_to_dataset(edit_ds)
        return self.model

