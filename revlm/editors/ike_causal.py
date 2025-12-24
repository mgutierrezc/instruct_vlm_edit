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


class IKE_CAUSAL(nn.Module):
    """Causal next-key prediction in VLM embedding space.
    
    Learns p(key<img, s_n> | key<img, s_{n-1}>) for sequential retrieval.
    Both queries and keys are VLM layer activations.
    """

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        cfg = getattr(config, "editor", config)

        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))

        # Hyperparams
        self.clip_dim = int(getattr(cfg, "clip_dim", 512))  # Larger for more capacity
        self.num_epochs = int(getattr(cfg, "clip_epochs", 1000))
        self.batch_size = int(getattr(cfg, "clip_batch_size", 10))
        self.lr = float(getattr(cfg, "clip_lr", 1e-3))  # Lower to prevent fast LR decay
        self.temperature = float(getattr(cfg, "clip_temperature", 1.0))  # Sharper distribution
        self.max_retrieve_steps = int(getattr(cfg, "max_retrieve_steps", 5)) # BFS depth
        self.kk = int(getattr(cfg, "kk", 3))  # Max outliers per step in BFS retrieval
        self.top_n = int(getattr(cfg, "top_n", 50))  # Choose from top n keys for auto_k
        self.early_stop_acc = float(getattr(cfg, "early_stop_acc", 0.8))
        self.prefix = getattr(cfg, "cot_prefix", "New Fact: ")
        self.train_last_only = bool(getattr(cfg, "train_last_only", False))  # Fast but may forget
        self.retrain_every = int(getattr(cfg, "retrain_every", -1))  # Full retrain every N batches (-1 = always all)
        self.early_stop_patience = int(getattr(cfg, "early_stop_patience", 100))  # Early stop if no loss decay for N epochs
        self.distance = getattr(cfg, "distance", "l2")  # "l2" (default) or "cosine"

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

        # Index: stores (key_embedding, sentence_value)
        self.key_emb = None      # [N, clip_dim] - projected VLM embeddings
        self.key_values = []     # [N] - sentence strings

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
        act = act.to(self.device, torch.float32)
        
        # For MLP layers (gate_proj), input is always (B, seq_len, hidden_dim)
        if act.dim() != 3:
            raise RuntimeError(f"Expected 3D activation (B, seq, hidden), got shape {act.shape}")
        
        # Attention pool over sequence dimension: (B, seq, hidden) -> (B, hidden)
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
        print(f"[IKE_CAUSAL] built {len(chains)} chains", flush=True)
        return chains

    def _train(self, chains):
        """Train next-key prediction with IN-BATCH NEGATIVES + AUGMENTATION."""
        if not chains:
            print("[IKE_CAUSAL] no chains to train", flush=True)
            return

        # Option: train only on last chain (faster, but may forget)
        train_chains = [chains[-1]] if self.train_last_only else chains
        n_chains = len(train_chains)
        total_keys = sum(len(c["sentences"]) for c in train_chains)
        mode = "last-only" if self.train_last_only else "all"
        print(f"[IKE_CAUSAL] {n_chains} chains, {total_keys} keys ({mode})", flush=True)

        if total_keys < 2:
            print("[IKE_CAUSAL] need >= 2 keys", flush=True)
            return

        # Lazy init projectors
        probe_emb = self._encode_vlm([train_chains[0]["image"]], [train_chains[0]["sentences"][0]])
        self._ensure_proj(probe_emb.shape[-1])

        params = list(self.attn_pool.parameters()) + list(self.query_proj.parameters()) + list(self.key_proj.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)  # AdamW with weight decay
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=50, min_lr=1e-6)  # More patient

        best_loss, no_improve = float('inf'), 0
        for epoch in range(self.num_epochs):
            perm = torch.randperm(n_chains)
            loss_sum, cnt = 0.0, 0

            for start in range(0, n_chains, self.batch_size):
                batch_chains = [train_chains[i] for i in perm[start:start + self.batch_size].tolist()]

                # Build in-batch keys (NO augmentation for stable targets)
                k_imgs, k_texts = [], []
                for chain in batch_chains:
                    for sent in chain["sentences"]:
                        k_imgs.append(chain["image"])
                        k_texts.append(sent)

                if len(k_imgs) < 2:
                    continue

                # Build queries (WITH augmentation)
                q_imgs, q_texts, targets = [], [], []
                key_offset = 0
                for chain in batch_chains:
                    n_sent = len(chain["sentences"])
                    for si in range(n_sent):
                        img = self.augmenter.image(chain["image"]) if self.augmenter else chain["image"]
                        if si == 0:
                            # Route 1: "" → s1
                            q_imgs.append(img)
                            q_texts.append("")
                            targets.append(key_offset)
                            # Route 2: question → s1 (extra entry)
                            q = chain["question"]
                            q_text = self.augmenter.question(q) if self.augmenter and random.random() < 0.5 else q
                            q_imgs.append(self.augmenter.image(chain["image"]) if self.augmenter else chain["image"])
                            q_texts.append(q_text)
                            targets.append(key_offset)
                        else:
                            prev = chain["sentences"][si - 1]
                            q_text = self.augmenter.rationale(prev) if self.augmenter and random.random() < 0.5 else prev
                            q_imgs.append(img)
                            q_texts.append(q_text)
                            targets.append(key_offset + si)
                    key_offset += n_sent

                # Encode and project
                k_emb = F.normalize(self.key_proj(self._encode_vlm(k_imgs, k_texts)), dim=-1)
                q_emb = F.normalize(self.query_proj(self._encode_vlm(q_imgs, q_texts)), dim=-1)

                logits = (q_emb @ k_emb.t()) / self.temperature
                loss = F.cross_entropy(logits, torch.tensor(targets, device=self.device))

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  # Gradient clipping
                opt.step()
                loss_sum += loss.item()
                cnt += 1

            avg_loss = loss_sum / max(1, cnt)
            sched.step(avg_loss)

            # Track loss plateau
            if avg_loss < best_loss - 1e-4:
                best_loss, no_improve = avg_loss, 0
            else:
                no_improve += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                self._build_index_full(train_chains)  # Eval on train_chains
                acc = self._eval_acc(train_chains)
                lr = opt.param_groups[0]['lr']
                print(f"[IKE_CAUSAL] ep {epoch+1}/{self.num_epochs} loss:{avg_loss:.4f} acc:{acc:.3f} lr:{lr:.1e}", flush=True)
                if acc >= self.early_stop_acc:
                    print(f"[IKE_CAUSAL] early stop at acc {acc:.3f}", flush=True)
                    break
            
            # Early stop if no loss improvement 
            if no_improve >= self.early_stop_patience:
                print(f"[IKE_CAUSAL] early stop: no loss decay for {self.early_stop_patience} epochs", flush=True)
                break

        # Final index from ALL chains (so all prior edits are retrievable)
        self._build_index_full(chains)

    @torch.no_grad()
    def _build_index_full(self, chains):
        """Build retrieval index from ALL chains."""
        if self.key_proj is None or not chains:
            return
        all_emb, all_texts = [], []
        batch_sz = 4
        for chain in chains:
            for sent in chain["sentences"]:
                all_texts.append(sent)
        all_imgs = [chain["image"] for chain in chains for _ in chain["sentences"]]
        for i in range(0, len(all_imgs), batch_sz):
            emb = self._encode_vlm(all_imgs[i:i+batch_sz], all_texts[i:i+batch_sz])
            all_emb.append(self.key_proj(emb))
        key_emb = torch.cat(all_emb, dim=0)
        # Only normalize for cosine similarity
        self.key_emb = F.normalize(key_emb, dim=-1) if self.distance == "cosine" else key_emb
        self.key_values = all_texts

    @torch.no_grad()
    def _eval_acc(self, chains):
        """Evaluate next-key prediction accuracy on full index."""
        if self.key_emb is None or not chains:
            return 0.0
        hits, total = 0, 0
        key_offset = 0
        for chain in chains:
            n_sent = len(chain["sentences"])
            for si in range(n_sent):
                q_text = "" if si == 0 else chain["sentences"][si - 1]
                q_emb = self.query_proj(self._encode_vlm([chain["image"]], [q_text]))
                if self.distance == "cosine":
                    q_emb = F.normalize(q_emb, dim=-1)
                    scores = (q_emb @ self.key_emb.t()).squeeze(0)
                else:
                    # L2: lower distance = better, so use negative for argmax
                    dists = torch.norm(self.key_emb - q_emb, dim=-1)
                    scores = -dists
                if scores.argmax().item() == key_offset + si:
                    hits += 1
                total += 1
            key_offset += n_sent
        return hits / max(1, total)

    @staticmethod
    def _auto_k(sims, top_n=50, alpha=0.05):
        """Use Grubbs' test on similarity gaps to find natural cutoff.
        
        Returns: number of outliers (0 if no clear cutoff)
        """
        sims = np.asarray(sims, dtype=float)
        if sims.size < 4:
            return 0
        vals = np.sort(sims)[::-1][:top_n]  # Top N, descending
        spread = vals[0] - vals[-1]
        if spread <= 0:
            return 0
        # Normalized gaps between consecutive values
        d = (vals[:-1] - vals[1:]) / spread
        n = d.size
        if n < 3:
            return 0
        mean, std = d.mean(), d.std(ddof=1)
        if std <= 1e-12:
            return 0
        # Find largest gap
        i = int(np.argmax(d))
        G = abs(d[i] - mean) / std
        # Grubbs critical value
        p = alpha / (2 * n)
        tcrit = t_dist.ppf(1 - p, df=n - 2)
        Gcrit = ((n - 1) / np.sqrt(n)) * np.sqrt(tcrit**2 / (n - 2 + tcrit**2))
        return (i + 1) if G > Gcrit else 0

    def _compute_scores(self, q_emb):
        """Compute similarity scores (higher = better match)."""
        if self.distance == "cosine":
            q_emb = F.normalize(q_emb, dim=-1)
            return (q_emb @ self.key_emb.t()).squeeze(0).cpu().numpy()
        else:
            # L2: lower distance = better, convert to scores (negative distance)
            dists = torch.norm(self.key_emb - q_emb, dim=-1)
            return -dists.cpu().numpy()

    @torch.no_grad()
    def _retrieve_chain(self, image, start_text=""):
        """BFS-style retrieval: at each step, expand all frontier keys by their outliers.
        
        Args:
            image: query image
            start_text: initial text (empty for image-only, or question)
        
        Returns:
            List of retrieved sentences (unique, in BFS order)
        
        Supports L2 distance (default) or cosine similarity.
        """
        if self.key_emb is None or len(self.key_values) == 0:
            return []

        collected = []  # Ordered list of sentences
        seen_idx = set()
        
        # Step 0: Initial query
        q_emb = self.query_proj(self._encode_vlm([image], [start_text]))
        scores = self._compute_scores(q_emb)
        
        k = self._auto_k(scores, top_n=self.top_n)
        if k == 0:
            return []
        k = min(k, self.kk)  # Cap at kk
        
        # Get top-k indices (highest scores)
        top_indices = np.argsort(scores)[::-1][:k]
        frontier = []
        for idx in top_indices:
            if idx not in seen_idx:
                seen_idx.add(idx)
                collected.append(self.key_values[idx])
                frontier.append(idx)
        
        # BFS steps
        for step in range(1, self.max_retrieve_steps):
            if not frontier:
                break
            
            new_frontier = []
            for f_idx in frontier:
                # Query from this frontier key
                q_text = self.key_values[f_idx]
                q_emb = self.query_proj(self._encode_vlm([image], [q_text]))
                scores = self._compute_scores(q_emb)
                
                k = self._auto_k(scores, top_n=self.top_n)
                if k == 0:
                    continue
                k = min(k, self.kk)
                
                top_indices = np.argsort(scores)[::-1][:k]
                for idx in top_indices:
                    if idx not in seen_idx:
                        seen_idx.add(idx)
                        collected.append(self.key_values[idx])
                        new_frontier.append(idx)
            
            frontier = new_frontier
        
        return collected

    def apply_to_dataset(self, dataset):
        """Apply retrieved facts to dataset prompts (two routes: image-only + image-question)."""
        applied = 0
        for ex in getattr(dataset, "data", []):
            prompt, q, img = ex.get("prompt", ""), ex.get("question", ""), ex.get("image")
            if not prompt or img is None:
                continue
            # Route 1: <image, ""> chain
            facts1 = self._retrieve_chain(img, "")
            # Route 2: <image, question> chain
            facts2 = self._retrieve_chain(img, q) if q else []
            # Merge unique, preserving order from route 1 first
            seen = set(facts1)
            facts = facts1 + [f for f in facts2 if f not in seen]
            if facts:
                ex["prompt"] = f"{self.prefix}{' '.join(facts)} {prompt}"
                applied += 1
        print(f"[IKE_CAUSAL] applied facts to {applied} examples", flush=True)

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        print(f"[IKE_CAUSAL] edit called, edit_ds has {len(getattr(edit_ds, 'data', []))} examples", flush=True)
        if edit_ds is None:
            return self.model
        chains = self._build_chains(edit_ds)
        self._train(chains)
        self.apply_to_dataset(edit_ds)
        return self.model

