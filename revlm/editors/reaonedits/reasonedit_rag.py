"""ReasonEdit E2E: End-to-end RAG with trainable retriever + layer finetuning."""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import brackets_to_periods, parent_module


class ReasonEdit(nn.Module):
    """
    End-to-end RAG editor with jointly trained retriever and generator.
    
    - Query/Doc encoder: VLM hidden states + trainable projection MLP
    - Retrieval: Soft attention over all documents (differentiable)
    - Generator: VLM with trainable layer_edit
    - Training: Joint optimization of projection + layer_edit
    
    For each edit (image, question, answer, rationale):
    - k1: train on (image, question, answer) with retrieved context
    - k2: train on (image, full_cot) for rationale generation
    """

    def __init__(self, config, model):
        super().__init__()
        cfg = getattr(config, "editor", config)

        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cuda"))

        # Hyperparams
        self.ft_epochs = int(getattr(cfg, "ft_epochs", 100))
        self.ft_lr = float(getattr(cfg, "edit_lr", 1e-4))
        self.proj_lr = float(getattr(cfg, "proj_lr", 1e-3))  # Separate LR for projection
        self.ft_batch_size = int(getattr(cfg, "ft_batch_size", 4))
        self.early_stop_patience = int(getattr(cfg, "early_stop_patience", 10))
        self.temperature = float(getattr(cfg, "temperature", 0.1))  # Softmax temperature
        self.top_k_soft = int(getattr(cfg, "top_k_soft", 5))  # Top-k for sparse attention
        self.prefix = getattr(cfg, "cot_prefix", "")
        self.reindex_interval = int(getattr(cfg, "reindex_interval", 10))  # Re-encode docs every N epochs

        # Freeze entire model
        for p in self.model.parameters():
            p.requires_grad = False

        # Disable KV cache for training
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        # Layer setup for hidden state extraction
        inner_params = getattr(getattr(config, "model", config), "inner_params", [])
        if not inner_params:
            raise ValueError("Requires config.model.inner_params")
        raw = inner_params[0]
        param_name = raw.rsplit(".", 1)[0] if raw.endswith((".weight", ".bias")) else raw
        self.edit_mod = parent_module(self.model, brackets_to_periods(param_name))
        self.layer_name = param_name.rsplit(".", 1)[-1]
        self.target_layer = getattr(self.edit_mod, self.layer_name)
        
        # Hook for VLM hidden states
        self._last_act = None
        self._hook = self.target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_last_act", i[0].detach() if isinstance(i[0], torch.Tensor) else None)
        )
        
        # Get hidden dim from layer
        if hasattr(self.target_layer, 'weight'):
            self.hidden_dim = self.target_layer.weight.shape[1]
        else:
            self.hidden_dim = 4096  # Default, will be set on first encode
        
        # Trainable projection MLP (shared for query and docs)
        self.proj_dim = int(getattr(cfg, "proj_dim", 256))
        self.proj = None  # Initialized on first encode when we know hidden_dim
        
        # layer_edit for generator
        self.layer_edit = None
        self.switcher = None

        # Document storage: list of {"text": str, "image": PIL, "emb": tensor}
        self._documents = []
        self._doc_embs = None  # Cached [N, proj_dim] tensor
        
        # Training samples
        self._k1_samples = []  # {image, question, answer, doc_indices}
        self._k2_samples = []  # {image, prompt (full COT)}
        self._added_uids = set()

    def _init_proj(self, hidden_dim):
        """Initialize projection MLP once we know hidden_dim."""
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.proj_dim),
        ).to(self.device)
        # Initialize with small weights for stable training
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def generate(self, *a, **kw):
        if hasattr(self.wrapper, "generate"):
            return self.wrapper.generate(*a, **kw)
        return self.model.generate(*a, **kw)

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    @torch.no_grad()
    def _encode_vlm(self, images, texts):
        """Get VLM hidden states for (image, text) pairs."""
        self.model.eval()
        self._last_act = None
        inputs = self.wrapper.encode(images, texts, tokenize=False)
        self.model(**inputs)
        act = self._last_act
        if act is None:
            raise RuntimeError("Hook failed to capture activations")
        act = (act.unsqueeze(0) if act.dim() == 2 else act).to(self.device, dtype=torch.float32)
        return act.mean(dim=1)  # [B, hidden_dim]

    def _encode_with_proj(self, images, texts, detach_vlm=True):
        """Encode with VLM + projection MLP."""
        if detach_vlm:
            with torch.no_grad():
                vlm_emb = self._encode_vlm(images, texts)
        else:
            vlm_emb = self._encode_vlm(images, texts)
        
        # Initialize proj if needed
        if self.proj is None:
            self._init_proj(vlm_emb.shape[-1])
        
        return self.proj(vlm_emb)  # [B, proj_dim]

    def _parse_rationale(self, rationale):
        """Split rationale into sentences."""
        import re
        if not rationale:
            return []
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", rationale.strip()) if s.strip()]

    def _add_documents(self, image, rationale):
        """Add rationale sentences as documents."""
        sentences = self._parse_rationale(rationale)
        doc_indices = []
        for sent in sentences:
            self._documents.append({
                "text": sent,
                "image": image,
                "emb": None,  # Will be computed in _reindex_documents
            })
            doc_indices.append(len(self._documents) - 1)
        return doc_indices

    @torch.no_grad()
    def _reindex_documents(self):
        """Re-encode all documents with current projection."""
        if not self._documents or self.proj is None:
            self._doc_embs = None
            return
        
        embs = []
        for doc in self._documents:
            emb = self._encode_with_proj([doc["image"]], [doc["text"]], detach_vlm=True)
            doc["emb"] = emb.cpu()
            embs.append(emb)
        self._doc_embs = torch.cat(embs, dim=0)  # [N, proj_dim]

    def _soft_retrieve(self, query_emb, return_weights=False):
        """Soft attention retrieval over documents.
        
        Args:
            query_emb: [B, proj_dim] query embeddings
            return_weights: if True, return attention weights
            
        Returns:
            context_texts: list of retrieved context strings (one per query)
            weights: [B, N] attention weights (if return_weights=True)
        """
        if self._doc_embs is None or len(self._documents) == 0:
            if return_weights:
                return [""] * query_emb.shape[0], None
            return [""] * query_emb.shape[0]
        
        doc_embs = self._doc_embs.to(query_emb.device)  # [N, proj_dim]
        
        # Compute similarities
        query_norm = F.normalize(query_emb, p=2, dim=-1)  # [B, proj_dim]
        doc_norm = F.normalize(doc_embs, p=2, dim=-1)  # [N, proj_dim]
        sims = torch.matmul(query_norm, doc_norm.T) / self.temperature  # [B, N]
        
        # Top-k sparse attention for efficiency
        if self.top_k_soft < len(self._documents):
            topk_vals, topk_idx = sims.topk(self.top_k_soft, dim=-1)  # [B, k]
            mask = torch.zeros_like(sims).scatter_(-1, topk_idx, 1.0)
            sims = sims.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(sims, dim=-1)  # [B, N]
        
        # Build context strings (weighted by attention)
        context_texts = []
        for b in range(query_emb.shape[0]):
            w = weights[b]  # [N]
            # Get top contributing docs
            top_indices = w.topk(min(self.top_k_soft, len(self._documents)))[1].tolist()
            # Filter by threshold
            selected = [(i, w[i].item()) for i in top_indices if w[i].item() > 0.01]
            if selected:
                texts = [self._documents[i]["text"] for i, _ in selected]
                context_texts.append(" ".join(texts))
            else:
                context_texts.append("")
        
        if return_weights:
            return context_texts, weights
        return context_texts

    def _add_training_samples(self, image, question, answer, rationale, doc_indices):
        """Add k1 and k2 training samples for a single edit."""
        # k1: (image, question, answer) - will use retrieved context during training
        self._k1_samples.append({
            "image": image,
            "question": question,
            "answer": answer,
            "doc_indices": doc_indices,  # For computing retrieval loss
        })

        # k2: (image, full_cot) for rationale generation
        sentences = self._parse_rationale(rationale)
        if sentences:
            full_cot = " ".join(sentences)
            self._k2_samples.append({
                "image": image,
                "prompt": full_cot,
            })

    def _prepare_k1_batch_with_retrieval(self, samples):
        """Prepare k1 batch with soft-retrieved context."""
        # Encode queries
        query_embs = []
        for s in samples:
            emb = self._encode_with_proj([s["image"]], [s["question"]], detach_vlm=True)
            query_embs.append(emb)
        query_emb = torch.cat(query_embs, dim=0)  # [B, proj_dim]
        
        # Soft retrieve
        contexts, weights = self._soft_retrieve(query_emb, return_weights=True)
        
        # Build prompts with context
        prompts = []
        for s, ctx in zip(samples, contexts):
            if ctx:
                prompt = f"{self.prefix}{ctx} {s['question']}"
            else:
                prompt = s["question"]
            prompts.append(prompt)
        
        # Prepare batch
        tokens = self.wrapper.prepare_training_batch({
            "images": [s["image"] for s in samples],
            "prompts": prompts,
            "golds": [{"label": s["answer"], "label_train": s["answer"]} for s in samples],
            "idxs": list(range(len(samples))),
        })
        
        return tokens, weights

    def _prepare_k2_batch(self, samples):
        """Prepare k2 training batch: full COT as answer."""
        tokens = self.wrapper.prepare_training_batch({
            "images": [s["image"] for s in samples],
            "prompts": ["" for s in samples],
            "golds": [{"label": s["prompt"], "label_train": s["prompt"]} for s in samples],
            "idxs": list(range(len(samples))),
        })
        return tokens

    def _compute_retrieval_loss(self, weights, samples):
        """Compute retrieval supervision loss.
        
        Encourages retrieving documents from the same edit.
        """
        if weights is None:
            return torch.tensor(0.0, device=self.device)
        
        loss = 0.0
        for b, s in enumerate(samples):
            target_indices = s.get("doc_indices", [])
            if not target_indices:
                continue
            # Target: high weight on relevant docs
            target = torch.zeros(weights.shape[1], device=weights.device)
            for idx in target_indices:
                if idx < len(target):
                    target[idx] = 1.0 / len(target_indices)
            # KL divergence
            loss = loss + F.kl_div(
                (weights[b] + 1e-8).log(),
                target,
                reduction='sum'
            )
        return loss / max(1, len(samples))

    def _train_e2e(self):
        """End-to-end training of projection + layer_edit."""
        all_k1 = self._k1_samples
        all_k2 = self._k2_samples
        
        if not all_k1 and not all_k2:
            return

        # Initialize layer_edit if needed
        if self.layer_edit is None:
            self.layer_edit = deepcopy(self.target_layer)
            for p in self.layer_edit.parameters():
                p.requires_grad = True
            self.switcher = _Switcher(self.target_layer, self.layer_edit)
            setattr(self.edit_mod, self.layer_name, self.switcher)

        # Initialize proj if needed (encode one sample to get hidden_dim)
        if self.proj is None and all_k1:
            _ = self._encode_with_proj([all_k1[0]["image"]], [all_k1[0]["question"]])

        # Initial document indexing
        self._reindex_documents()

        self.switcher.use_edit = True

        # Combine samples
        all_samples = [("k1", s) for s in all_k1] + [("k2", s) for s in all_k2]
        n = len(all_samples)
        
        # Separate optimizers for projection and layer_edit
        opt_proj = torch.optim.Adam(self.proj.parameters(), lr=self.proj_lr, eps=1e-4)
        opt_layer = torch.optim.Adam(self.layer_edit.parameters(), lr=self.ft_lr, eps=1e-4)
        
        scheduler_proj = torch.optim.lr_scheduler.CosineAnnealingLR(opt_proj, T_max=max(1, self.ft_epochs))
        scheduler_layer = torch.optim.lr_scheduler.CosineAnnealingLR(opt_layer, T_max=max(1, self.ft_epochs))
        
        best_loss, patience_cnt = float('inf'), 0

        for ep in range(self.ft_epochs):
            # Periodic re-indexing of documents
            if ep > 0 and ep % self.reindex_interval == 0:
                self._reindex_documents()
            
            perm = torch.randperm(n)
            loss_sum, cnt = 0.0, 0

            for start in range(0, n, self.ft_batch_size):
                batch_indices = perm[start:start + self.ft_batch_size].tolist()
                batch = [all_samples[i] for i in batch_indices]
                
                k1_batch = [s for t, s in batch if t == "k1"]
                k2_batch = [s for t, s in batch if t == "k2"]
                
                opt_proj.zero_grad(set_to_none=True)
                opt_layer.zero_grad(set_to_none=True)
                
                batch_loss = torch.tensor(0.0, device=self.device)
                
                # k1 loss with retrieval
                if k1_batch:
                    tokens, weights = self._prepare_k1_batch_with_retrieval(k1_batch)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        gen_loss = self.model(**tokens).loss
                    # Retrieval supervision loss
                    ret_loss = self._compute_retrieval_loss(weights, k1_batch)
                    batch_loss = batch_loss + gen_loss + 0.1 * ret_loss
                
                # k2 loss
                if k2_batch:
                    tokens = self._prepare_k2_batch(k2_batch)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        loss = self.model(**tokens).loss
                    batch_loss = batch_loss + loss
                
                if batch_loss.requires_grad:
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.proj.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.layer_edit.parameters(), 1.0)
                    opt_proj.step()
                    opt_layer.step()
                    loss_sum += batch_loss.item()
                    cnt += 1

            scheduler_proj.step()
            scheduler_layer.step()
            avg_loss = loss_sum / max(1, cnt)

            if avg_loss < best_loss:
                best_loss, patience_cnt = avg_loss, 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.early_stop_patience:
                    print(f"[ReasonEdit E2E] early stop at epoch {ep+1}, loss: {avg_loss:.4f}")
                    break

            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"[ReasonEdit E2E] epoch {ep+1}/{self.ft_epochs} loss: {avg_loss:.4f} (k1: {len(all_k1)}, k2: {len(all_k2)}, docs: {len(self._documents)})")

        # Final re-indexing
        self._reindex_documents()
        self.switcher.use_edit = False

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add new edits and train E2E."""
        if edit_ds is None:
            return self.model

        data = getattr(edit_ds, "data", [])
        added = 0
        
        for ex in data:
            uid = ex.get("uid") or (id(ex.get("image")), ex.get("question"))
            if uid in self._added_uids:
                continue
            
            image = ex.get("image")
            question = ex.get("question", "")
            answer = ex.get("gold", {}).get("label", "") or ex.get("gold", {}).get("label_train", "")
            rationale = ex.get("cot") or ex.get("rationale") or ""
            
            if image is None or not question:
                continue
            
            # Add documents (rationale sentences)
            doc_indices = self._add_documents(image, rationale)
            
            # Add training samples
            self._add_training_samples(image, question, answer, rationale, doc_indices)
            self._added_uids.add(uid)
            added += 1

        print(f"[ReasonEdit E2E] Added {added} edits, k1: {len(self._k1_samples)}, k2: {len(self._k2_samples)}, docs: {len(self._documents)}")

        # Train E2E
        print("[ReasonEdit E2E] Training...")
        self._train_e2e()
        
        return self.model

    def retrieve_and_apply(self, image, question, prompt):
        """Retrieve context and toggle layer_edit.
        
        Returns: (modified_prompt, is_edit)
        """
        if self.proj is None or not self._documents:
            return prompt, False
        
        query_emb = self._encode_with_proj([image], [question], detach_vlm=True)
        contexts = self._soft_retrieve(query_emb)
        ctx = contexts[0]
        
        is_edit = bool(ctx)
        if self.switcher:
            self.switcher.use_edit = is_edit
        
        if is_edit:
            prompt = f"{self.prefix}{ctx} {prompt}"
        
        return prompt, is_edit

    def apply_to_dataset(self, dataset):
        """Inference: prepend retrieved context and toggle switcher per sample."""
        if self.switcher is None:
            return
        
        for ex in getattr(dataset, "data", []):
            if "_original_prompt" not in ex:
                ex["_original_prompt"] = ex.get("prompt", "")
            
            img = ex.get("image")
            q = ex.get("question", "")
            prompt = ex["_original_prompt"]
            
            if img is None or not prompt:
                continue
            
            new_prompt, _ = self.retrieve_and_apply(img, q, prompt)
            ex["prompt"] = new_prompt

    def get_stats(self):
        """Return statistics."""
        return {
            "num_documents": len(self._documents),
            "num_k1_samples": len(self._k1_samples),
            "num_k2_samples": len(self._k2_samples),
            "num_edits": len(self._added_uids),
            "proj_initialized": self.proj is not None,
            "layer_edit_initialized": self.layer_edit is not None,
            "hidden_dim": self.hidden_dim,
            "proj_dim": self.proj_dim,
        }


class _Switcher(nn.Module):
    """Toggles between original layer and edited layer."""
    
    def __init__(self, layer_orig, layer_edit):
        super().__init__()
        self.layer_orig = layer_orig
        self.layer_edit = layer_edit
        self.use_edit = False

    def forward(self, *args, **kwargs):
        if self.use_edit:
            return self.layer_edit(*args, **kwargs)
        return self.layer_orig(*args, **kwargs)

