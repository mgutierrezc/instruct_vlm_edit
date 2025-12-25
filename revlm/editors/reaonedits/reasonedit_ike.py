"""ReasonEdit: IKE retrieval + k1/k2 finetuning."""

from copy import deepcopy
from itertools import combinations
import torch
import torch.nn as nn
from .ike_proto import IKE_PROTO
from .ike_cot import IKE_COT
from .ike_chain import IKE_CHAIN
from .utils import brackets_to_periods, parent_module


class ReasonEdit(nn.Module):
    """
    Retriever-finetuner editor combining retrieval with layer finetuning.
    
    For each edit (image, question, answer, rationale=[s1, s2, ...]):
    - k1: train on (image, question, answer) with answer-only unmasked
    - k2: train on (image, rationale) fully autoregressive
      - k2_use_subsets=True: all subsets (s1), (s2), (s1 s2), ... as separate samples
      - k2_use_subsets=False: single full COT (s1 s2 s3...) as one sample
    
    Inference: if retrieves facts → prepend facts to prompt AND use layer_edit.
    """

    def __init__(self, config, model):
        super().__init__()
        cfg = getattr(config, "editor", config)

        # for retrieval
        self.retrievername = "chain"
        if self.retrievername == "chain":
            self.retriever = IKE_CHAIN(config, model)
        elif self.retrievername == "proto":
            self.retriever = IKE_PROTO(config, model)
        elif self.retrievername == "cot":
            self.retriever = IKE_COT(config, model)
        else:
            raise ValueError(f"Invalid retriever name: {self.retrievername}")
        self.prefix = getattr(cfg, "cot_prefix", "")

        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cuda"))

        # Hyperparams
        self.ft_epochs = int(getattr(cfg, "ft_epochs", 1000))
        self.ft_lr = float(getattr(cfg, "edit_lr", 1e-4))
        self.ft_batch_size = int(getattr(cfg, "ft_batch_size", 4))
        self.early_stop_patience = int(getattr(cfg, "early_stop_patience", 20))
        self.max_subset_size = int(getattr(cfg, "max_subset_size", 3))
        self.k2_use_subsets = bool(getattr(cfg, "k2_use_subsets", False))  # False = single full COT

        # Freeze entire model
        for p in self.model.parameters():
            p.requires_grad = False

        # Disable KV cache for training
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        # Layer setup
        inner_params = getattr(getattr(config, "model", config), "inner_params", [])
        if not inner_params:
            raise ValueError("Requires config.model.inner_params")
        raw = inner_params[0]
        param_name = raw.rsplit(".", 1)[0] if raw.endswith((".weight", ".bias")) else raw
        self.edit_mod = parent_module(self.model, brackets_to_periods(param_name))
        self.layer_name = param_name.rsplit(".", 1)[-1]
        self.target_layer = getattr(self.edit_mod, self.layer_name)
        self.layer_edit = None
        self.switcher = None

        # Cumulative training samples
        self._k1_samples = []  # List of dicts: {image, prompt, answer}
        self._k2_samples = []  # List of dicts: {image, prompt} (fully autoregressive)
        self._added_uids = set()

    def generate(self, *a, **kw):
        return self.retriever.generate(*a, **kw)

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def _parse_rationale(self, rationale):
        """Split rationale into sentences."""
        import re
        if not rationale:
            return []
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", rationale.strip()) if s.strip()]

    def _add_training_samples(self, image, question, answer, rationale):
        """Add k1 and k2 training samples for a single edit."""
        sentences = self._parse_rationale(rationale)
        
        # k1: (image, question, answer) with answer-only unmasked
        self._k1_samples.append({
            "image": image,
            "prompt": question,
            "answer": answer,
        })

        # k2: rationale training samples
        if sentences:
            if self.k2_use_subsets:
                # All subsets → multiple training samples
                n = len(sentences)
                for size in range(1, min(n, self.max_subset_size) + 1):
                    for subset_indices in combinations(range(n), size):
                        subset_text = " ".join(sentences[i] for i in subset_indices)
                        self._k2_samples.append({
                            "image": image,
                            "prompt": subset_text,
                        })
            else:
                # Single full COT → one training sample
                full_cot = " ".join(sentences)
                self._k2_samples.append({
                    "image": image,
                    "prompt": full_cot,
                })

    def _prepare_k1_batch(self, samples):
        """Prepare k1 training batch: answer-only unmasked."""
        tokens = self.wrapper.prepare_training_batch({
            "images": [s["image"] for s in samples],
            "prompts": [s["prompt"] for s in samples],
            "golds": [{"label": s["answer"], "label_train": s["answer"]} for s in samples],
            "idxs": list(range(len(samples))),
        })
        # Labels already have answer-only unmasked by prepare_training_batch
        return tokens

    def _prepare_k2_batch(self, samples):
        """Prepare k2 training batch: rationale subset as answer, image tokens masked."""
        tokens = self.wrapper.prepare_training_batch({
            "images": [s["image"] for s in samples],
            "prompts": ["" for s in samples],  # Empty prompt so subset becomes "answer"
            "golds": [{"label": s["prompt"], "label_train": s["prompt"]} for s in samples],
            "idxs": list(range(len(samples))),
        })
        # Labels already have image tokens masked, only rationale subset is supervised
        return tokens

    def _train_on_all_samples(self):
        """Train layer_edit on ALL accumulated k1 + k2 samples."""
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

        self.switcher.use_edit = True

        # Combine samples with type tags
        all_samples = [("k1", s) for s in all_k1] + [("k2", s) for s in all_k2]
        n = len(all_samples)
        
        opt = torch.optim.Adam(self.layer_edit.parameters(), lr=self.ft_lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.ft_epochs))
        best_loss, patience_cnt = float('inf'), 0

        for ep in range(self.ft_epochs):
            perm = torch.randperm(n)
            loss_sum, cnt = 0.0, 0

            for start in range(0, n, self.ft_batch_size):
                batch_indices = perm[start:start + self.ft_batch_size].tolist()
                batch = [all_samples[i] for i in batch_indices]
                
                # Separate k1 and k2 samples in this batch
                k1_batch = [s for t, s in batch if t == "k1"]
                k2_batch = [s for t, s in batch if t == "k2"]
                
                opt.zero_grad(set_to_none=True)
                batch_loss = 0.0
                
                # k1 loss
                if k1_batch:
                    tokens = self._prepare_k1_batch(k1_batch)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        loss = self.model(**tokens).loss
                    batch_loss = batch_loss + loss
                
                # k2 loss
                if k2_batch:
                    tokens = self._prepare_k2_batch(k2_batch)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        loss = self.model(**tokens).loss
                    batch_loss = batch_loss + loss
                
                if isinstance(batch_loss, torch.Tensor):
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.layer_edit.parameters(), 1.0)
                    opt.step()
                    loss_sum += batch_loss.item()
                    cnt += 1

            scheduler.step()
            avg_loss = loss_sum / max(1, cnt)

            if avg_loss < best_loss:
                best_loss, patience_cnt = avg_loss, 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.early_stop_patience:
                    print(f"[ReasonEdit] early stop at epoch {ep+1}, loss: {avg_loss:.4f}")
                    break

            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"[ReasonEdit] epoch {ep+1}/{self.ft_epochs} loss: {avg_loss:.4f} (k1: {len(all_k1)}, k2: {len(all_k2)})")

        self.switcher.use_edit = False  # Default off; retrieval toggles

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add new edits: store in retriever + build training samples + finetune."""
        if edit_ds is None:
            return self.model

        # Stage 1: Add to retriever (stores keys for retrieval)
        print("[ReasonEdit] Stage 1: Adding to retriever...")
        self.retriever.edit(config, tokens, batch_history, edit_ds, train_ds)

        # Stage 2: Build training samples
        print("[ReasonEdit] Stage 2: Building training samples...")
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
            
            self._add_training_samples(image, question, answer, rationale)
            self._added_uids.add(uid)
            added += 1

        print(f"[ReasonEdit] Added {added} edits, k1: {len(self._k1_samples)}, k2: {len(self._k2_samples)}")

        # Stage 3: Train layer_edit on ALL accumulated samples
        print("[ReasonEdit] Stage 3: Training layer_edit...")
        self._train_on_all_samples()
        
        return self.model

    def retrieve_and_apply(self, image, question, prompt):
        """Retrieve facts; if found → prepend to prompt AND use layer_edit.
        
        Returns: (modified_prompt, is_edit)
        """
        facts = self.retriever._retrieve(image, question)
        is_edit = bool(facts)
        
        if self.switcher:
            self.switcher.use_edit = is_edit
        
        if is_edit:
            prompt = f"{self.prefix}{' '.join(facts)} {prompt}"
        
        return prompt, is_edit

    def apply_to_dataset(self, dataset):
        """Inference: prepend retrieved facts and toggle switcher per sample."""
        if self.switcher is None:
            return
        
        for ex in getattr(dataset, "data", []):
            # Store original prompt to avoid accumulation
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
        """Return statistics about stored keys and samples."""
        retriever_stats = self.retriever.get_stats()
        return {
            **retriever_stats,
            "num_k1_samples": len(self._k1_samples),
            "num_k2_samples": len(self._k2_samples),
            "num_edits": len(self._added_uids),
            "layer_edit_initialized": self.layer_edit is not None,
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
