"""ReasonEdit: IKE_TUPLE retrieval + adapter finetuning."""

from copy import deepcopy
import torch
import torch.nn as nn
from .ike_tuple import IKE_TUPLE
from .ike_cot import IKE_COT
from .ike_proto import IKE_PROTO
from .utils import brackets_to_periods, parent_module


class ReasonEdit(nn.Module):
    """
    Stage 1: Retrieval (self.ike = "cot" | "proto" | "tuple")
    Stage 2: Adapter finetuning
    Inference: facts retrieved -> use layer_edit, else original
    """

    def __init__(self, config, model):
        super().__init__()
        cfg = getattr(config, "editor", config)

        self.ike = "cot"
        if self.ike == "proto":
            self.ike_tuple = IKE_PROTO(config, model)
        elif self.ike == "tuple":
            self.ike_tuple = IKE_TUPLE(config, model)
        elif self.ike == "cot":
            self.ike_tuple = IKE_COT(config, model)
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model

        self.ft_epochs = int(getattr(cfg, "ft_epochs", 1000))
        self.ft_lr = float(getattr(cfg, "edit_lr", 1e-3))
        self.ft_batch_size = int(getattr(cfg, "ft_batch_size", 4))
        self.early_stop_patience = int(getattr(cfg, "early_stop_patience", 20))

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
        raw = inner_params[0]
        param_name = raw.rsplit(".", 1)[0] if raw.endswith((".weight", ".bias")) else raw
        self.edit_mod = parent_module(self.model, brackets_to_periods(param_name))
        self.layer_name = param_name.rsplit(".", 1)[-1]
        self.target_layer = getattr(self.edit_mod, self.layer_name)
        self.layer_edit = None
        self.switcher = None

    def generate(self, *a, **kw):
        return self.ike_tuple.generate(*a, **kw)

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def retrieve_and_apply(self, image, question, prompt):
        """Retrieve facts; if found -> prepend & use layer_edit. Returns (prompt, is_edit)."""
        facts = self.ike_tuple._retrieve(image, question, self.ike_tuple.k)
        is_edit = bool(facts)
        if self.switcher:
            self.switcher.use_edit = is_edit
        if is_edit:
            prompt = f"{self.ike_tuple.prefix}{' '.join(facts)} {prompt}"
        return prompt, is_edit

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        if edit_ds is None:
            return self.model

        # Stage 1: IKE_TUPLE retrieval
        print("[ReasonEdit] Stage 1: IKE_TUPLE retrieval...")
        self.ike_tuple.edit(config, tokens, batch_history, edit_ds, train_ds)

        # Stage 2: Train adapter
        print("[ReasonEdit] Stage 2: Training adapter...")
        self._train_adapter(edit_ds)
        return self.model

    def apply_to_dataset(self, dataset):
        """Inference-time: prepend retrieved facts and toggle switcher per sample."""
        if self.switcher is None:
            return
        for ex in getattr(dataset, "data", []):
            img, q, prompt = ex.get("image"), ex.get("question", ""), ex.get("prompt", "")
            if img is None or not prompt:
                continue
            new_prompt, _ = self.retrieve_and_apply(img, q, prompt)
            ex["prompt"] = new_prompt

    def _make_cot_answer_labels(self, tokens, batch):
        """Train on COT + answer only. Uses tokenizer to count positions."""
        labels = tokens["labels"].clone()
        input_ids = tokens["input_ids"]
        tokenizer = getattr(self.wrapper, "tokenizer", None)
        if not tokenizer:
            return labels

        for i, ex in enumerate(batch):
            cot = (ex.get("cot") or ex.get("rationale") or "").strip()
            prompt = ex.get("prompt", "")
            if not cot or not prompt or cot not in prompt:
                continue
            
            # Find answer start
            ans_mask = labels[i] != -100
            if not ans_mask.any():
                continue
            ans_start = ans_mask.nonzero(as_tuple=True)[0][0].item()
            
            # Find COT position in prompt string
            cot_idx = prompt.index(cot)
            before_cot = prompt[:cot_idx]
            up_to_cot_end = prompt[:cot_idx + len(cot)]
            
            # Count tokens for each portion
            prompt_toks = len(tokenizer.encode(prompt, add_special_tokens=False))
            before_cot_toks = len(tokenizer.encode(before_cot, add_special_tokens=False)) if before_cot else 0
            up_to_cot_end_toks = len(tokenizer.encode(up_to_cot_end, add_special_tokens=False))
            
            # Calculate token positions (text_start = where prompt begins in input)
            text_start = max(0, ans_start - prompt_toks)
            cot_tok_start = text_start + before_cot_toks
            cot_tok_end = min(text_start + up_to_cot_end_toks, ans_start)
            
            # Unmask COT tokens
            if cot_tok_start < cot_tok_end:
                labels[i, cot_tok_start:cot_tok_end] = input_ids[i, cot_tok_start:cot_tok_end]
        
        return labels

    def _train_adapter(self, edit_ds):
        samples = [ex for ex in getattr(edit_ds, "data", []) if ex.get("image") and ex.get("prompt")]
        if not samples:
            return

        # Init layer_edit
        self.layer_edit = deepcopy(self.target_layer)
        for p in self.layer_edit.parameters():
            p.requires_grad = True
        self.switcher = _Switcher(self.target_layer, self.layer_edit)
        setattr(self.edit_mod, self.layer_name, self.switcher)
        self.switcher.use_edit = True

        n = len(samples)
        opt = torch.optim.Adam(self.layer_edit.parameters(), lr=self.ft_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.ft_epochs))
        best_loss, patience_cnt = float('inf'), 0

        for ep in range(self.ft_epochs):
            perm = torch.randperm(n)
            loss_sum, cnt = 0.0, 0

            for start in range(0, n, self.ft_batch_size):
                batch = [samples[i] for i in perm[start:start + self.ft_batch_size].tolist()]
                tokens = self.wrapper.prepare_training_batch({
                    "images": [ex["image"] for ex in batch],
                    "prompts": [ex["prompt"] for ex in batch],
                    "golds": [{"label": ex.get("gold", {}).get("label", ""),
                               "label_train": ex.get("gold", {}).get("label_train", "")} for ex in batch],
                    "idxs": list(range(len(batch)))
                })
                # Option 1: Train on COT + Answer only (mask image/question)
                tokens["labels"] = self._make_cot_answer_labels(tokens, batch)

                opt.zero_grad()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = self.model(**tokens).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.layer_edit.parameters(), 1.0)  # Gradient clipping
                opt.step()
                loss_sum += loss.item()
                cnt += 1

            scheduler.step()  # Step per epoch
            avg_loss = loss_sum / max(1, cnt)

            if avg_loss < best_loss:
                best_loss, patience_cnt = avg_loss, 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.early_stop_patience:
                    print(f"[ReasonEdit] early stop at epoch {ep+1}, loss: {avg_loss:.4f}")
                    break

            if (ep + 1) % 10 == 0:
                print(f"[ReasonEdit] epoch {ep+1}/{self.ft_epochs} loss: {avg_loss:.4f}")

        self.switcher.use_edit = False  # default off; retrieve_and_apply toggles


class _Switcher(nn.Module):
    def __init__(self, layer_orig, layer_edit):
        super().__init__()
        self.layer_orig, self.layer_edit, self.use_edit = layer_orig, layer_edit, False

    def forward(self, *args, **kwargs):
        return (self.layer_edit if self.use_edit else self.layer_orig)(*args, **kwargs)
