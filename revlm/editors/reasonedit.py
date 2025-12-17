"""ReasonEdit: IKE_TUPLE retrieval + adapter finetuning."""

from copy import deepcopy
import torch
import torch.nn as nn
from .ike_tuple import IKE_TUPLE
from .utils import brackets_to_periods, parent_module


class ReasonEdit(nn.Module):
    """
    Stage 1: IKE_TUPLE retrieval
    Stage 2: Adapter finetuning (no masking)
    Inference: facts retrieved -> use layer_edit, else original
    """

    def __init__(self, config, model):
        super().__init__()
        cfg = getattr(config, "editor", config)

        self.ike_tuple = IKE_TUPLE(config, model)
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model

        self.ft_epochs = int(getattr(cfg, "ft_epochs", 1000))
        self.ft_lr = float(getattr(cfg, "edit_lr", 1e-3))
        self.ft_batch_size = int(getattr(cfg, "ft_batch_size", 4))
        self.early_stop_patience = int(getattr(cfg, "early_stop_patience", 20))

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

        opt = torch.optim.Adam(self.layer_edit.parameters(), lr=self.ft_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.ft_epochs))
        n = len(samples)
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
                tokens["labels"] = tokens["input_ids"].clone()  # no masking

                opt.zero_grad()
                loss = self.model(**tokens).loss
                loss.backward()
                opt.step()
                loss_sum += loss.item()
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

            if (ep + 1) % 10 == 0:
                print(f"[ReasonEdit] epoch {ep+1}/{self.ft_epochs} loss: {avg_loss:.4f}")

        self.switcher.use_edit = False  # default off; retrieve_and_apply toggles


class _Switcher(nn.Module):
    def __init__(self, layer_orig, layer_edit):
        super().__init__()
        self.layer_orig, self.layer_edit, self.use_edit = layer_orig, layer_edit, False

    def forward(self, *args, **kwargs):
        return (self.layer_edit if self.use_edit else self.layer_orig)(*args, **kwargs)
