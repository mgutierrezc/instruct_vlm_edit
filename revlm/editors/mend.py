import torch
from .utils import get_inner_params, brackets_to_periods, hook_model
import transformers
import torch.nn.functional as F


class GradientTransform(torch.nn.Module):
    """Transforms gradients for MEND (same backbone as GRACE, lightly adapted)."""

    def __init__(self, x_dim, delta_dim):
        super(GradientTransform, self).__init__()
        self.mlp1 = torch.nn.Linear(x_dim, x_dim)
        self.mlp2 = torch.nn.Linear(delta_dim, delta_dim)

    def forward(self, x, delta):
        # If we've got grads for each token, just grab the last representation
        if len(x.shape) == 3:
            x = x[:, -1, :]
            delta = delta[:, -1, :]

        # Ensure dtypes match Linear weights (handles bf16 / fp16 backbones)
        x = x.to(self.mlp1.weight.dtype)
        delta = delta.to(self.mlp2.weight.dtype)

        return self.mlp1(x), self.mlp2(delta)


def get_shape(p, model):
    """Get shape for gradient transform (GRACE-style logic)."""
    if isinstance(model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
        return p.shape
    # Generic linear layer: weight [out, in] → (in, out)
    return (p.shape[1], p.shape[0])


class MEND(torch.nn.Module):
    """MEND: Model Editing Networks using Gradient Decomposition (GRACE version online editing)."""

    def __init__(self, config, model, tokenizer, device, mend=None):
        super().__init__()

        # Unwrap VLM wrapper (e.g., VQAModel) to get underlying HF model
        if mend is None:
            self.model = model.model if hasattr(model, "model") else model
        else:
            self.model = model

        self.tokenizer = tokenizer
        self.device = device
        self.config = config

        # Memory-focused tweaks that do NOT change the editing algorithm:
        # - disable KV cache during editing
        # - enable gradient checkpointing when available
        # - enable input gradients if the model supports it
        core_model = self.model
        if hasattr(core_model, "config") and hasattr(core_model.config, "use_cache"):
            core_model.config.use_cache = False
        if hasattr(core_model, "enable_input_require_grads"):
            core_model.enable_input_require_grads()
        if hasattr(core_model, "gradient_checkpointing_enable"):
            core_model.gradient_checkpointing_enable()
        elif hasattr(core_model, "enable_gradient_checkpointing"):
            core_model.enable_gradient_checkpointing()

        # Use revlm NestedConfig: inner_params is already flattened
        params_dict = dict(self.model.named_parameters())
        self.bias_map = {}
        self.pnames = []
        for inner in config.inner_params:
            pname = brackets_to_periods(inner)
            self.pnames.append(pname)
            if pname.endswith(".weight"):
                bias_name = pname[:-7] + ".bias"
                if bias_name in params_dict:
                    self.bias_map[pname] = bias_name

        # Install hooks that populate p.weight.__x__ and p.weight.__delta__
        hook_model(self.model, self.pnames)

        # GPT-2 uses transposed convention; others (VLMs) use standard [out, in]
        if not isinstance(
            self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel
        ):
            transpose = False
        else:
            transpose = True
        self._transpose = transpose

        # Build (or re-use) the GradientTransform hypernets
        if mend is None:
            self.mend = torch.nn.ModuleDict({})
            for n, p in get_inner_params(self.model.named_parameters(), self.pnames):
                shape = get_shape(p, self.model)
                if transpose:
                    # GPT-2: keep original orientation
                    self.mend[n.replace(".", "#")] = GradientTransform(
                        shape[0], shape[1]
                    ).to(device)
                else:
                    # Generic (VLM) case: x_dim=in, delta_dim=out
                    self.mend[n.replace(".", "#")] = GradientTransform(
                        shape[0], shape[1]
                    ).to(device)
        else:
            self.mend = mend

        self.loss = None
        self.losses = []
        self._key_idx = -1

    def outer_parameters(self):
        return list(self.mend.parameters())

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def get_model_loss(self, model, logits, batch):
        # Optional hook for custom loss (kept from GRACE code)
        if hasattr(model, "get_loss"):
            return model.get_loss(logits, batch)
        if hasattr(model, "model") and hasattr(model.model, "get_loss"):
            return model.model.get_loss(logits, batch)
        return None

    def edit(self, config, tokens, batch_history):
        """
        Online training of the MEND hypernetwork on a single edit example.

        Exactly the GRACE-style inner loop, but using revlm's `config.edit_lr`
        and `config.n_iter` instead of Hydra dicts.
        """
        del batch_history  # not used in this simple variant

        opt = torch.optim.Adam(self.outer_parameters(), lr=float(config.edit_lr))
        editor_config = getattr(config, 'editor', config)
        n_iter = int(getattr(editor_config, 'n_iter', config.n_iter))
        early_stop_patience = editor_config.early_stop_patience
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_iter))
        self.losses = []
        self._key_idx = self._compute_key_idx(tokens)
        
        best_loss = float('inf')
        patience_counter = 0

        for i in range(n_iter):
            self.edit_step(tokens)

            outputs = self.model(**tokens)
            loss = outputs.loss if hasattr(outputs, "loss") else None

            if loss is None:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                if "labels" in tokens:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        tokens["labels"].view(-1),
                        ignore_index=-100,
                    )
                else:
                    break

            loss_value = loss.detach().cpu().item()
            self.losses.append(loss_value)
            self.loss = loss

            self.loss.backward()
            opt.step()
            opt.zero_grad()
            scheduler.step()
            
            # Early stopping: check if loss improved
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break
            
            # Print loss every 10 iterations or on first/last iteration
            if (i + 1) % 10 == 0 or i == 0 or i == n_iter - 1:
                print(f"[mend] iter {i+1}/{n_iter} - loss: {loss_value:.4f}")

        return self.model

    def edit_step(self, batch):
        """
        Single inner step:
        - run base model on the edit batch to obtain gradients & hooks (x, δ),
        - pass (x, δ) through GradientTransform,
        - form low-rank updates and apply them via a functional copy.
        """
        outputs = self.model(**batch)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = outputs.loss if hasattr(outputs, "loss") else None

        if loss is None:
            if "labels" in batch:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100,
                )
            else:
                return self.model

        # Backprop once to populate p.weight.__x__ and p.weight.__delta__ via hooks
        loss.backward()

        # Use hypernetwork to transform (x, δ) into update factors
        transformed_factors = {}
        for n, p in get_inner_params(self.model.named_parameters(), self.pnames):
            x = self._select_token(p.__x__)
            delta = self._select_token(p.__delta__)
            transformed_factors[n] = self.mend[n.replace(".", "#")](x, delta)

        # Build low-rank update (outer product) per parameter
        mean_grads = {
            n: torch.matmul(delta.view(-1, 1), x.view(1, -1))
            for n, (x, delta) in transformed_factors.items()
        }

        # Clear gradients on base model before constructing functional version
        self.model.zero_grad()

        updates = mean_grads
        bias_updates = {
            n: delta.mean(dim=0) if delta.dim() > 1 else delta
            for n, (_, delta) in transformed_factors.items()
        }

        param_dict = dict(self.model.named_parameters())

        with torch.no_grad():
            for n, p in param_dict.items():
                if n in updates:
                    upd = updates[n].T if self._transpose else updates[n]
                    upd = upd.to(p.dtype)
                    p.add_(upd)
                    if n in self.bias_map:
                        bias_name = self.bias_map[n]
                        bias_param = param_dict[bias_name]
                        b_upd = bias_updates[n]
                        if b_upd.dim() == 2:
                            b_upd = b_upd.mean(dim=0)
                        b_upd = b_upd.to(bias_param.dtype)
                        bias_param.add_(b_upd)

        loss.detach()

    def _select_token(self, tensor):
        if tensor.dim() == 3:
            idx = self._key_idx if self._key_idx is not None and self._key_idx >= 0 else tensor.shape[1] - 1
            idx = min(idx, tensor.shape[1] - 1)
            tensor = tensor[:, idx, :]
        return tensor

    def _compute_key_idx(self, tokens):
        labels = tokens.get("labels")
        if labels is None:
            return -1
        if labels.dim() == 1:
            non_masked = (labels != -100)
            if non_masked.any():
                return non_masked.nonzero().max().item()
            return labels.numel() - 1
        non_masked = (labels != -100)
        if non_masked.any():
            return non_masked.sum(dim=1).max().item() - 1
        return labels.shape[1] - 1

