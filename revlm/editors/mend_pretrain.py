"""
MEND with Pretraining: Model Editing Networks using Gradient Decomposition.

BalancEdit-style implementation with memory optimizations for large VLMs.
Uses torch.func.functional_call instead of higher library for better memory efficiency.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Dict, List, Optional, Tuple
from functools import partial
from .utils import get_inner_params, brackets_to_periods, hook_model

# Use torch.func for functional calls (PyTorch 2.0+)
try:
    from torch.func import functional_call
    HAS_FUNC = True
except ImportError:
    try:
        from functorch import make_functional_with_buffers
        HAS_FUNC = False
    except ImportError:
        HAS_FUNC = False


def update_counter(x, m, s, k):
    """Online update of running mean and variance (Welford's algorithm). From BalancEdit."""
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)
    return new_m, new_s


class GradientTransform(nn.Module):
    """
    Hypernetwork that transforms (u, v) gradients. BalancEdit-style with:
    - 2-layer MLPs with 2x hidden dim
    - Online running normalization
    - Proper batch handling
    """
    
    def __init__(self, x_dim: int, delta_dim: int, n_hidden: int = 1, normalize: bool = True):
        super().__init__()
        self.x_dim, self.delta_dim = x_dim, delta_dim
        self.normalize = normalize
        
        # BalancEdit-style MLPs: 2-layer with 2x hidden
        self.mlp1 = nn.Sequential(
            nn.Linear(x_dim, x_dim * 2),
            nn.ReLU(),
            nn.Linear(x_dim * 2, x_dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(delta_dim, delta_dim * 2),
            nn.ReLU(),
            nn.Linear(delta_dim * 2, delta_dim)
        )
        
        # Running stats for normalization (BalancEdit style)
        self.norm_init = False
        self.register_buffer("u_mean", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_mean", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_std", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_std", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_s", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_s", torch.full((delta_dim,), float("nan")))
        self.register_buffer("k", torch.full((1,), float("nan")))
        
        self._init_weights()
    
    def _init_weights(self):
        for mlp in [self.mlp1, self.mlp2]:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform (u, v) through MLPs with optional normalization."""
        u, v = u.to(torch.float32), v.to(torch.float32)
        
        u_ = u.view(-1, u.shape[-1])
        v_ = v.view(-1, v.shape[-1])
        
        # Skip zero gradients
        nz_mask = (u_ != 0).any(-1) & (v_ != 0).any(-1)
        if not nz_mask.any():
            return u_[:1], v_[:1]
        u_ = u_[nz_mask]
        v_ = v_[nz_mask]
        
        # Update running stats during training
        if self.training and self.normalize:
            for idx in range(u_.shape[0]):
                if not self.norm_init:
                    self.u_mean = u_[idx].clone().detach()
                    self.v_mean = v_[idx].clone().detach()
                    self.u_s.zero_()
                    self.v_s.zero_()
                    self.k[:] = 1
                    self.norm_init = True
                else:
                    self.k += 1
                    self.u_mean, self.u_s = update_counter(u_[idx].detach(), self.u_mean, self.u_s, self.k)
                    self.v_mean, self.v_s = update_counter(v_[idx].detach(), self.v_mean, self.v_s, self.k)
            
            if self.k >= 2:
                self.u_std = (self.u_s / (self.k - 1)) ** 0.5
                self.v_std = (self.v_s / (self.k - 1)) ** 0.5
        
        # Normalize inputs
        if self.normalize and self.norm_init and self.k >= 2:
            u_input = (u_ - self.u_mean) / (self.u_std + 1e-7)
            v_input = (v_ - self.v_mean) / (self.v_std + 1e-7)
        else:
            u_input = u_
            v_input = v_
        
        # Transform through MLPs
        u_out = self.mlp1(u_input)
        v_out = self.mlp2(v_input)
        
        # CRITICAL: Normalize outputs to have same magnitude as inputs
        # This prevents the hypernetwork from producing arbitrarily large updates
        u_out = u_out * (u_.norm(dim=-1, keepdim=True) / (u_out.norm(dim=-1, keepdim=True) + 1e-8))
        v_out = v_out * (v_.norm(dim=-1, keepdim=True) / (v_out.norm(dim=-1, keepdim=True) + 1e-8))
        
        return u_out, v_out


def get_shape(p, model):
    """Get (x_dim, delta_dim) for GradientTransform."""
    if isinstance(model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
        return p.shape
    return (p.shape[1], p.shape[0])  # weight [out, in] -> (in, out)


class MEND_Pretrain(nn.Module):
    """
    Pretrained MEND (BalancEdit-style) with memory optimizations for VLMs.
    
    Key features from BalancEdit:
    - GradientTransform with normalized MLPs
    - Learned per-parameter edit learning rates
    - Einsum for batch-aware outer products
    - Proper gradient flow through edit step
    
    Memory optimizations:
    - Uses torch.func.functional_call instead of higher.monkeypatch
    - Gradient checkpointing enabled
    - Mixed precision support
    - Periodic cache clearing
    
    Usage:
        editor = MEND_Pretrain(config, model, tokenizer, device)
        editor.pretrain(train_data, epochs=5)
        edited_model = editor.edit(config, tokens, None)
    """
    
    def __init__(self, config, model, tokenizer, device, checkpoint_path: Optional[str] = None):
        super().__init__()
        
        # Unwrap VLM wrapper
        self.model = model.model if hasattr(model, "model") else model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        # Memory optimizations
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()
        
        # Get parameter names
        params_dict = dict(self.model.named_parameters())
        self.pnames = [brackets_to_periods(p) for p in config.inner_params if brackets_to_periods(p) in params_dict]
        
        # Install hooks
        hook_model(self.model, self.pnames)
        
        # GPT-2 uses transposed weights
        self._is_gpt2 = isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel)
        
        # Build hypernetworks (BalancEdit style)
        self.mend = nn.ModuleDict()
        for n, p in get_inner_params(self.model.named_parameters(), self.pnames):
            shape = get_shape(p, self.model)
            self.mend[n.replace(".", "#")] = GradientTransform(shape[0], shape[1]).to(device)
        
        # Learnable per-parameter edit LRs (BalancEdit style)
        edit_lr = float(getattr(getattr(config, 'editor', config), 'edit_lr', config.edit_lr))
        self.edit_lrs = nn.Parameter(torch.tensor([edit_lr] * len(self.pnames)))
        
        self.is_pretrained = False
        self.losses = []
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
    
    def outer_parameters(self):
        """Parameters to train (hypernetwork + edit LRs)."""
        return list(self.mend.parameters()) + [self.edit_lrs]
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def _compute_loss(self, logits, labels):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    
    def _get_transformed_grads(self, batch: Dict) -> Tuple[Dict, torch.Tensor]:
        """
        Forward + backward to populate hooks, then transform through hypernetwork.
        Returns transformed factors and the model loss.
        """
        outputs = self.model(**batch)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else self._compute_loss(logits, batch["labels"])
        
        # Backward to populate hooks
        loss.backward(retain_graph=True)
        
        # Transform through hypernetwork (keeps gradients for training)
        transformed = {}
        for n, p in get_inner_params(self.model.named_parameters(), self.pnames):
            if hasattr(p, "__x__") and hasattr(p, "__delta__"):
                transformed[n] = self.mend[n.replace(".", "#")](p.__x__, p.__delta__)
        
        self.model.zero_grad()
        return transformed, loss
    
    def _compute_updates(self, transformed: Dict) -> Dict[str, torch.Tensor]:
        """Compute parameter updates from transformed gradients using einsum."""
        targ = "ij" if self._is_gpt2 else "ji"
        updates = {}
        for n, (u, v) in transformed.items():
            updates[n] = torch.einsum(f"bi,bj->{targ}", u, v)
        return updates
    
    def _apply_updates_get_loss(self, batch: Dict, updates: Dict) -> torch.Tensor:
        """
        Apply updates to model and compute loss.
        Uses functional_call for memory efficiency (doesn't patch whole model).
        """
        # Build new parameter dict with updates applied
        param_dict = dict(self.model.named_parameters())
        new_params = {}
        
        for n, p in param_dict.items():
            if n in updates:
                idx = self.pnames.index(n)
                new_params[n] = p + self.edit_lrs[idx] * updates[n].to(p.dtype)
            else:
                new_params[n] = p
        
        # Use functional_call if available (PyTorch 2.0+)
        if HAS_FUNC:
            outputs = functional_call(self.model, new_params, args=(), kwargs=batch)
        else:
            # Fallback: temporarily apply updates
            originals = {}
            with torch.no_grad():
                for n in updates:
                    if n in param_dict:
                        originals[n] = param_dict[n].data.clone()
                        idx = self.pnames.index(n)
                        param_dict[n].data.add_((self.edit_lrs[idx] * updates[n]).to(param_dict[n].dtype))
            
            outputs = self.model(**batch)
            
            # Restore
            with torch.no_grad():
                for n, orig in originals.items():
                    param_dict[n].data.copy_(orig)
        
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        return self._compute_loss(logits, batch["labels"])
    
    def pretrain(
        self,
        train_data: List[Dict],
        val_data: Optional[List[Dict]] = None,
        epochs: int = 5,
        lr: float = 1e-4,
        lr_lr: float = 1e-3,
        loc_coef: float = 0.1,
        log_interval: int = 10,
        early_stop_patience: int = 5,
        save_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Pretrain hypernetwork on edit dataset (BalancEdit style).
        
        The training objective:
        1. Edit loss: loss on edited model should be low
        2. Locality loss (optional): KL divergence from original should be small
        
        train_data: List of {'edit_input': {...}, 'loc_input': {...} (optional)}
        """
        print(f"[MEND_Pretrain] Training on {len(train_data)} examples, {len(self.pnames)} params, edit_lr={self.edit_lrs.data[0].item():.6f}")
        
        opt = torch.optim.Adam(self.mend.parameters(), lr=lr)
        lr_opt = torch.optim.Adam([self.edit_lrs], lr=lr_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_data))
        
        best_loss = float('inf')
        patience = 0
        
        self.mend.train()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for step, ex in enumerate(train_data, 1):
                edit_batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in ex['edit_input'].items()}
                
                # Get locality batch if available
                loc_batch = ex.get('loc_input')
                if loc_batch:
                    loc_batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in loc_batch.items()}
                
                # Step 1: Get transformed gradients
                transformed, base_loss = self._get_transformed_grads(edit_batch)
                
                # Step 2: Compute updates
                updates = self._compute_updates(transformed)
                
                # Step 3: Compute edit loss on edited model
                edit_loss = self._apply_updates_get_loss(edit_batch, updates)
                
                # Step 4: Locality loss (optional) + edit_lr regularization
                total_loss = edit_loss
                if loc_batch and loc_coef > 0:
                    with torch.no_grad():
                        base_out = self.model(**loc_batch)
                        base_logits = base_out.logits if hasattr(base_out, "logits") else base_out
                    
                    loc_loss_val = self._apply_updates_get_loss(loc_batch, updates)
                    # KL divergence approximation
                    loc_loss = loc_coef * loc_loss_val
                    total_loss = edit_loss + loc_loss
                
                # L2 regularization on edit_lrs to prevent them from growing too large
                lr_reg = 0.01 * (self.edit_lrs ** 2).sum()
                total_loss = total_loss + lr_reg
                
                # Step 5: Backward and update
                opt.zero_grad()
                lr_opt.zero_grad()
                total_loss.backward()
                
                nn.utils.clip_grad_norm_(self.outer_parameters(), 1.0)
                opt.step()
                lr_opt.step()
                scheduler.step()
                
                # Clamp edit_lrs to reasonable range
                with torch.no_grad():
                    self.edit_lrs.clamp_(1e-6, 1.0)
                
                epoch_losses.append(edit_loss.item())
                
                if step % log_interval == 0:
                    avg = sum(epoch_losses[-log_interval:]) / min(log_interval, len(epoch_losses))
                    print(f"[MEND] E{epoch+1} S{step}/{len(train_data)} loss={avg:.4f}")
                
                # Memory management
                if step % 20 == 0:
                    torch.cuda.empty_cache()
            
            epoch_avg = sum(epoch_losses) / len(epoch_losses)
            self.losses.append(epoch_avg)
            print(f"[MEND] Epoch {epoch+1} loss={epoch_avg:.4f} edit_lr={self.edit_lrs.data[0].item():.6f}")
            
            # Early stopping check
            if epoch_avg < best_loss:
                best_loss = epoch_avg
                patience = 0
                if save_path:
                    self.save_checkpoint(save_path)
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print(f"[MEND] Early stopping at epoch {epoch+1}")
                    break
        
        self.is_pretrained = True
        if save_path and patience < early_stop_patience:
            self.save_checkpoint(save_path)
        print("[MEND_Pretrain] Done!")
    
    def edit(self, config, tokens: Dict, batch_history=None):
        """
        Fast single-pass edit using pretrained hypernetwork.
        For multi-sample editing, use edit_batch() instead.
        """
        if not self.is_pretrained:
            print("[MEND_Pretrain] WARNING: Not pretrained!")
        
        self.mend.eval()
        
        # Get transformed gradients
        transformed, _ = self._get_transformed_grads(tokens)
        
        # Compute updates
        updates = self._compute_updates(transformed)
        
        # Apply updates permanently to model
        param_dict = dict(self.model.named_parameters())
        with torch.no_grad():
            for idx, n in enumerate(self.pnames):
                if n in updates and n in param_dict:
                    param_dict[n].data.add_((self.edit_lrs[idx] * updates[n]).to(param_dict[n].dtype))
        
        return self.model
    
    def edit_batch(self, all_tokens: List[Dict]):
        """Edit model with multiple samples using running average (constant memory)."""
        self.mend.eval()
        n_samples = len(all_tokens)
        print(f"[MEND_Pretrain] Computing updates for {n_samples} samples...")
        
        # Running average of updates
        running_avg = {n: None for n in self.pnames}
        for i, tokens in enumerate(all_tokens):
            transformed, _ = self._get_transformed_grads(tokens)
            updates = self._compute_updates(transformed)
            for n in self.pnames:
                if n in updates:
                    u = updates[n].detach()
                    running_avg[n] = u.clone() if running_avg[n] is None else running_avg[n] + (u - running_avg[n]) / (i + 1)
            if (i + 1) % 100 == 0 or i == n_samples - 1:
                print(f"[MEND_Pretrain] Processed {i+1}/{n_samples}")
        
        # Apply updates (scaled by edit_lr only, no normalization)
        param_dict = dict(self.model.named_parameters())
        with torch.no_grad():
            for idx, n in enumerate(self.pnames):
                if running_avg[n] is not None and n in param_dict:
                    avg = running_avg[n]
                    orig_norm = param_dict[n].norm().item()
                    scaled = self.edit_lrs[idx] * avg
                    final_norm = scaled.norm().item()
                    print(f"[MEND_Pretrain] {n}: orig={orig_norm:.2f} upd={final_norm:.2f} rel_change={final_norm/orig_norm:.4f}")
                    param_dict[n].data.add_(scaled.to(param_dict[n].dtype))
        
        print(f"[MEND_Pretrain] Applied edits from {n_samples} samples, edit_lr={self.edit_lrs.data[0].item():.6f}")
        return self.model
    
    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            'mend': self.mend.state_dict(),
            'edit_lrs': self.edit_lrs.data,
            'pnames': self.pnames,
            'is_pretrained': True,
            'losses': self.losses,
        }, path)
        print(f"[MEND_Pretrain] Saved to {path}")
    
    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.mend.load_state_dict(ckpt['mend'])
        self.edit_lrs.data = ckpt['edit_lrs'].to(self.device)
        self.is_pretrained = ckpt.get('is_pretrained', True)
        self.losses = ckpt.get('losses', [])
        print(f"[MEND_Pretrain] Loaded from {path}")
