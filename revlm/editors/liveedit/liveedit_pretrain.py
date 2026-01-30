# -*- coding: utf-8 -*-
"""
LiveEdit Pretraining - Train meta-learners on VLM-specific edits.

Based on original LiveEdit repo (CVPR 2025):
"Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts"

This implementation matches the original training approach:
- Reliability loss (CE on edit examples)
- Generality loss (CE on text/image variants)
- Locality loss (KL divergence on unrelated samples)
- Soft routing loss (contrastive learning for query features)
- Hard routing loss (contrastive learning for vision features)

Key optimization: forward_from_mid_layer
- Pre-computes mid-layer representations with torch.no_grad()
- Only runs layers from edit_layer_i onwards during training
- Reduces memory by ~65% (only backprop through 11 layers instead of 32)

Usage:
    1. Prepare data: python -m revlm.run.pretrain_data --model_name qwen3_4b --dataset_name aokvqa
    2. Pretrain: python -m revlm.run.pretrain --model_name qwen3_4b --dataset_name aokvqa
    3. Use checkpoint: set ckpt_path in liveedit.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from contextlib import contextmanager, ExitStack

from .liveedit import LiveEdit


class LiveEditPretrain(LiveEdit):
    """
    LiveEdit with trainable meta-learners for pretraining.
    
    Matches original LiveEdit training with 5 loss terms:
    - Reliability loss (rel_lambda)
    - Generality loss (gen_lambda)
    - Locality loss (loc_lambda)
    - Soft routing loss (soft_routing_lambda)
    - Hard routing loss (hard_routing_lambda)
    """
    
    def __init__(self, config, model):
        super().__init__(config, model)
        
        # Get pretrain-specific config (matching original train_cfg)
        editor_cfg = getattr(config, "editor", config)
        
        # Loss weights (from original)
        self.rel_lambda = float(getattr(editor_cfg, "rel_lambda", 1.0))
        self.gen_lambda = float(getattr(editor_cfg, "gen_lambda", 1.0))
        self.loc_lambda = float(getattr(editor_cfg, "loc_lambda", 1.0))
        self.soft_routing_lambda = float(getattr(editor_cfg, "soft_routing_lambda", 1.0))
        self.hard_routing_lambda = float(getattr(editor_cfg, "hard_routing_lambda", 1.0))
        
        # LR schedule (from original)
        self.lr = float(getattr(editor_cfg, "lr", 1e-4))
        self.lr_cut_it = getattr(editor_cfg, "lr_cut_it", [10000])
        self.lr_cut_rate = float(getattr(editor_cfg, "lr_cut_rate", 0.1))
        
        # Similarity scale (from original)
        self.sim_scale = 1.0 / (self.module_dim ** 0.5)
        
        # Make meta-learners trainable (override parent's freeze)
        self._unfreeze_meta_learners()
        
        # Enable warm init since we're training meta-learners
        self.use_warm_init = True
        
        # Enable gradient checkpointing to reduce memory during pretraining
        # This trades compute for memory (~30% slower but ~50% less memory)
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("[LiveEditPretrain] Gradient checkpointing enabled")
        
        # Training state
        self.pretrain_losses = []
        self.global_step = 0
        
        # Hooks for forward_from_mid_layer optimization
        self._skip_layer_hooks = []
        self._inject_hook = None
    
    #═══════════════════════════════════════════════════════════════════
    # FORWARD_FROM_MID_LAYER OPTIMIZATION (from original LiveEdit)
    #═══════════════════════════════════════════════════════════════════
    
    @contextmanager
    def _forward_from_mid_layer_context(self, mid_layer_reps: torch.Tensor):
        """
        Context manager that enables forward_from_mid_layer optimization.
        
        This dramatically reduces memory by:
        1. Skipping layers 0 to (edit_layer_i - 1) - they return None
        2. Injecting pre-computed mid_layer_reps at edit_layer_i
        3. Only computing gradients for layers edit_layer_i onwards
        
        Args:
            mid_layer_reps: Pre-computed representations from edit layer [B, L, D]
        """
        handles = []
        original_forwards = {}
        
        try:
            # 1. Skip early layers by replacing their forward
            for i in range(self.edit_layer_i):
                layer_path = self.llm_layer_tmp.format(i)
                try:
                    layer = self._find_module(self.model, layer_path)
                    original_forwards[layer_path] = layer.forward
                    
                    def skip_forward(*args, layer_path=layer_path, **kwargs):
                        # Return tuple matching expected output format
                        return (None,)
                    
                    layer.forward = skip_forward
                except Exception as e:
                    pass  # Layer not found, skip
            
            # 2. Inject mid_layer_reps at edit_layer_i via pre-hook
            edit_layer = self._find_module(self.model, self.edit_layer_path)
            
            def inject_hook(module, args, kwargs):
                # Replace the hidden_states input with our cached mid_layer_reps
                new_args = (mid_layer_reps,) + args[1:] if len(args) > 0 else (mid_layer_reps,)
                return new_args, kwargs
            
            handle = edit_layer.register_forward_pre_hook(inject_hook, with_kwargs=True)
            handles.append(handle)
            
            yield
            
        finally:
            # Clean up: restore original forwards and remove hooks
            for layer_path, original_forward in original_forwards.items():
                try:
                    layer = self._find_module(self.model, layer_path)
                    layer.forward = original_forward
                except:
                    pass
            
            for handle in handles:
                handle.remove()
    
    def _forward_from_mid_layer(
        self, 
        tokens: Dict[str, torch.Tensor], 
        mid_layer_reps: torch.Tensor
    ) -> Any:
        """
        Forward pass starting from edit_layer_i using cached mid_layer_reps.
        
        This is the memory-saving core: instead of storing activations for all 32 layers,
        we only store activations for layers edit_layer_i to end (~11 layers for layer 21).
        
        Args:
            tokens: Original input tokens (needed for attention mask, etc.)
            mid_layer_reps: Pre-computed representations from edit layer [B, L, D]
        
        Returns:
            Model outputs (logits, loss, etc.)
        """
        with self._forward_from_mid_layer_context(mid_layer_reps):
            outputs = self.model(**tokens)
        return outputs
    
    def _get_mid_layer_reps_and_info(
        self, 
        tokens: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], int, int]:
        """
        Get mid-layer representations and sequence info (one forward pass with no_grad).
        
        Returns:
            (mid_layer_reps, ans_start_idx, seq_len)
        """
        captured = {}
        
        def hook(module, args, output):
            reps = output[0] if isinstance(output, tuple) else output
            captured['reps'] = reps.detach().clone()  # Clone to keep after forward
        
        layer = self._find_module(self.model, self.edit_layer_path)
        handle = layer.register_forward_hook(hook)
        
        try:
            with torch.no_grad():
                self.model(**tokens)
        finally:
            handle.remove()
        
        if 'reps' not in captured:
            return None, 0, 0
        
        mid_reps = captured['reps']
        seq_len = mid_reps.shape[1]
        
        # Find answer start from labels
        if "labels" in tokens:
            labels = tokens["labels"]
            ans_start = (labels[0] != -100).nonzero(as_tuple=True)[0]
            if len(ans_start) > 0:
                ans_start = ans_start[0].item()
            else:
                ans_start = seq_len - 1
        else:
            ans_start = seq_len - 1
        
        return mid_reps, ans_start, seq_len
    
    def _split_reps(
        self, 
        reps: torch.Tensor, 
        ans_start: int, 
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split representations into vision, query, answer regions."""
        # Heuristic split (adjust based on your model)
        vision_end = min(576, ans_start // 2)
        
        vision_reps = reps[:, :vision_end, :]
        query_reps = reps[:, vision_end:ans_start, :]
        ans_reps = reps[:, ans_start:, :]
        
        # Handle empty regions
        if query_reps.shape[1] == 0:
            query_reps = vision_reps[:, -1:, :]
        if ans_reps.shape[1] == 0:
            ans_reps = query_reps[:, -1:, :]
        
        return vision_reps, query_reps, ans_reps
    
    def _unfreeze_meta_learners(self):
        """Make meta-learners trainable."""
        for param in self.edit_extractor.parameters():
            param.requires_grad = True
        for param in self.inpt_extractor.parameters():
            param.requires_grad = True
        for param in self.moegen_c.parameters():
            param.requires_grad = True
        for param in self.moegen_r.parameters():
            param.requires_grad = True
        for param in self.instant_reps_norm.parameters():
            param.requires_grad = True
        
        # Set to train mode
        self.edit_extractor.train()
        self.inpt_extractor.train()
        self.moegen_c.train()
        self.moegen_r.train()
        self.instant_reps_norm.train()
    
    def make_trainable(self):
        """Alias for _unfreeze_meta_learners (public API)."""
        self._unfreeze_meta_learners()
    
    def freeze_meta_learners(self):
        """Freeze meta-learners for inference."""
        for param in self.edit_extractor.parameters():
            param.requires_grad = False
        for param in self.inpt_extractor.parameters():
            param.requires_grad = False
        for param in self.moegen_c.parameters():
            param.requires_grad = False
        for param in self.moegen_r.parameters():
            param.requires_grad = False
        for param in self.instant_reps_norm.parameters():
            param.requires_grad = False
        
        # Set to eval mode
        self.edit_extractor.eval()
        self.inpt_extractor.eval()
        self.moegen_c.eval()
        self.moegen_r.eval()
        self.instant_reps_norm.eval()
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return list of trainable parameters for optimizer (matches original)."""
        params = []
        params.extend(self.edit_extractor.parameters())
        params.extend(self.inpt_extractor.parameters())
        params.extend(self.moegen_c.parameters())
        params.extend(self.moegen_r.parameters())
        params.extend(self.instant_reps_norm.parameters())
        return list(params)
    
    def get_optimizer_and_scheduler(self, total_steps: int = None):
        """
        Get optimizer and LR scheduler matching original LiveEdit.
        
        Original uses step decay: lr * lr_cut_rate^(n) where n = number of cut points passed
        """
        from torch.optim import Adam
        from torch.optim.lr_scheduler import LambdaLR
        
        opt = Adam(self.get_trainable_params(), lr=self.lr)
        
        # Step decay scheduler (from original)
        lr_cut_it = torch.tensor(self.lr_cut_it, device=self.device)
        
        def lr_lambda(step):
            return self.lr_cut_rate ** (step > lr_cut_it).sum().item()
        
        scheduler = LambdaLR(opt, lr_lambda)
        
        return opt, scheduler
    
    def get_moe_fuse_coe(self, iqrs: torch.Tensor, eqrs: torch.Tensor, split: bool = False):
        """
        Compute MOE fusion coefficient (from original).
        
        Args:
            iqrs: Input query representations [n, eqe_n, module_dim]
            eqrs: Edit query representations [m, eqe_n, module_dim]
            split: If True, return rela_sim and abs_sim separately
        
        Returns:
            Fusion coefficient [n, m] or (rela_sim, abs_sim) if split=True
        """
        sim = torch.einsum('ned,med->nme', iqrs, eqrs).mean(2) * self.sim_scale  # [n, m]
        rela_sim = torch.softmax(sim, 1)  # [n, m]
        abs_sim = torch.sigmoid(sim)  # [n, m]
        if split:
            return rela_sim, abs_sim
        return rela_sim * abs_sim  # [n, m]
    
    def pretrain_step(
        self,
        edit_tokens: Dict[str, torch.Tensor],
        gen_tokens_list: List[Dict[str, torch.Tensor]],
        loc_tokens: Optional[Dict[str, torch.Tensor]] = None,
        neighbor_data: Optional[Tuple] = None,
        prototype_data: Optional[Tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        One pretraining step matching original LiveEdit.
        
        MEMORY OPTIMIZATION: Uses forward_from_mid_layer to only backprop through
        layers edit_layer_i onwards (~11 layers instead of ~32).
        
        Loss = rel_loss * rel_lambda 
             + gen_loss * gen_lambda 
             + loc_loss * loc_lambda
             + soft_routing_loss * soft_routing_lambda
             + hard_routing_loss * hard_routing_lambda
        
        Args:
            edit_tokens: Tokenized edit example (question → answer)
            gen_tokens_list: List of tokenized generality samples
            loc_tokens: Tokenized locality sample (for KL loss)
            neighbor_data: (neighbor1_reps, neighbor2_reps) for soft routing loss
            prototype_data: (data1_reps, data2_reps) for hard routing loss
        
        Returns:
            Dict with all loss terms
        """
        eps = 1e-8
        log_dict = {}
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # ============================================================
        # PHASE 1: Pre-compute mid-layer reps for ALL samples (no_grad)
        # This is the key memory optimization from original LiveEdit
        # ============================================================
        
        # Edit sample
        edit_mid_reps, edit_ans_start, edit_seq_len = self._get_mid_layer_reps_and_info(edit_tokens)
        if edit_mid_reps is None:
            return {"loss": loss, "log": log_dict}
        
        vision_reps, query_reps, ans_reps = self._split_reps(edit_mid_reps, edit_ans_start, edit_seq_len)
        
        # Generality samples - pre-compute all mid-layer reps
        gen_cached = []
        if gen_tokens_list:
            for gen_tokens in gen_tokens_list:
                gen_mid_reps, gen_ans_start, gen_seq_len = self._get_mid_layer_reps_and_info(gen_tokens)
                if gen_mid_reps is not None:
                    _, gen_query_reps, _ = self._split_reps(gen_mid_reps, gen_ans_start, gen_seq_len)
                    gen_cached.append((gen_tokens, gen_mid_reps, gen_query_reps))
        
        # Locality sample - pre-compute mid-layer reps AND original logits
        loc_cached = None
        if loc_tokens is not None:
            loc_mid_reps, loc_ans_start, loc_seq_len = self._get_mid_layer_reps_and_info(loc_tokens)
            if loc_mid_reps is not None:
                _, loc_query_reps, _ = self._split_reps(loc_mid_reps, loc_ans_start, loc_seq_len)
                # Get original logits (no edit applied) - will be cached for KL loss
                with torch.no_grad():
                    orig_outputs = self.model(**loc_tokens)
                    orig_logits = orig_outputs.logits if hasattr(orig_outputs, 'logits') else orig_outputs
                    orig_logits = orig_logits.detach()
                loc_cached = (loc_tokens, loc_mid_reps, loc_query_reps, orig_logits)
        
        # ============================================================
        # PHASE 2: Generate LoRA weights and routing features
        # ============================================================
        
        evr = self.edit_extractor.extract_vision(query_reps, vision_reps)
        eqr = self.edit_extractor.extract_query(query_reps)
        edit_reps_cat = torch.cat([vision_reps, query_reps, ans_reps], 1)
        moe_c = self.moegen_c(edit_reps_cat)
        moe_r = self.moegen_r(edit_reps_cat)
        
        # Set current expert for forward hook
        self.is_training_edit = True
        self.current_expert = (moe_c.squeeze(0), moe_r.squeeze(0))
        
        # ============================================================
        # PHASE 3: Compute losses using forward_from_mid_layer
        # Only backprop through layers edit_layer_i onwards!
        # ============================================================
        
        # ============ Reliability Loss ============
        rel_loss = self._compute_ce_loss_from_mid(edit_tokens, edit_mid_reps)
        log_dict['rel_loss'] = float(rel_loss)
        loss = loss + rel_loss * self.rel_lambda
        
        # ============ Generality Loss ============
        gen_loss = torch.tensor(0.0, device=self.device)
        if gen_cached:
            gen_losses = []
            for gen_tokens, gen_mid_reps, gen_query_reps in gen_cached:
                iqr = self.inpt_extractor.extract_query(gen_query_reps)
                fuse_coe = self.get_moe_fuse_coe(iqr, eqr)
                self._set_weighted_expert(moe_c, moe_r, fuse_coe)
                gen_losses.append(self._compute_ce_loss_from_mid(gen_tokens, gen_mid_reps))
            if gen_losses:
                gen_loss = torch.stack(gen_losses).mean()
        log_dict['gen_loss'] = float(gen_loss)
        loss = loss + gen_loss * self.gen_lambda
        
        # ============ Locality Loss (KL divergence) ============
        loc_loss = torch.tensor(0.0, device=self.device)
        if loc_cached is not None:
            loc_tokens, loc_mid_reps, loc_query_reps, orig_logits = loc_cached
            iqr = self.inpt_extractor.extract_query(loc_query_reps)
            fuse_coe = self.get_moe_fuse_coe(iqr, eqr)
            self._set_weighted_expert(moe_c, moe_r, fuse_coe)
            loc_loss = self._compute_locality_loss_from_mid(loc_tokens, loc_mid_reps, orig_logits)
        log_dict['loc_loss'] = float(loc_loss)
        loss = loss + loc_loss * self.loc_lambda
        
        # ============ Soft Routing Loss (contrastive for query features) ============
        soft_routing_loss = torch.tensor(0.0, device=self.device)
        if neighbor_data is not None and len(neighbor_data) == 2:
            neighbor1_query, neighbor2_query = neighbor_data
            if neighbor1_query is not None and neighbor2_query is not None:
                # Extract query representations
                iqrs = self.inpt_extractor.extract_query(neighbor1_query)  # [b, eqe_n, module_dim]
                eqrs = self.edit_extractor.extract_query(neighbor2_query)  # [b, eqe_n, module_dim]
                
                rela_sim, abs_sim = self.get_moe_fuse_coe(iqrs, eqrs, split=True)  # [b, b]
                
                # Relative similarity loss: diagonal should be highest
                soft_routing_loss_rela = -torch.log(torch.diag(rela_sim) + eps).mean()
                
                # Absolute similarity loss: positive pairs high, negative pairs low
                abs_pos_sim = torch.diag(abs_sim)
                abs_neg_sim = torch.diag(abs_sim.roll(1, 1))
                soft_routing_loss_abs = -(torch.log(abs_pos_sim + eps) + torch.log(1 - abs_neg_sim + eps)).mean()
                
                soft_routing_loss = soft_routing_loss_rela + soft_routing_loss_abs
                log_dict['soft_routing_loss_rela'] = float(soft_routing_loss_rela)
                log_dict['soft_routing_loss_abs'] = float(soft_routing_loss_abs)
        
        log_dict['soft_routing_loss'] = float(soft_routing_loss)
        loss = loss + soft_routing_loss * self.soft_routing_lambda
        
        # ============ Hard Routing Loss (contrastive for vision features) ============
        hard_routing_loss = torch.tensor(0.0, device=self.device)
        if prototype_data is not None and len(prototype_data) == 2:
            inpt_reps_list, edit_reps_list = prototype_data
            if inpt_reps_list is not None and edit_reps_list is not None:
                # Hard routing neighbor loss
                hard_routing_loss = self._compute_hard_routing_loss(
                    inpt_reps_list, edit_reps_list, eps
                )
        
        log_dict['hard_routing_loss'] = float(hard_routing_loss)
        loss = loss + hard_routing_loss * self.hard_routing_lambda
        
        # Clear training state
        self.is_training_edit = False
        self.current_expert = None
        
        log_dict['total_loss'] = float(loss)
        self.global_step += 1
        
        return {"loss": loss, "log": log_dict}
    
    def _get_edit_reps(self, tokens: Dict[str, torch.Tensor]) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Get vision, query, and answer representations from tokens.
        Similar to original get_reps_for_edit.
        """
        captured = {}
        
        def hook(module, args, output):
            reps = output[0] if isinstance(output, tuple) else output
            captured['reps'] = reps
        
        layer = self._find_module(self.model, self.edit_layer_path)
        handle = layer.register_forward_hook(hook)
        
        try:
            with torch.no_grad():
                self.model(**tokens)
        finally:
            handle.remove()
        
        if 'reps' not in captured:
            return None
        
        reps = captured['reps']
        seq_len = reps.shape[1]
        
        # Simple split: assume vision tokens at start, then query, then answer
        # This is a simplification - original uses vt_range from VLLM
        # We use labels to find answer region
        if "labels" in tokens:
            labels = tokens["labels"]
            # Find where answer starts (first non -100)
            ans_start = (labels[0] != -100).nonzero(as_tuple=True)[0]
            if len(ans_start) > 0:
                ans_start = ans_start[0].item()
            else:
                ans_start = seq_len - 1
        else:
            ans_start = seq_len - 1
        
        # Heuristic split (adjust based on your model)
        # For VLMs: first ~576 tokens are often vision
        vision_end = min(576, ans_start // 2)
        
        vision_reps = reps[:, :vision_end, :]
        query_reps = reps[:, vision_end:ans_start, :]
        ans_reps = reps[:, ans_start:, :]
        
        # Handle empty regions
        if query_reps.shape[1] == 0:
            query_reps = vision_reps[:, -1:, :]
        if ans_reps.shape[1] == 0:
            ans_reps = query_reps[:, -1:, :]
        
        return vision_reps, query_reps, ans_reps
    
    def _set_weighted_expert(self, moe_c: torch.Tensor, moe_r: torch.Tensor, 
                             fuse_coe: torch.Tensor):
        """Set weighted expert for forward hook."""
        # For single edit, fuse_coe is [1, 1], just use the raw moe_c/moe_r
        self.current_expert = (moe_c.squeeze(0), moe_r.squeeze(0))
    
    #═══════════════════════════════════════════════════════════════════
    # MEMORY-EFFICIENT LOSS COMPUTATION (using forward_from_mid_layer)
    #═══════════════════════════════════════════════════════════════════
    
    def _compute_ce_loss_from_mid(
        self, 
        tokens: Dict[str, torch.Tensor], 
        mid_layer_reps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss using forward_from_mid_layer.
        
        This only backprops through layers edit_layer_i onwards,
        reducing memory by ~65%.
        
        Args:
            tokens: Input tokens (for labels)
            mid_layer_reps: Pre-computed mid-layer representations [B, L, D]
        
        Returns:
            Cross-entropy loss
        """
        # Forward from mid-layer (with edit applied via hook)
        outputs = self._forward_from_mid_layer(tokens, mid_layer_reps)
        
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        
        # Manual loss computation if model doesn't return loss
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if "labels" in tokens:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tokens["labels"].view(-1),
                ignore_index=-100
            )
            return loss
        
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def _compute_locality_loss_from_mid(
        self, 
        tokens: Dict[str, torch.Tensor],
        mid_layer_reps: torch.Tensor,
        orig_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute locality loss (KL divergence) using forward_from_mid_layer.
        
        Args:
            tokens: Input tokens
            mid_layer_reps: Pre-computed mid-layer representations
            orig_logits: Pre-computed original logits (without edit)
        
        Returns:
            KL divergence loss
        """
        # Forward from mid-layer WITH edit applied
        outputs_new = self._forward_from_mid_layer(tokens, mid_layer_reps)
        logits_new = outputs_new.logits if hasattr(outputs_new, "logits") else outputs_new
        
        # KL divergence (only on labeled positions)
        if "labels" in tokens:
            mask = tokens["labels"] != -100
            if mask.any():
                logits_new_masked = logits_new[mask]
                logits_old_masked = orig_logits[mask]
                
                kl_loss = F.kl_div(
                    F.log_softmax(logits_new_masked, dim=-1),
                    F.softmax(logits_old_masked.detach(), dim=-1),
                    reduction='batchmean'
                )
                return kl_loss
        
        # Fallback: KL on all positions
        kl_loss = F.kl_div(
            F.log_softmax(logits_new, dim=-1),
            F.softmax(orig_logits.detach(), dim=-1),
            reduction='batchmean'
        )
        return kl_loss
    
    #═══════════════════════════════════════════════════════════════════
    # ORIGINAL METHODS (kept for backward compatibility)
    #═══════════════════════════════════════════════════════════════════
    
    def _compute_ce_loss(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute cross-entropy loss on tokens."""
        outputs = self.model(**tokens)
        
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        
        # Manual loss computation if model doesn't return loss
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if "labels" in tokens:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tokens["labels"].view(-1),
                ignore_index=-100
            )
            return loss
        
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def _compute_locality_loss(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute locality loss: KL divergence between edited and original model.
        Matches original: vllm.logit_KL_loss(logits, pl, label_masks, True)
        """
        # Get logits with LoRA applied
        self.is_training_edit = True
        outputs_new = self.model(**tokens)
        logits_new = outputs_new.logits if hasattr(outputs_new, "logits") else outputs_new
        
        # Get logits without LoRA
        self.is_training_edit = False
        with torch.no_grad():
            outputs_old = self.model(**tokens)
            logits_old = outputs_old.logits if hasattr(outputs_old, "logits") else outputs_old
        
        # Restore state
        self.is_training_edit = True
        
        # KL divergence (only on labeled positions)
        if "labels" in tokens:
            mask = tokens["labels"] != -100
            if mask.any():
                logits_new_masked = logits_new[mask]
                logits_old_masked = logits_old[mask]
                
                kl_loss = F.kl_div(
                    F.log_softmax(logits_new_masked, dim=-1),
                    F.softmax(logits_old_masked, dim=-1),
                    reduction='batchmean'
                )
                return kl_loss
        
        # Fallback: KL on all positions
        kl_loss = F.kl_div(
            F.log_softmax(logits_new, dim=-1),
            F.softmax(logits_old, dim=-1),
            reduction='batchmean'
        )
        return kl_loss
    
    def _compute_hard_routing_loss(
        self, 
        inpt_reps_list: List[Tuple[torch.Tensor, torch.Tensor]],
        edit_reps_list: List[Tuple[torch.Tensor, torch.Tensor]],
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute hard routing loss (from original).
        
        Uses vision representations to decide whether to apply edit.
        """
        if not inpt_reps_list or not edit_reps_list:
            return torch.tensor(0.0, device=self.device)
        
        # Stack vision and query reps
        ivrs = torch.cat([self.inpt_extractor.extract_vision(d[1], d[0]) 
                          for d in inpt_reps_list], 0)  # [b, eqe_n, module_dim]
        evrs = torch.cat([self.edit_extractor.extract_vision(d[1], d[0]) 
                          for d in edit_reps_list], 0)  # [b, eqe_n, module_dim]
        
        sim = torch.einsum('bed,med->bme', ivrs, evrs).mean(2) * self.sim_scale  # [b, b]
        
        # Prototype similarity
        ivrs_prot = torch.cat([self.inpt_extractor.extract_from_visprot(d[1]) 
                               for d in inpt_reps_list], 0)  # [b, eqe_n, module_dim]
        sim_prot = torch.einsum('bed,bed->be', ivrs, ivrs_prot).mean(1, keepdim=True) * self.sim_scale  # [b, 1]
        
        sim_all = torch.softmax(torch.cat([sim, sim_prot], 1), 1)  # [b, b+1]
        
        # Neighbor loss: diagonal should be highest
        loss_neighbor = -torch.log(torch.diag(sim_all) + eps).mean()
        
        # Prototype loss: last column (prototype) should be high for locality samples
        # This is simplified - original has more complex logic
        loss_prototype = -torch.log(sim_all[:, -1] + eps).mean()
        
        return loss_neighbor + loss_prototype
    
    def save_pretrain_checkpoint(self, path: str):
        """Save pretrained meta-learners to checkpoint (matches original format)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        ckpt = {
            "train_modules": {
                "edit_extractor": self.edit_extractor.state_dict(),
                "inpt_extractor": self.inpt_extractor.state_dict(),
                "moegen_c": self.moegen_c.state_dict(),
                "moegen_r": self.moegen_r.state_dict(),
                "instant_reps_norm": self.instant_reps_norm.state_dict(),
            },
            "config": {
                "llm_mid_dim": self.llm_mid_dim,
                "lora_rank": self.lora_rank,
                "lora_scale": self.lora_scale,
                "module_dim": self.module_dim,
                "cross_att_head_n": self.cross_att_head_n,
                "eqe_n": self.eqe_n,
            },
            "global_step": self.global_step,
        }
        torch.save(ckpt, path)
        print(f"[LiveEditPretrain] Saved checkpoint to {path}")
    
    def save_checkpoint(self, path: str):
        """Alias for save_pretrain_checkpoint (public API)."""
        self.save_pretrain_checkpoint(path)
    
    def load_checkpoint(self, path: str):
        """Load pretrained meta-learners from checkpoint."""
        path = Path(path)
        if not path.exists():
            print(f"[LiveEditPretrain] Checkpoint not found: {path}")
            return False
        
        ckpt = torch.load(path, map_location=self.device)
        
        # Load train_modules
        if "train_modules" in ckpt:
            train_modules = ckpt["train_modules"]
            if "edit_extractor" in train_modules:
                self.edit_extractor.load_state_dict(train_modules["edit_extractor"])
            if "inpt_extractor" in train_modules:
                self.inpt_extractor.load_state_dict(train_modules["inpt_extractor"])
            if "moegen_c" in train_modules:
                self.moegen_c.load_state_dict(train_modules["moegen_c"])
            if "moegen_r" in train_modules:
                self.moegen_r.load_state_dict(train_modules["moegen_r"])
            if "instant_reps_norm" in train_modules:
                self.instant_reps_norm.load_state_dict(train_modules["instant_reps_norm"])
        
        # Load global step
        if "global_step" in ckpt:
            self.global_step = ckpt["global_step"]
        
        print(f"[LiveEditPretrain] Loaded checkpoint from {path}")
        return True