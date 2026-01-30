# -*- coding: utf-8 -*-
"""
LiveEdit adapter for revlm - based on CVPR 2025 paper:
"Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts"

This implementation adapts LiveEdit to revlm's per-edit training paradigm:
- Trains moe_c and moe_r (LoRA weights) per edit
- Keeps original pool-based retrieval for lifelong editing
- Supports optional warm initialization from pre-trained meta-learners
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .liveedit_modules import QVExtractor, LowRankGenerator
from .utils import parent_module, brackets_to_periods


class LiveEdit(torch.nn.Module):
    """
    LiveEdit: Lifelong VLM Editor with Low-Rank Mixture-of-Experts.
    
    Adapted for revlm's per-edit training paradigm:
    - Each edit trains moe_c, moe_r (LoRA weights) to minimize loss
    - Expert pools grow for lifelong editing
    - Retrieval-based expert selection at inference
    """
    
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        
        # Store references (same pattern as GRACE/FT)
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
        self.device = config.device
        
        # Get editor config
        editor_cfg = getattr(config, "editor", config)
        
        # LiveEdit hyperparameters (from original config)
        self.edit_layer_i = int(getattr(editor_cfg, "edit_layer_i", 21))
        self.llm_mid_dim = int(getattr(editor_cfg, "llm_mid_dim", 4096))
        self.lora_rank = int(getattr(editor_cfg, "lora_rank", 4))
        self.lora_scale = float(getattr(editor_cfg, "lora_scale", 5.0))
        self.module_dim = int(getattr(editor_cfg, "module_dim", 1024))
        self.cross_att_head_n = int(getattr(editor_cfg, "cross_att_head_n", 8))
        self.eqe_n = int(getattr(editor_cfg, "eqe_n", 4))
        
        # Per-edit training hyperparameters
        self.n_iter = int(getattr(editor_cfg, "n_iter", getattr(config, "n_iter", 1000)))
        self.edit_lr = float(getattr(editor_cfg, "edit_lr", getattr(config, "edit_lr", 1e-3)))
        self.early_stop_patience = int(getattr(editor_cfg, "early_stop_patience", 50))
        
        # Retrain mode: train on all accumulated edits (like ft_retrain)
        self.retrain = bool(getattr(editor_cfg, "retrain", False))
        self.retrain_batch_size = int(getattr(editor_cfg, "retrain_batch_size", 4))  # Batch size for retrain mode
        self.edit_history = []  # Store past edit tokens for retrain mode
        
        # Retrieval hyperparameters
        self.eps = float(getattr(editor_cfg, "eps", 0.5))
        self.dist_fn = getattr(editor_cfg, "dist_fn", "cos")
        
        # Layer path template
        self.llm_layer_tmp = getattr(editor_cfg, "llm_layer_tmp", None) or self._detect_layer_template()
        if self.llm_layer_tmp is None:
            raise ValueError(
                "Could not auto-detect layer template. Please specify 'llm_layer_tmp' in config, "
                "e.g., 'model.layers.{}' or 'language_model.model.layers.{}'"
            )
        self.edit_layer_path = self.llm_layer_tmp.format(self.edit_layer_i)
        
        # Auto-detect hidden dimension if not specified
        if self.llm_mid_dim == 4096:  # Default value, try to auto-detect
            self.llm_mid_dim = self._detect_hidden_dim()
        
        # Scaling factor for similarity
        self.sim_scale = 1 / (self.module_dim ** 0.5)
        
        # Detect model dtype BEFORE initializing meta-learners
        self.model_dtype = self._detect_model_dtype()
        
        # Initialize meta-learners (for optional warm init)
        self._init_meta_learners()
        
        # Initialize expert pools
        self._init_pools()
        
        # Setup layer hook for inference
        self._setup_edit_hook()
        
        # Load checkpoint if provided
        ckpt_path = getattr(editor_cfg, "ckpt_path", None)
        if ckpt_path:
            self.load_checkpoint(ckpt_path)
        
        # Training state
        self.is_training_edit = False
        self.current_expert = None  # (moe_c, moe_r) during training
        self.losses = []
        self.loss = None
        
        # Disable KV cache for editing
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
    
    def _detect_model_dtype(self) -> torch.dtype:
        """Detect the model's dtype for LoRA weight initialization."""
        # Try to get dtype from model parameters
        for param in self.model.parameters():
            return param.dtype
        return torch.float32  # Default fallback
    
    def _detect_hidden_dim(self) -> int:
        """Auto-detect the model's hidden dimension."""
        # Try to get from model config
        if hasattr(self.model, 'config'):
            cfg = self.model.config
            # Different models store this differently
            for attr in ['hidden_size', 'd_model', 'text_config']:
                if hasattr(cfg, attr):
                    val = getattr(cfg, attr)
                    if isinstance(val, int):
                        return val
                    elif hasattr(val, 'hidden_size'):
                        return val.hidden_size
        
        # Try to infer from edit layer
        try:
            layer = self._get_layer_by_path(self.edit_layer_path)
            if layer is not None:
                # Check common attribute names
                for attr in ['hidden_size', 'embed_dim', 'in_features']:
                    if hasattr(layer, attr):
                        return getattr(layer, attr)
                # Check first linear layer
                for m in layer.modules():
                    if isinstance(m, nn.Linear):
                        return m.in_features
        except:
            pass
        
        # Default fallback
        return 4096
    
    def _get_layer_by_path(self, path: str):
        """Get a module by dot-separated path."""
        module = self.model
        for part in path.split('.'):
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module
    
    def _detect_layer_template(self) -> Optional[str]:
        """Auto-detect layer path template based on model architecture."""
        model_name = self.model.__class__.__name__.lower()
        
        # First, try to detect from inner_params if available
        if hasattr(self.config, 'model') and hasattr(self.config.model, 'inner_params'):
            inner_params = self.config.model.inner_params
            if inner_params and len(inner_params) > 0:
                # Extract layer template from inner_params like "language_model.model.layers.31.mlp.gate_proj.weight"
                param = inner_params[0]
                import re
                match = re.search(r'(.+\.layers)\.\d+', param)
                if match:
                    return match.group(1) + ".{}"
        
        # Then try model name matching
        if "qwen" in model_name:
            return "model.layers.{}"
        elif "llava" in model_name:
            return "language_model.model.layers.{}"
        elif "blip" in model_name or "instructblip" in model_name:
            # InstructBLIP with Vicuna backend uses language_model.model.layers
            return "language_model.model.layers.{}"
        
        # Default: try common patterns by inspecting model structure
        for n, _ in self.model.named_modules():
            if ".layers.0." in n:
                # Found a layers module, extract the prefix
                parts = n.split(".layers.0")[0]
                return parts + ".layers.{}"
            elif ".decoder.layers.0." in n:
                parts = n.split(".decoder.layers.0")[0]
                return parts + ".decoder.layers.{}"
        
        # Final fallback
        return "model.layers.{}"
    
    def _init_meta_learners(self):
        """Initialize meta-learners for warm initialization (optional)."""
        # These are used for warm initialization if checkpoint is loaded
        # Otherwise, experts are initialized randomly
        self.edit_extractor = QVExtractor(
            self.eqe_n, self.llm_mid_dim, self.module_dim, 
            self.cross_att_head_n, vision_tok_n=576, vis_prot=False
        ).to(self.device, dtype=self.model_dtype)
        
        self.inpt_extractor = QVExtractor(
            self.eqe_n, self.llm_mid_dim, self.module_dim,
            self.cross_att_head_n, vision_tok_n=576, vis_prot=True
        ).to(self.device, dtype=self.model_dtype)
        
        self.moegen_c = LowRankGenerator(
            self.llm_mid_dim, self.lora_rank, self.lora_scale,
            self.llm_mid_dim, self.module_dim, self.cross_att_head_n
        ).to(self.device, dtype=self.model_dtype)
        
        self.moegen_r = LowRankGenerator(
            self.llm_mid_dim, self.lora_rank, self.lora_scale,
            self.llm_mid_dim, self.module_dim, self.cross_att_head_n
        ).to(self.device, dtype=self.model_dtype)
        
        self.instant_reps_norm = nn.LayerNorm(self.llm_mid_dim).to(self.device, dtype=self.model_dtype)
        
        # Set to eval mode (meta-learners are frozen during per-edit training)
        self.edit_extractor.eval()
        self.inpt_extractor.eval()
        self.moegen_c.eval()
        self.moegen_r.eval()
        self.instant_reps_norm.eval()
        
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
        
        self.use_warm_init = False  # Set to True if checkpoint loaded
    
    def _init_pools(self):
        """Initialize expert pools for lifelong editing."""
        self.eqr_pool = torch.zeros(0, self.eqe_n, self.module_dim, device=self.device)
        self.evr_pool = torch.zeros(0, self.eqe_n, self.module_dim, device=self.device)
        self.moe_cs_pool = torch.zeros(0, self.lora_rank, self.llm_mid_dim, device=self.device)
        self.moe_rs_pool = torch.zeros(0, self.lora_rank, self.llm_mid_dim, device=self.device)
        self.key_pool = torch.zeros(0, self.llm_mid_dim, device=self.device)  # Simple keys
    
    def _setup_edit_hook(self):
        """Hook into edit layer to apply experts during inference."""
        
        def edit_hook(module, args, output):
            """Apply LoRA experts during forward pass."""
            # During training: apply current expert
            if self.is_training_edit and self.current_expert is not None:
                moe_c, moe_r = self.current_expert
                residual = self._compute_lora_residual(output, moe_c, moe_r)
                if isinstance(output, tuple):
                    return (output[0] + residual,) + output[1:]
                return output + residual
            
            # During inference: apply experts from pool
            if not self.is_training_edit and len(self.key_pool) > 0:
                reps = output[0] if isinstance(output, tuple) else output
                query = self._extract_key_from_reps(reps)
                moe_cs, moe_rs, weights = self._retrieve_experts(query)
                
                if moe_cs is not None:
                    residual = self._compute_weighted_residual(reps, moe_cs, moe_rs, weights)
                    if isinstance(output, tuple):
                        return (output[0] + residual,) + output[1:]
                    return output + residual
            
            return output
        
        # Register hook
        try:
            edit_layer = self._find_module(self.model, self.edit_layer_path)
            edit_layer._forward_hooks.clear()  # Remove any existing hooks
            edit_layer.register_forward_hook(edit_hook)
        except Exception as e:
            print(f"[LiveEdit] Could not hook {self.edit_layer_path}: {e}")
    
    def _find_module(self, model, path: str) -> nn.Module:
        """Find module by path."""
        parts = path.split('.')
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    #═══════════════════════════════════════════════════════════════════
    # REVLM EDITOR INTERFACE
    #═══════════════════════════════════════════════════════════════════
    
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def edit(self, config, tokens, batch_history=None, edit_ds=None, train_ds=None):
        """
        Main edit interface - trains LoRA expert per edit.
        
        Args:
            config: Configuration
            tokens: Tokenized batch from prepare_training_batch
            batch_history: History of previous batches (unused)
            edit_ds: Optional dataset (for IKE-style interface)
            train_ds: Optional training dataset (unused)
        
        Returns:
            self.model (weights unchanged, experts in pools)
        """
        self.model.train()
        
        if tokens is not None:
            # Single batch edit (like GRACE/FT)
            self._edit_one(tokens)
        elif edit_ds is not None:
            # Dataset edit (like IKE)
            for batch in getattr(edit_ds, 'loader', [edit_ds]):
                if hasattr(self.wrapper, 'prepare_training_batch'):
                    tokens = self.wrapper.prepare_training_batch(batch)
                else:
                    tokens = batch
                self._edit_one(tokens)
        
        self.model.eval()
        return self.model
    
    #═══════════════════════════════════════════════════════════════════
    # PER-EDIT TRAINING (Your Paradigm)
    #═══════════════════════════════════════════════════════════════════
    
    def _edit_one(self, tokens: Dict[str, torch.Tensor]):
        """
        Train a single LoRA expert on one edit example.
        
        Steps:
        1. Extract routing key from input
        2. Initialize moe_c, moe_r
        3. Training loop: optimize moe_c, moe_r (on all history if retrain=True)
        4. Add to pools
        5. Store tokens in history for retrain mode
        """
        # Step 1: Extract key for retrieval (at prompt-end position)
        with torch.no_grad():
            key = self._extract_key(tokens)
        
        # Step 2: Initialize expert
        moe_c, moe_r = self._init_expert(tokens)
        
        # Step 3: Training loop
        # If retrain=True, train on all accumulated edits; otherwise just current
        all_tokens = self.edit_history + [tokens] if self.retrain else [tokens]
        moe_c, moe_r = self._train_expert(all_tokens, moe_c, moe_r)
        
        # Step 4: Extract routing features and add to pools
        with torch.no_grad():
            # Get mid-layer representations for routing
            mid_reps = self._get_mid_layer_reps(tokens)
            if mid_reps is not None and self.use_warm_init:
                # Use meta-learners for routing features (cast to model dtype)
                mid_reps_cast = mid_reps.to(self.model_dtype)
                eqr = self.edit_extractor.extract_query(mid_reps_cast)
                evr = self.edit_extractor.extract_vision(mid_reps_cast, mid_reps_cast)
            else:
                # Simple: use key as routing feature
                eqr = key.unsqueeze(0).unsqueeze(0).expand(1, self.eqe_n, -1)
                eqr = eqr[:, :, :self.module_dim]  # Truncate to module_dim
                evr = eqr.clone()
        
        # Add to pools
        self._add_to_pool(key, eqr, evr, moe_c, moe_r)
        
        # Step 5: Store tokens in history for retrain mode
        if self.retrain:
            self.edit_history.append(tokens)
    
    def _extract_key(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract routing key from tokens using layer activation."""
        captured = {}
        
        def hook(module, args, output):
            reps = output[0] if isinstance(output, tuple) else output
            captured['reps'] = reps.detach()
        
        layer = self._find_module(self.model, self.edit_layer_path)
        handle = layer.register_forward_hook(hook)
        
        with torch.no_grad():
            self.model(**tokens)
        
        handle.remove()
        
        if 'reps' in captured:
            return self._extract_key_from_reps(captured['reps'])
        else:
            # Fallback: random key
            return torch.randn(self.llm_mid_dim, device=self.device)
    
    def _extract_key_from_reps(self, reps: torch.Tensor) -> torch.Tensor:
        """Extract key from layer representations."""
        if reps.dim() == 3:
            # Use mean pooling over sequence
            return reps.mean(dim=1).squeeze(0)
        return reps.mean(dim=0)
    
    def _get_mid_layer_reps(self, tokens: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Get mid-layer representations for routing feature extraction."""
        captured = {}
        
        def hook(module, args, output):
            reps = output[0] if isinstance(output, tuple) else output
            captured['reps'] = reps.detach()
        
        layer = self._find_module(self.model, self.edit_layer_path)
        handle = layer.register_forward_hook(hook)
        
        with torch.no_grad():
            self.model(**tokens)
        
        handle.remove()
        
        return captured.get('reps', None)
    
    def _init_expert(self, tokens: Dict[str, torch.Tensor]) -> Tuple[nn.Parameter, nn.Parameter]:
        """Initialize LoRA expert weights."""
        # Use float32 for training stability, will cast during forward
        dtype = torch.float32
        
        if self.use_warm_init:
            # Warm init from meta-learners
            with torch.no_grad():
                mid_reps = self._get_mid_layer_reps(tokens)
                if mid_reps is not None:
                    # Cast to meta-learner dtype (matches model dtype)
                    mid_reps_cast = mid_reps.to(self.model_dtype)
                    moe_c_init = self.moegen_c(mid_reps_cast).squeeze(0)
                    moe_r_init = self.moegen_r(mid_reps_cast).squeeze(0)
                else:
                    moe_c_init = torch.randn(self.lora_rank, self.llm_mid_dim, device=self.device, dtype=dtype) * 0.01
                    moe_r_init = torch.randn(self.lora_rank, self.llm_mid_dim, device=self.device, dtype=dtype) * 0.01
        else:
            # Random initialization (scaled)
            scale = 1 / (self.lora_scale * self.lora_rank ** 0.5)
            moe_c_init = torch.randn(self.lora_rank, self.llm_mid_dim, device=self.device, dtype=dtype) * scale
            moe_r_init = torch.randn(self.lora_rank, self.llm_mid_dim, device=self.device, dtype=dtype) * scale
        
        moe_c = nn.Parameter(moe_c_init.clone().to(dtype))
        moe_r = nn.Parameter(moe_r_init.clone().to(dtype))
        
        return moe_c, moe_r
    
    def _train_expert(self, tokens_list: List[Dict[str, torch.Tensor]], 
                      moe_c: nn.Parameter, moe_r: nn.Parameter) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training loop for LoRA expert with cosine LR decay and early stopping.
        
        Args:
            tokens_list: List of token batches to train on. If retrain=False, this is [current_tokens].
                        If retrain=True, this is [all_previous_tokens..., current_tokens].
            moe_c: LoRA down projection weights
            moe_r: LoRA up projection weights
        """
        n_edits = len(tokens_list)
        
        # Group edits into batches for efficiency (like ft_retrain)
        batch_size = self.retrain_batch_size if self.retrain else n_edits
        n_groups = (n_edits + batch_size - 1) // batch_size
        
        # Scheduler: one step per group per epoch
        total_steps = self.n_iter * n_groups
        opt = torch.optim.Adam([moe_c, moe_r], lr=self.edit_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))
        
        self.losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        # Minimum iterations before early stopping kicks in (like GRACE/Balancedit)
        min_iter_before_early_stop = 100
        
        # Set training state
        self.is_training_edit = True
        self.current_expert = (moe_c, moe_r)
        
        if self.retrain and n_edits > 1:
            print(f"[liveedit] retrain mode: {n_edits} edits | batch_size={batch_size} | groups={n_groups}")
        
        for epoch in range(self.n_iter):
            epoch_loss = 0.0
            
            # Iterate over groups (batched gradient accumulation)
            for group_idx in range(n_groups):
                start = group_idx * batch_size
                end = min(n_edits, start + batch_size)
                group = tokens_list[start:end]
                group_size = len(group)
                
                opt.zero_grad()
                group_loss = 0.0
                
                # Accumulate gradients within group
                for tokens in group:
                    outputs = self.model(**tokens)
                    loss = outputs.loss if hasattr(outputs, "loss") else None
                    
                    if loss is None:
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                        if "labels" in tokens:
                            loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                tokens["labels"].view(-1),
                                ignore_index=-100
                            )
                        else:
                            continue
                    
                    # Scale loss for gradient accumulation
                    scaled_loss = loss / group_size
                    scaled_loss.backward()
                    group_loss += loss.detach().cpu().item()
                
                # One optimizer step per group
                opt.step()
                scheduler.step()
                epoch_loss += group_loss / group_size
            
            avg_loss = epoch_loss / n_groups
            self.losses.append(avg_loss)
            
            # Early stopping (only after min_iter_before_early_stop iterations)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if epoch >= min_iter_before_early_stop and patience_counter >= self.early_stop_patience:
                    break
            
            # Print progress every 10 iterations
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.n_iter - 1:
                print(f"[liveedit] iter {epoch+1}/{self.n_iter} - loss: {avg_loss:.4f} - lr: {scheduler.get_last_lr()[0]:.2e}")
        
        self.loss = avg_loss if 'avg_loss' in locals() else None
        
        # Clear training state
        self.is_training_edit = False
        self.current_expert = None
        
        return moe_c.detach(), moe_r.detach()
    
    def _compute_lora_residual(self, output: torch.Tensor, moe_c: torch.Tensor, 
                                moe_r: torch.Tensor) -> torch.Tensor:
        """Compute LoRA residual: moe_r @ relu(moe_c @ x)."""
        reps = output[0] if isinstance(output, tuple) else output
        original_dtype = reps.dtype
        
        # Cast to model dtype for LayerNorm, then to float32 for LoRA computation
        reps_norm = self.instant_reps_norm(reps.to(self.model_dtype))
        
        # Cast everything to float32 for computation stability
        x = reps_norm.float()
        moe_c_f32 = moe_c.float()
        moe_r_f32 = moe_r.float()
        
        # LoRA: down -> relu -> up
        # x: [B, L, D], moe_c: [R, D], moe_r: [R, D]
        down = torch.einsum('bld,rd->blr', x, moe_c_f32)  # [B, L, R]
        down = F.relu(down)
        up = torch.einsum('blr,rd->bld', down, moe_r_f32)  # [B, L, D]
        
        # Cast back to original dtype
        return up.to(original_dtype)
    
    #═══════════════════════════════════════════════════════════════════
    # RETRIEVAL (From original LiveEdit)
    #═══════════════════════════════════════════════════════════════════
    
    def _add_to_pool(self, key: torch.Tensor, eqr: torch.Tensor, evr: torch.Tensor,
                     moe_c: torch.Tensor, moe_r: torch.Tensor):
        """Add trained expert to pools."""
        self.key_pool = torch.cat([self.key_pool, key.unsqueeze(0)], dim=0)
        self.eqr_pool = torch.cat([self.eqr_pool, eqr], dim=0)
        self.evr_pool = torch.cat([self.evr_pool, evr], dim=0)
        self.moe_cs_pool = torch.cat([self.moe_cs_pool, moe_c.unsqueeze(0)], dim=0)
        self.moe_rs_pool = torch.cat([self.moe_rs_pool, moe_r.unsqueeze(0)], dim=0)
    
    def _retrieve_experts(self, query: torch.Tensor) -> Tuple[Optional[torch.Tensor], 
                                                               Optional[torch.Tensor], 
                                                               Optional[torch.Tensor]]:
        """Retrieve relevant experts based on query similarity."""
        if len(self.key_pool) == 0:
            return None, None, None
        
        # Compute distances
        if self.dist_fn == "cos":
            query_norm = F.normalize(query.float().unsqueeze(0), dim=-1)
            key_norm = F.normalize(self.key_pool.float(), dim=-1)
            sims = torch.mm(query_norm, key_norm.T).squeeze(0)
            dists = 1 - sims
        else:  # euclidean
            dists = torch.cdist(query.float().unsqueeze(0), self.key_pool.float()).squeeze(0)
        
        # Find matches within epsilon threshold
        mask = dists < self.eps
        
        if not mask.any():
            return None, None, None
        
        # Get matching experts
        moe_cs = self.moe_cs_pool[mask]
        moe_rs = self.moe_rs_pool[mask]
        
        # Compute fusion weights: softmax(sim) * sigmoid(sim) (original LiveEdit formula)
        # Note: For cosine similarity (already in [0,1]), use temperature scaling
        # to map high similarity to high sigmoid outputs
        if self.dist_fn == "cos":
            # Temperature scaling: sigmoid(T*(sim-0.5)) maps sim=0.9+ to ~0.95+
            temperature = 10.0  # High temp makes sigmoid steeper
            sim_for_softmax = sims[mask] * temperature
            sim_for_sigmoid = (sims[mask] - 0.5) * temperature  # Center at 0.5
        else:
            sim_for_softmax = -dists[mask] * self.sim_scale
            sim_for_sigmoid = sim_for_softmax
        
        rela_sim = F.softmax(sim_for_softmax, dim=0)  # relative similarity
        abs_sim = torch.sigmoid(sim_for_sigmoid)       # absolute similarity (0.9 sim -> ~0.98)
        weights = rela_sim * abs_sim              # combined fusion coefficient
        
        return moe_cs, moe_rs, weights
    
    def _compute_weighted_residual(self, reps: torch.Tensor, moe_cs: torch.Tensor,
                                    moe_rs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted LoRA residual from multiple experts."""
        original_dtype = reps.dtype
        
        # Cast to model dtype for LayerNorm, then to float32 for computation
        reps_norm = self.instant_reps_norm(reps.to(self.model_dtype))
        
        # Cast to float32 for computation stability
        x = reps_norm.float()
        moe_cs_f32 = moe_cs.float()
        moe_rs_f32 = moe_rs.float()
        weights_f32 = weights.float()
        
        # x: [B, L, D], moe_cs: [M, R, D], moe_rs: [M, R, D], weights: [M]
        # Compute each expert's contribution and weight
        down = torch.einsum('bld,mrd->blmr', x, moe_cs_f32)  # [B, L, M, R]
        down = F.relu(down)
        up = torch.einsum('blmr,mrd,m->bld', down, moe_rs_f32, weights_f32)  # [B, L, D]
        
        # Cast back to original dtype
        return up.to(original_dtype)
    
    #═══════════════════════════════════════════════════════════════════
    # POOL MANAGEMENT & UTILITIES
    #═══════════════════════════════════════════════════════════════════
    
    def restore_to_original_model(self):
        """Clear pools and history - reset to unedited state."""
        self._init_pools()
        self.edit_history = []  # Clear retrain history
    
    def print_stats(self):
        """Print edit application statistics."""
        print(f"[LiveEdit] Pool: {len(self.key_pool)}")
    
    def load_checkpoint(self, ckpt_path: str):
        """Load pre-trained meta-learners from checkpoint."""
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            
            if 'train_modules' in ckpt:
                # LiveEdit checkpoint format
                modules = ckpt['train_modules']
                if 'edit_extractor' in modules:
                    self.edit_extractor.load_state_dict(modules['edit_extractor'])
                if 'inpt_extractor' in modules:
                    self.inpt_extractor.load_state_dict(modules['inpt_extractor'])
                if 'moegen_c' in modules:
                    self.moegen_c.load_state_dict(modules['moegen_c'])
                if 'moegen_r' in modules:
                    self.moegen_r.load_state_dict(modules['moegen_r'])
                if 'instant_reps_norm' in modules:
                    self.instant_reps_norm.load_state_dict(modules['instant_reps_norm'])
            
            self.use_warm_init = True
            print(f"[LiveEdit] Loaded checkpoint from {ckpt_path}")
            print(f"[LiveEdit] Warm initialization enabled")
            
        except Exception as e:
            print(f"[LiveEdit] Warning: Could not load checkpoint: {e}")
            print(f"[LiveEdit] Using random initialization")
            self.use_warm_init = False
    
    def get_modules_for_training(self) -> Dict[str, nn.Module]:
        """Get meta-learners (for saving/loading)."""
        return {
            'edit_extractor': self.edit_extractor,
            'inpt_extractor': self.inpt_extractor,
            'moegen_c': self.moegen_c,
            'moegen_r': self.moegen_r,
            'instant_reps_norm': self.instant_reps_norm
        }
