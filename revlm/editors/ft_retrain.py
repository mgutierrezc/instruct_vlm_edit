import torch
import copy
from .utils import brackets_to_periods, parent_module


class Finetune_retrain(torch.nn.Module):
    """
    Fine-tuning editor with periodic retraining on accumulated edit history.
    
    Unlike naive FT which just accumulates weight changes, FT_retrain periodically
    retrains from the original model on ALL accumulated edits, preventing drift.
    """
    def __init__(self, config, model):
        # Call Module init directly to avoid super() instance/subtype issues in notebooks
        torch.nn.Module.__init__(self)
        self.model = model.model if hasattr(model, 'model') else model
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        # Keep original pname for logging / compatibility
        self.pnames = [brackets_to_periods(config.inner_params[0])]
        self.device = config.device
        self.edit_lr = float(config.edit_lr)  # Ensure float type
        
        # Get editor-specific config
        editor_config = getattr(config, 'editor', config)
        self.retrain_memory = int(getattr(editor_config, 'retrain_memory', 100))
        self.retrain_frequency = int(getattr(editor_config, 'retrain_frequency', 50))
        
        # AMP configuration
        first_param = next(self.model.parameters(), None)
        model_dtype = getattr(first_param, 'dtype', torch.bfloat16)
        self.autocast_dtype = torch.float16 if model_dtype == torch.float16 else torch.bfloat16
        self.scaler = torch.amp.GradScaler('cuda') if self.autocast_dtype == torch.float16 else None

        # Resolve inner_params[0] to a module (finetune weight + bias together)
        layer_spec = config.inner_params[0]
        suffixes = [".weight", ".bias"]
        layer = layer_spec.rsplit(".", 1)[0] if any(layer_spec.endswith(s) for s in suffixes) else layer_spec
        self.layer_path = brackets_to_periods(layer)

        edit_module = parent_module(self.model, self.layer_path)
        layer_name = layer.rsplit(".", 1)[-1]
        self.layer_module = getattr(edit_module, layer_name)
        
        # Disable KV cache for training
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
        
        # Freeze all parameters except target module
        train_params = set(self.layer_module.parameters())
        for p in self.model.parameters():
            p.requires_grad = p in train_params
        if train_params:
            print(f"Finetuning module {layer} (with periodic retrain)")
        
        # Store original weights for retraining
        self.original_state = {
            name: param.clone().detach()
            for name, param in self.layer_module.named_parameters()
        }

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def reset_to_original(self):
        """Reset layer weights to original values before retraining."""
        with torch.no_grad():
            for name, param in self.layer_module.named_parameters():
                if name in self.original_state:
                    param.copy_(self.original_state[name])

    def retrain(self, config, batch_history):
        """Retrain from original weights on accumulated batch history."""
        if not batch_history:
            return self.model
        
        # Reset to original weights
        self.reset_to_original()
        
        self.model.train()
        params = list(self.layer_module.parameters())
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        
        n_iter = getattr(config, 'n_iter', 100)
        
        # Train on recent history (limited by retrain_memory)
        history_to_use = batch_history[-self.retrain_memory:]
        print(f"[ft_retrain] Retraining on {len(history_to_use)} batches from history")
        
        for tokens in history_to_use:
            for _ in range(n_iter):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                    outputs = self.model(**tokens)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = outputs.loss if hasattr(outputs, "loss") else None
                
                if loss is None:
                    if "labels" in tokens:
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            tokens["labels"].view(-1),
                            ignore_index=-100
                        )
                    else:
                        break
                
                # Early stopping if correct
                argmaxs = torch.argmax(logits, dim=-1)
                response_indices = (tokens.get('labels', torch.zeros_like(argmaxs)) != -100)
                if response_indices.any():
                    if torch.all(tokens['labels'][response_indices] == argmaxs[response_indices]).item():
                        break
                
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    opt.step()
        
        return self.model
        
    def edit(self, config, tokens, batch_history=None):
        """Single edit step (also accumulates to history externally)."""
        self.model.train()
        
        params = list(self.layer_module.parameters())
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        self.losses = []
        
        n_iter = getattr(config, 'n_iter', 100)
        
        for _ in range(n_iter):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                outputs = self.model(**tokens)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = outputs.loss if hasattr(outputs, "loss") else None
            
            if loss is None:
                if "labels" in tokens:
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        tokens["labels"].view(-1),
                        ignore_index=-100
                    )
                else:
                    break
            
            # Early stopping if correct
            argmaxs = torch.argmax(logits, dim=-1)
            response_indices = (tokens.get('labels', torch.zeros_like(argmaxs)) != -100)
            if response_indices.any():
                if torch.all(tokens['labels'][response_indices] == argmaxs[response_indices]).item():
                    break
            
            self.loss = loss
            self.losses.append(self.loss.detach().cpu().numpy())
            
            if self.scaler is not None:
                self.scaler.scale(self.loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
            else:
                self.loss.backward()
                opt.step()
        
        return self.model
