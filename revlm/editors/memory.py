import torch
from .utils import brackets_to_periods, parent_module
import transformers


class MemoryNetwork:
    """Memory Network editor with key-value storage"""
    def __init__(self, config, model):
        self.model = model.model if hasattr(model, 'model') else model
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        layer = config.inner_params[0]
        self.config = config
        self.device = config.device
        self.nkeys = getattr(config, 'nkeys', 10)
        
        for p in self.model.parameters():
            p.requires_grad = False
            
        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer
        
        # Determine transpose based on model type
        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True
            
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        original_layer = getattr(edit_module, layer_name)
        setattr(edit_module, layer_name, MemoryAdaptor(config, original_layer, transpose=transpose).to(self.device))
                
    def __call__(self, **kwargs):
        if self.config.task == "hallucination":
            key_id = (kwargs.get("labels", torch.tensor([])) == -100).sum() - 1
            setattr(eval(f"self.model.{self.layer}"), "key_id", key_id)
        return self.model(**kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def edit(self, config, tokens, batch_history):
        setattr(eval(f"self.model.{self.layer}"), "train", True)
        optimizer = torch.optim.Adam(self.model.parameters(), config.edit_lr)
        editor_config = getattr(config, 'editor', config)
        n_iter = getattr(editor_config, 'n_iter', config.n_iter)
        early_stop_patience = editor_config.early_stop_patience
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_iter))
        self.losses = []
        
        if config.task == "hallucination":
            key_id = (tokens.get("labels", torch.tensor([])) == -100).sum() - 1
            setattr(eval(f"self.model.{self.layer}"), "key_id", key_id)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for i in range(n_iter):
            outputs = self.model(**tokens)
            loss = outputs.loss if hasattr(outputs, "loss") else None
            
            if loss is None:
                break
            
            loss_value = loss.detach().cpu().item()
            self.losses.append(loss_value)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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
                print(f"[memory] iter {i+1}/{n_iter} - loss: {loss_value:.4f}")
        
        self.loss = loss if 'loss' in locals() else None
        setattr(eval(f"self.model.{self.layer}"), "train", False)
        return self.model


class MemoryAdaptor(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(MemoryAdaptor, self).__init__()
        
        self.model = layer
        self.device = layer.weight.device
        
        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]               
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        
        nkeys = getattr(config, 'nkeys', 10)
        self.fc = torch.nn.Linear(self.key_shape, nkeys).to(self.device)
        self.values = torch.nn.Parameter(torch.rand((nkeys, self.value_shape), requires_grad=True, device=self.device))
        self.key_id = -1
    
    def forward(self, *args):
        query = args[0][:, self.key_id, :]
        key_weights = torch.softmax(self.fc(query), 1)
        value = (key_weights.view(-1, 1) * self.values).sum(0)
        return value.unsqueeze(0)

