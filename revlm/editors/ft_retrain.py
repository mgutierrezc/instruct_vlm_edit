import torch
from .utils import param_subset, brackets_to_periods


class Finetune_retrain(torch.nn.Module):
    """
    Fine-tuning with periodic retraining on history.
    """
    def __init__(self, config, model):
        super(Finetune_retrain, self).__init__()
        self.model = model.model if hasattr(model, 'model') else model
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        self.pnames = [brackets_to_periods(config.inner_params[0])]
        self.device = config.device
        self.edit_lr = float(config.edit_lr)  # Ensure float type (YAML may parse 1e-4 as string)
        # Get config values - editor configs may be nested under config.editor
        editor_config = getattr(config, 'editor', config) if hasattr(config, 'editor') else config
        self.retrain_memory = int(getattr(editor_config, 'retrain_memory', 100))

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def retrain(self, init_model, config, batch_history):
        """Retrain on batch history"""
        model = init_model.model if hasattr(init_model, 'model') else init_model
        params = param_subset(model.named_parameters(), self.pnames)
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        
        n_iter = config.n_iter
        
        for tokens in batch_history[-self.retrain_memory:]:  # Only use recent history
            for _ in range(n_iter):
                model.zero_grad()
                outputs = model(**tokens)
                loss = outputs.loss if hasattr(outputs, "loss") else None
                
                if loss is None and "labels" in tokens:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        tokens["labels"].view(-1), 
                        ignore_index=-100
                    )
                
                if loss is None:
                    break
                
                loss.backward()
                opt.step()
                opt.zero_grad()

        if hasattr(init_model, 'model'):
            init_model.model = model
        return init_model
        
    def edit(self, config, tokens, batch_history):
        params = param_subset(self.model.named_parameters(), self.pnames)
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        self.losses = []
        
        n_iter = getattr(config, 'n_iter', 100)
        
        for _ in range(n_iter):
            self.model.zero_grad()
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
            
            argmaxs = torch.argmax(logits, dim=-1)
            response_indices = (tokens.get('labels', torch.zeros_like(argmaxs)) != -100)
            if response_indices.any():
                if torch.all(tokens['labels'][response_indices] == argmaxs[response_indices]).item():
                    break
            
            self.loss = loss
            self.losses.append(self.loss.detach().cpu().numpy())
            self.loss.backward()
            opt.step()
            opt.zero_grad()
        
        return self.model

