import torch
from .utils import param_subset, brackets_to_periods


class Finetune(torch.nn.Module):
    """
    Fine-tuning editor - directly finetunes chosen weights given new inputs.
    """
    def __init__(self, config, model):
        super(Finetune, self).__init__()
        self.model = model.model if hasattr(model, 'model') else model
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        self.pnames = [brackets_to_periods(config.inner_params[0])]
        self.device = config.device
        self.edit_lr = float(config.edit_lr)  # Ensure float type (YAML may parse 1e-4 as string)
        
        # Freeze all parameters except the ones to edit
        for n, p in self.model.named_parameters():
            if n != self.pnames[0]:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def edit(self, config, tokens, batch_history):
        params = param_subset(self.model.named_parameters(), self.pnames)
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        self.losses = []
        
        n_iter = config.n_iter
        
        for _ in range(n_iter):
            self.model.zero_grad()
            outputs = self.model(**tokens)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = outputs.loss if hasattr(outputs, "loss") else None
            
            if loss is None:
                # Compute loss manually if not provided
                if "labels" in tokens:
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        tokens["labels"].view(-1), 
                        ignore_index=-100
                    )
                else:
                    break
            
            # Early stopping if prediction is correct
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

