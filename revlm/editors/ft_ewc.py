import torch
from .utils import param_subset, brackets_to_periods


class Finetune_ewc(torch.nn.Module):
    """
    Fine-tuning with EWC (Elastic Weight Consolidation) regularization.
    """
    def __init__(self, config, model):
        super(Finetune_ewc, self).__init__()
        self.model = model.model if hasattr(model, 'model') else model
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        self.pnames = [brackets_to_periods(config.inner_params[0])]
        self.device = config.device
        # Get config values - editor configs may be nested under config.editor
        editor_config = getattr(config, 'editor', config) if hasattr(config, 'editor') else config
        self.ewc_lambda = float(getattr(editor_config, 'ewc_lambda', 1.0))
        self.fisher_mem = int(getattr(editor_config, 'fisher_mem', 10))
        self.edit_lr = float(config.edit_lr)  # Ensure float type (YAML may parse 1e-4 as string)
        
        for n, p in self.model.named_parameters():
            if n != self.pnames[0]:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def compute_fisher_matrix(self, batch_history):
        """Compute Fisher information matrix for EWC regularization"""
        optpar_dict = {}
        fisher_dict = {}
        model_dict = dict(self.model.named_parameters())
        
        for item_num, tokens in enumerate(batch_history[::-1]):
            if item_num < self.fisher_mem:
                outputs = self.model(**tokens)
                loss = outputs.loss if hasattr(outputs, "loss") else None
                if loss is None:
                    continue
                
                loss.backward()

                for name in self.pnames:
                    if name not in optpar_dict:
                        optpar_dict[name] = model_dict[name].data.clone()
                        fisher_dict[name] = model_dict[name].grad.data.clone().pow(2) if model_dict[name].grad is not None else torch.zeros_like(model_dict[name])
                    else:
                        optpar_dict[name] += model_dict[name].data.clone()
                        if model_dict[name].grad is not None:
                            fisher_dict[name] += model_dict[name].grad.data.clone().pow(2)
        
        for name in self.pnames:
            optpar_dict[name] /= min(self.fisher_mem, len(batch_history))
            fisher_dict[name] /= min(self.fisher_mem, len(batch_history))

        return fisher_dict, optpar_dict

    def edit(self, config, tokens, batch_history=None):
        params = param_subset(self.model.named_parameters(), self.pnames)
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        editor_config = getattr(config, 'editor', config)
        n_iter = getattr(editor_config, 'n_iter', config.n_iter)
        early_stop_patience = editor_config.early_stop_patience
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_iter))
        self.losses = []
        
        # Compute Fisher matrix if we have history, otherwise skip EWC regularization
        if batch_history:
            fisher_dict, optpar_dict = self.compute_fisher_matrix(batch_history)
        else:
            fisher_dict, optpar_dict = {}, {}
        
        best_loss = float('inf')
        patience_counter = 0
        
        for i in range(n_iter):
            self.model.zero_grad()
            outputs = self.model(**tokens)
            loss = outputs.loss if hasattr(outputs, "loss") else None
            
            if loss is None:
                if "labels" in tokens:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        tokens["labels"].view(-1), 
                        ignore_index=-100
                    )
                else:
                    break

            # Early stopping if prediction is correct
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            argmaxs = torch.argmax(logits, dim=-1)
            response_indices = (tokens.get('labels', torch.zeros_like(argmaxs)) != -100)
            if response_indices.any():
                if torch.all(tokens['labels'][response_indices] == argmaxs[response_indices]).item():
                    break

            # Add EWC regularization term
            for n, p in zip(self.pnames, params):
                if n in fisher_dict and n in optpar_dict:
                    ewc_regularizer = self.ewc_lambda * torch.sum(fisher_dict[n] * (p - optpar_dict[n]) ** 2)
                    loss += ewc_regularizer

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
                print(f"[ft_ewc] iter {i+1}/{n_iter} - loss: {loss_value:.4f}")
        
        return self.model

