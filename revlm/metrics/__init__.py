# from .mcq import *
# from .qa import *





# def Accuracy(model, tokens):
#     """Accuracy metric for classification tasks"""
#     labels = tokens["labels"]
#     new_tokens = {k: v for k, v in tokens.items() if k != "labels"}
    
#     with torch.no_grad():
#         outputs = model(**new_tokens)
#         logits = outputs.logits if hasattr(outputs, "logits") else outputs
#         probs = torch.softmax(logits, -1).squeeze()
#         argmaxs = torch.argmax(probs, dim=-1).squeeze()
        
#         if labels.dim() == 0:
#             return (labels == argmaxs).float()
#         return (labels == argmaxs).float().mean()


# def F1(model, batch):
#     """F1 metric for VQA/generation tasks - token overlap"""
#     input_ids = batch["input_ids"]
#     preds = model.generate(input_ids, max_length=50).squeeze()
    
#     if len(preds) > 1 and hasattr(model, "tokenizer"):
#         preds = preds[preds != model.tokenizer.pad_token_id]
    
#     labels = batch["labels"]
#     gold_toks = labels[labels != -100].cpu().squeeze()
    
#     if gold_toks.numel() == 0:
#         return 0.0
    
#     preds_np = preds.cpu().numpy() if torch.is_tensor(preds) else np.array(preds)
#     gold_np = gold_toks.numpy() if torch.is_tensor(gold_toks) else np.array(gold_toks)
    
#     num_same = len(np.intersect1d(preds_np, gold_np))
    
#     if num_same == 0 or preds_np.size == 0:
#         return 0.0
#     if gold_np.size == 1 and preds_np.size == 1 and (gold_np == preds_np).all():
#         return 1.0
#     precision = num_same / preds_np.size
#     recall = num_same / gold_np.size
#     f1 = (2 * precision * recall) / (precision + recall + 1e-8)
#     return float(f1)


# def is_acc_error(model, tokens):
#     """Check if model prediction is incorrect (for classification)"""
#     labels = tokens.get("labels")
#     with torch.no_grad():
#         new_tokens = {k: v for k, v in tokens.items() if k != "labels"}
#         outputs = model(**new_tokens)
#         logits = outputs.logits if hasattr(outputs, "logits") else outputs
#         probs = torch.softmax(logits, -1).squeeze()
#         argmaxs = torch.argmax(probs, dim=-1).squeeze()
#         return labels != argmaxs if labels.dim() == 0 else (labels != argmaxs).any()


# def is_qa_error(model, tokens):
#     """Check if model prediction is incorrect (for generation/VQA)"""
#     f1_score = F1(model, tokens)
#     return f1_score < 1.0


# def get_metric(task):
#     """Get metric function for given task"""
#     if task == "vqa":
#         return F1
#     elif task == "captioning":
#         return F1
#     else:
#         # Default to accuracy
#         return Accuracy


# def get_error_fn(task):
#     """Get error checking function for given task"""
#     if task in ["vqa", "captioning"]:
#         return is_qa_error
#     else:
#         return is_acc_error

