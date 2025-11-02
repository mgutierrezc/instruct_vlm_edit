import torch
import logging
import math
from PIL import Image
from .utils import *
from typing import Dict

LOG = logging.getLogger(__name__)


class VQAModel(torch.nn.Module):
    """Vision Question Answering model wrapper - works with all VLMs"""
    def __init__(self, config):
        super(VQAModel, self).__init__()
        self.config = config
        self.device = config.device
        self.temp = getattr(config.model, "temperature", 1.0)

        self.model = get_hf_model(config)
        self.model.eval()
        self.processor = get_processor(config)
        self.tokenizer = get_tokenizer(config)
        self.preprocess = get_preprocess(config)
        
    
    def forward(self, inputs):
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        output = self.model(**inputs)
        self.loss = getattr(output, "loss", None)
        return output.logits if hasattr(output, "logits") else output

    
    def encode(self, images, prompts, tokenize=False):
        images = [images] if isinstance(images, Image.Image) else images
        prompts = [prompts] if isinstance(prompts, str) else prompts
        inputs = self.preprocess(images, prompts, self.processor, tokenize=tokenize) # preprocess images and prompts into tensors (CPU), then move to device
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return inputs
        

    def generate(self, images, prompts, **kwargs):
        inputs = self.encode(images, prompts, tokenize=False)
        kwargs.setdefault("temperature", self.temp)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)
            outputs_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
            answers = [clean_answer(o, i) for (o, i) in zip(outputs_text, prompts)]
        return answers
        
    
    @torch.no_grad()
    def get_loss_y(self, image, prompt, label, inputs=None, add_special_tokens=False):
        # stay close to old behavior: no extra tokens by default
        if inputs is None:
            inputs = self.encode(image, prompt, tokenize=False)

        label_ids = self.tokenizer(
            label,
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        ).input_ids.to(self.device)

        is_enc_dec = bool(getattr(self.model.config, "is_encoder_decoder", False))

        if is_enc_dec:
            out = self.model(**inputs, labels=label_ids)
            avg_nll = float(out.loss.item())
            num_tokens = label_ids.size(1)
            return avg_nll, avg_nll * num_tokens, num_tokens

        # decoder-only: concat and mask
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask", None)

        full_ids = torch.cat([input_ids, label_ids], dim=1)
        if attn is not None:
            label_attn = torch.ones_like(label_ids, device=self.device)
            full_attn = torch.cat([attn, label_attn], dim=1)
        else:
            full_attn = None

        labels_mask = torch.full_like(full_ids, -100)
        prompt_len = input_ids.size(1)
        labels_mask[:, prompt_len:] = full_ids[:, prompt_len:]

        model_inputs = dict(inputs)
        model_inputs["input_ids"] = full_ids
        if full_attn is not None:
            model_inputs["attention_mask"] = full_attn
        model_inputs["labels"] = labels_mask

        out = self.model(**model_inputs)
        avg_nll = float(out.loss.item())
        num_tokens = full_ids.size(1) - prompt_len
        num_tokens = max(1, num_tokens)
        return avg_nll, avg_nll * num_tokens, num_tokens


    @torch.no_grad()
    def score_choices(self, images, prompts, label_words,
                    use_avg: bool = False,
                    temperature: float = 1.0):
        all_scores = []
        for img, pr, lbls in zip(images, prompts, label_words):
            inputs = self.encode(img, pr, tokenize=False)
            tmp = {}
            for lbl in lbls:
                avg_nll, sum_nll, ntok = self.get_loss_y(img, pr, lbl, inputs, add_special_tokens=False)
                tmp[lbl] = {
                    "avg_nll": avg_nll,
                    "sum_nll": sum_nll,
                    "num_tokens": ntok,
                }
            probs = nll_to_probs(tmp, use_avg=use_avg, temperature=temperature)
            for lbl, p in probs.items():
                tmp[lbl]["prob"] = float(p)

            all_scores.append(tmp)
        return all_scores


    # @torch.no_grad()
    # def letter_classifier(self, images, prompts, letters=("A", "B", "C", "D")):
    #     """Single-forward next-token classifier over letters.
    #     Formats prompts to elicit a single-letter answer, then classifies using next-token logits.
    #     Returns (pred_letters: List[str], probs: torch.Tensor[B, 4]).
    #     """
    #     prompts = [f"{p.strip()}\nAnswer with a single letter (A, B, C, or D) only." for p in prompts]
    #     inputs = self.encode(images, prompts, tokenize=False)

    #     is_enc_dec = bool(getattr(getattr(self.model, "config", object()), "is_encoder_decoder", False))
    #     if is_enc_dec:
    #         bos = self.tokenizer.bos_token_id or self.tokenizer.pad_token_id or 0
    #         dec_inp = torch.full((len(prompts), 1), bos, dtype=torch.long, device=self.device)
    #         outputs = self.model(**inputs, decoder_input_ids=dec_inp, use_cache=False, return_dict=True)
    #         next_logits = outputs.logits[:, -1, :]
    #     else:
    #         outputs = self.model(**inputs, return_dict=True)
    #         next_logits = outputs.logits[:, -1, :]

    #     letter_ids = [ids[0] for ids in self.tokenizer(list(letters), add_special_tokens=False).input_ids]
    #     letter_ids_t = torch.tensor(letter_ids, device=self.device)
    #     scores = next_logits.index_select(dim=1, index=letter_ids_t)
    #     probs = torch.softmax(scores, dim=1)
    #     pred_idx = probs.argmax(dim=1)
    #     preds = [letters[i] for i in pred_idx.tolist()]
    #     return preds, probs


    # def get_loss(self, batch: Dict):
    #     """Return differentiable loss tensor for a loader batch (for finetuning).
    #     Requires batch with 'images', 'prompts', and 'golds' (list of dicts with 'label').
    #     """
    #     images = batch.get("images")
    #     prompts = batch.get("prompts")
    #     prompt_inputs = self.encode(images, prompts, tokenize=False)

    #     # Collect gold answer texts (strict requirement)
    #     golds = batch["golds"]  # expect list of dicts
    #     gold_texts = [str(g["label"]) for g in golds]

    #     gold_tok = self.tokenizer(gold_texts, return_tensors="pt", add_special_tokens=False, padding=True)
    #     labels_ids = gold_tok.input_ids.to(self.device)
    #     if labels_ids.shape[1] == 0:
    #         # No target tokens → zero loss
    #         return torch.tensor(0.0, device=self.device, requires_grad=True)

    #     is_enc_dec = bool(getattr(getattr(self.model, "config", object()), "is_encoder_decoder", False))
    #     if is_enc_dec:
    #         out = self.model(**prompt_inputs, labels=labels_ids)
    #         return out.loss

    #     input_ids = prompt_inputs.get("input_ids")
    #     attn = prompt_inputs.get("attention_mask")
    #     if input_ids is None:
    #         out = self.model(**prompt_inputs, labels=labels_ids)
    #         return out.loss

    #     # Decoder-only: concatenate prompt + labels; mask prompt tokens
    #     full_ids = torch.cat([input_ids, labels_ids], dim=1)
    #     full_attn = torch.cat([attn, torch.ones_like(labels_ids)], dim=1) if attn is not None else None
    #     labels = torch.full_like(full_ids, -100)
    #     prompt_len = int(input_ids.shape[1])
    #     labels[:, prompt_len:] = full_ids[:, prompt_len:]

    #     model_inputs = dict(prompt_inputs)
    #     model_inputs["input_ids"] = full_ids
    #     if full_attn is not None:
    #         model_inputs["attention_mask"] = full_attn
    #     out = self.model(**model_inputs, labels=labels)
    #     return out.loss
