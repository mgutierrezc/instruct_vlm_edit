import re
import math
import string
from typing import List, Tuple

import numpy as np
import torch

def _accuracy(preds, gts):
    ok = 0
    for p, g in zip(preds, gts):
        if p.strip().lower() == g.strip().lower():
            ok += 1
    return ok / len(gts)




def _normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace (SQuAD-style)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    s = str(s or "")
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score_str(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return float(2 * precision * recall / (precision + recall))


def _exact_match_str(prediction: str, ground_truth: str) -> bool:
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def QA_metrics_text(model, vlmdataset):
    """Compute QA text metrics over a dataset loader.

    Returns: {"exact_match": em, "f1": f1, "contains": contains_acc, "n": n}
    - exact_match: normalized string equality
    - f1: token-level F1 on normalized text (SQuAD-style)
    - contains: case-insensitive containment of gold text in output (after normalization)
    """
    loader = vlmdataset.loader

    total = 0
    em_sum = 0.0
    f1_sum = 0.0
    contains_sum = 0.0

    for batch in loader:
        images = batch.get("images")
        prompts = batch.get("prompts")
        golds = batch.get("label", [])
        outputs = model.generate(images, prompts, max_new_tokens=64, temperature=getattr(model, "temp", 1.0))

        for i, pred in enumerate(outputs):
            gold = str(golds[i]) if i < len(golds) and golds[i] is not None else ""
            if not gold:
                continue
            total += 1
            pred_n = _normalize_answer(pred)
            gold_n = _normalize_answer(gold)
            em_sum += 1.0 if pred_n == gold_n else 0.0
            f1_sum += _f1_score_str(pred, gold)
            contains_sum += 1.0 if gold_n and gold_n in pred_n else 0.0

    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0, "contains": 0.0, "n": 0}

    return {
        "exact_match": float(em_sum / total),
        "f1": float(f1_sum / total),
        "contains": float(contains_sum / total),
        "n": int(total),
    }


@torch.no_grad()
def QA_metrics_loss(model, vlmdataset):
    """Compute average token NLL and perplexity of gold answers given prompts/images.

    Works for both encoder-decoder and decoder-only models by masking loss to answer tokens only.

    Returns: {"avg_nll": ..., "ppl": ..., "sum_nll": ..., "num_tokens": ..., "n": ...}
    """
    loader = vlmdataset.loader

    total_tokens = 0
    total_sum_nll = 0.0
    num_examples = 0

    # Model attributes
    tokenizer = getattr(model, "tokenizer", None)
    is_enc_dec = bool(getattr(getattr(model.model, "config", object()), "is_encoder_decoder", False))

    for batch in loader:
        images = batch.get("images")
        prompts = batch.get("prompts")
        golds = batch.get("label", [])

        # encode prompts/images into model-ready inputs
        prompt_inputs = model.encode(images, prompts, tokenize=False)

        # Tokenize gold answers as a batch without special tokens
        gold_texts: List[str] = [str(x) if x is not None else "" for x in golds]
        if tokenizer is None:
            continue
        gold_tok = tokenizer(gold_texts, return_tensors="pt", add_special_tokens=False, padding=True)
        cand_ids = gold_tok.input_ids.to(model.device)
        ans_len = cand_ids.shape[1]
        if ans_len == 0:
            continue

        from ..models.utils import compute_loss_stats
        avg_nll, sum_nll_b, num_tokens_b = compute_loss_stats(
            model, prompt_inputs, cand_ids, mask_prompt=True
        )

        total_tokens += int(num_tokens_b)
        total_sum_nll += float(sum_nll_b)
        num_examples += int(cand_ids.shape[0])

    if total_tokens == 0:
        return {"avg_nll": float("inf"), "ppl": float("inf"), "sum_nll": 0.0, "num_tokens": 0, "n": 0}

    avg_nll = float(total_sum_nll / total_tokens)
    ppl = float(math.exp(avg_nll))
    return {
        "avg_nll": avg_nll,
        "ppl": ppl,
        "sum_nll": float(total_sum_nll),
        "num_tokens": int(total_tokens),
        "n": int(num_examples),
    }


def QA_metrics_nli_bi(model, vlmdataset, nli_pipeline=None):
    """Bidirectional NLI between generated answer and gold label text.

    If a transformers NLI pipeline (e.g., roberta-large-mnli) is provided or importable,
    we use its entailment score. Otherwise, we fall back to a simple normalized substring
    heuristic as a proxy for entailment.

    Returns a dict with average entailment scores in both directions and the fraction
    of examples where both directions are entailed (bi-directional):
    {"pred_to_gold": ..., "gold_to_pred": ..., "bi_frac": ..., "n": ...}
    """
    loader = vlmdataset.loader

    # Try to create a default NLI pipeline if not supplied
    clf = nli_pipeline
    if clf is None:
        try:
            from transformers import pipeline  # type: ignore
            clf = pipeline("text-classification", model="roberta-large-mnli", return_all_scores=True)
        except Exception:
            clf = None

    def entail_score(prem: str, hyp: str) -> float:
        if not prem or not hyp:
            return 0.0
        if clf is None:
            # Heuristic fallback: normalized containment as proxy for entailment
            prem_n = _normalize_answer(prem)
            hyp_n = _normalize_answer(hyp)
            return 1.0 if hyp_n and hyp_n in prem_n else 0.0
        try:
            out = clf({"text": prem, "text_pair": hyp})
            # Normalize pipeline outputs to a list of dicts
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                scores_list = out
            elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
                scores_list = out[0]
            else:
                return 0.0
            ent = next((d.get("score", 0.0) for d in scores_list if str(d.get("label", "")).lower().startswith("entail")), 0.0)
            return float(ent)
        except Exception:
            return 0.0

    total = 0
    fwd_sum = 0.0  # pred -> gold
    bwd_sum = 0.0  # gold -> pred
    bi_cnt = 0

    for batch in loader:
        images = batch.get("images")
        prompts = batch.get("prompts")
        golds = batch.get("label", [])
        preds = model.generate(images, prompts, max_new_tokens=64)

        for i, pred in enumerate(preds):
            gold = str(golds[i]) if i < len(golds) and golds[i] is not None else ""
            if not gold:
                continue
            total += 1
            s_fwd = entail_score(pred, gold)
            s_bwd = entail_score(gold, pred)
            fwd_sum += s_fwd
            bwd_sum += s_bwd
            if s_fwd >= 0.5 and s_bwd >= 0.5:
                bi_cnt += 1

    if total == 0:
        return {"pred_to_gold": 0.0, "gold_to_pred": 0.0, "bi_frac": 0.0, "n": 0}

    return {
        "pred_to_gold": float(fwd_sum / total),
        "gold_to_pred": float(bwd_sum / total),
        "bi_frac": float(bi_cnt / total),
        "n": int(total),
    }

