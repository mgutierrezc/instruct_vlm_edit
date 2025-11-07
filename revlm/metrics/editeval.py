"""Editing Evaluation sMetrics (dataset-based, batch generation).

Quick usage
- reliability = ee_reliability(vlm, edit_ds)
- text_gen = compute_text_generality(vlm, edit_ds, related_texts)
- image_gen = compute_image_generality(vlm, edit_ds, related_images)
- locality = compute_locality(vlm_base, vlm_new, unrelated_ds)
- scores = combined_score(vlm_base, vlm_new, edit_ds, related_texts, related_images, unrelated_ds)
"""

from typing import Any, Dict, List, Mapping, Sequence, Tuple
from .utils.helper import generation  # uses dataset.task_generate under the hood


def ee_reliability(model_new: Any, edit_ds: Any) -> float:
    """Compute reliability via task-based generation on the dataset.

    Args
    - model_new: VQAModel 
    - edit_ds: VQADataset
    """
    pairs = generation(model_new, edit_ds)
    if not pairs:
        return 0.0
    correct = sum(1 for t, p in pairs if p == t)
    return correct / len(pairs)


def ee_locality(
    model_old: Any,
    model_new: Any,
    unrelated_ds: Any,
) -> float:
    """Agreement between base and new models on unrelated dataset inputs.

    Uses batch generation on (image, prompt) pairs from unrelated_ds.
    """

    pairs_old = generation(model_old, unrelated_ds)
    pairs_new = generation(model_new, unrelated_ds)
    preds_old = [p for _, p in pairs_old]
    preds_new = [p for _, p in pairs_new]
    correct = sum(1 for a, b in zip(preds_old, preds_new) if a == b)
    return correct / len(preds_old)


def compute_text_generality(
    model_new: Any,
    edit_ds: Any,
    related_texts: Mapping[int, Sequence[str]],
) -> float:
    """Accuracy on paraphrased/related texts using the same images.

    related_texts: {idx:[text1, text2, ...]} aligned to edit_ds.data indices.
    """
    newdata = []

    for id, ex in enumerate(edit_ds.data):
        rtexts = related_texts.get(ex['idx'], [])
        if not rtexts:
            continue
        for rid, t in enumerate(rtexts):
            # make a copy entrance of ex, that is another instance of the same example, replace the prompt with the related text
            # make it a new id of ex['idx']+"_"+str(rid)
            images.append(ex['image'])
            texts.append(t)
            targets.append(ex['target'])
            rt_id.append(ex['idx']+"_"+str(rid))

    if not images:
        return 0.0

    preds = model_new.generate(images, texts)
    correct = sum(1 for p, y in zip(preds, targets) if p == y)
    return correct / len(images)


def compute_image_generality(
    model_new: Any,
    edit_ds: Any,
    related_images: Mapping[int, Sequence[Any]],
) -> float:
    """Accuracy on related images using the same prompts.

    related_images: {idx -> [img1, img2, ...]} aligned to edit_ds.data indices.
    """
    images: List[Any] = []
    texts: List[str] = []
    targets: List[str] = []

    data = getattr(edit_ds, "data", [])
    for idx, ex in enumerate(data):
        rimgs = related_images.get(idx, [])
        if not rimgs:
            continue
        prompt = ex.get("prompt", "")
        gold = ex.get("gold", {})
        target = str(gold.get("label", ""))
        for img in rimgs:
            images.append(img)
            texts.append(prompt)
            targets.append(target)

    if not images:
        return 0.0

    preds = model_new.generate(images, texts)
    correct = sum(1 for p, y in zip(preds, targets) if p == y)
    return correct / len(images)


def combined_score(
    model_base: Any,
    model_new: Any,
    edit_ds: Any,
    related_texts: Mapping[int, Sequence[str]],
    related_images: Mapping[int, Sequence[Any]],
    unrelated_ds: Any,
    lambda_gen: float = 1.0,
    lambda_loc: float = 1.0,
    gen_agg: str = "mean",
) -> Dict[str, float]:
    """Combined metric: rel + λ_gen * gen + λ_loc * loc.

    gen can be mean or harmonic of text/image generality.
    """
    rel = compute_reliability(model_new, edit_ds)
    tgen = compute_text_generality(model_new, edit_ds, related_texts)
    igen = compute_image_generality(model_new, edit_ds, related_images)

    if gen_agg == "harmonic":
        gen = 0.0 if (tgen == 0 or igen == 0) else 2.0 / (1.0 / tgen + 1.0 / igen)
    else:
        gen = 0.5 * (tgen + igen)

    loc = compute_locality(model_base, model_new, unrelated_ds)
    score = rel + lambda_gen * gen + lambda_loc * loc

    return {
        "reliability": float(rel),
        "text_generality": float(tgen),
        "image_generality": float(igen),
        "locality": float(loc),
        "combined": float(score),
    }


