"""Evaluation metrics for model editing: reliability, generality, locality.

All functions are framework-agnostic and operate on plain Python types.
Models are simple callables: model(image, text) -> str.
Edit items: {"image": Any, "text": str, "target": str}.
Related mappings keyed by edit index. Unrelated items: {"image", "text"}.
"""

from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence


EditItem = MutableMapping[str, Any]
UnrelatedItem = MutableMapping[str, Any]
ModelCallable = Callable[[Any, str], str]


def compute_reliability(model_new: ModelCallable, edit_set: Sequence[EditItem]) -> float:
    """Fraction of edits where the edited model matches the target.

    Example
    -------
    >>> def new_model(img, txt):
    ...     return {"q1": "lenovo", "q2": "dog"}[txt]
    >>> edits = [
    ...     {"image": object(), "text": "q1", "target": "lenovo"},
    ...     {"image": object(), "text": "q2", "target": "cat"},
    ... ]
    >>> round(compute_reliability(new_model, edits), 3)
    0.5
    """
    correct = 0
    total = len(edit_set)
    for e in edit_set:
        y_pred = model_new(e["image"], e["text"])
        if y_pred == e["target"]:
            correct += 1
    return correct / max(total, 1)


def compute_text_generality(
    model_new: ModelCallable,
    edit_set: Sequence[EditItem],
    related_texts: Mapping[int, Sequence[str]],
) -> float:
    """Accuracy on paraphrased/related texts (same image).

    Example
    -------
    >>> def new_model(img, txt):
    ...     return "lenovo" if "brand" in txt else "?"
    >>> edits = [{"image": object(), "text": "What brand?", "target": "lenovo"}]
    >>> rtexts = {0: ["Brand of the laptop?", "Which company made it?\n"]}
    >>> compute_text_generality(new_model, edits, rtexts)
    1.0
    """
    correct = 0
    total = 0
    for idx, e in enumerate(edit_set):
        for t_rel in related_texts.get(idx, []):
            y_pred = model_new(e["image"], t_rel)
            if y_pred == e["target"]:
                correct += 1
            total += 1
    return correct / max(total, 1)


def compute_image_generality(
    model_new: ModelCallable,
    edit_set: Sequence[EditItem],
    related_images: Mapping[int, Sequence[Any]],
) -> float:
    """Accuracy on related images (same text).

    Example
    -------
    >>> def new_model(img, txt):
    ...     return "lenovo"
    >>> edits = [{"image": object(), "text": "What brand?", "target": "lenovo"}]
    >>> rimgs = {0: [object(), object()]}
    >>> compute_image_generality(new_model, edits, rimgs)
    1.0
    """
    correct = 0
    total = 0
    for idx, e in enumerate(edit_set):
        for i_rel in related_images.get(idx, []):
            y_pred = model_new(i_rel, e["text"])
            if y_pred == e["target"]:
                correct += 1
            total += 1
    return correct / max(total, 1)


def compute_locality(
    model_base: ModelCallable,
    model_new: ModelCallable,
    unrelated_inputs: Sequence[UnrelatedItem],
) -> float:
    """Agreement with base model on unrelated inputs.

    Example
    -------
    >>> def base(img, txt):
    ...     return "cat" if txt == "q" else "x"
    >>> def new(img, txt):
    ...     return "cat" if txt == "q" else "y"
    >>> unrelated = [{"image": object(), "text": "q"}]
    >>> compute_locality(base, new, unrelated)
    1.0
    """
    correct = 0
    total = len(unrelated_inputs)
    for u in unrelated_inputs:
        y_base = model_base(u["image"], u["text"])
        y_new = model_new(u["image"], u["text"])
        if y_new == y_base:
            correct += 1
    return correct / max(total, 1)


def combined_score(
    model_base: ModelCallable,
    model_new: ModelCallable,
    edit_set: Sequence[EditItem],
    related_texts: Mapping[int, Sequence[str]],
    related_images: Mapping[int, Sequence[Any]],
    unrelated_inputs: Sequence[UnrelatedItem],
    lambda_gen: float = 1.0,
    lambda_loc: float = 1.0,
    gen_agg: str = "mean",
) -> Dict[str, float]:
    """Combined metric: rel + λ_gen * gen + λ_loc * loc.

    Example
    -------
    >>> def base(img, txt):
    ...     return "hp"
    >>> def new(img, txt):
    ...     return "lenovo"
    >>> edits = [{"image": object(), "text": "brand?", "target": "lenovo"}]
    >>> rtexts = {0: ["brand now?", "who made it?"]}
    >>> rimgs = {0: [object()]}
    >>> unrelated = [{"image": object(), "text": "unrelated"}]
    >>> out = combined_score(base, new, edits, rtexts, rimgs, unrelated)
    >>> sorted(out.keys())
    ['combined', 'image_generality', 'locality', 'reliability', 'text_generality']
    """
    rel = compute_reliability(model_new, edit_set)
    tgen = compute_text_generality(model_new, edit_set, related_texts)
    igen = compute_image_generality(model_new, edit_set, related_images)

    if gen_agg == "harmonic":
        if tgen == 0 or igen == 0:
            gen = 0.0
        else:
            gen = 2.0 / (1.0 / tgen + 1.0 / igen)
    else:
        gen = 0.5 * (tgen + igen)

    loc = compute_locality(model_base, model_new, unrelated_inputs)
    score = rel + lambda_gen * gen + lambda_loc * loc

    return {
        "reliability": float(rel),
        "text_generality": float(tgen),
        "image_generality": float(igen),
        "locality": float(loc),
        "combined": float(score),
    }


