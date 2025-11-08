from typing import Any, Dict, List, Tuple, Mapping, Sequence

import pandas as pd

# ! Customize your task-specific generation function here
# inputs: 
# - vlm: VLMModel
# - edit_ds: VQADataset (or your structured dataset that has samples of <"image", "prompt", "target">)
# output: 
# - list of (target, prediction) pairs. 
def generation(model: Any, edit_ds: Any) -> List[Tuple[str, str]]:
	edit_ds.task_generate(model)
	edit_set: List[Dict[str, Any]] = []
	pred_set: List[Dict[str, Any]] = []
	for ex in edit_ds.data:
		gold = ex.get("gold", {})
		pred = ex.get("pred", {})
		if pred:
			edit_set.append({
				"idx": ex.get("idx"),
				"image": ex.get("image"),
				"text": ex.get("prompt", ""),
				"target": gold.get("label", ""),
				"rationale": ex.get("rationale", ""),
			})
			pred_set.append({
				"idx": ex.get("idx"),
				"image": ex.get("image"),
				"text": ex.get("prompt", ""),
				"pred": pred.get("label_maxprob", ""),
			})
	return [(e["target"], p["pred"]) for e, p in zip(edit_set, pred_set)]


def editeval(
    model_base: Any,
    model_new: Any,
    edit_ds: Any,
    related_texts: Mapping[int, Sequence[str]],
    related_images: Mapping[int, Sequence[Any]],
    unrelated_ds: Any,
    lambda_gen: float = 1.0,
    lambda_loc: float = 1.0,
    gen_agg: str = "harmonic",
) -> Dict[str, float]:
    """Combined metric: rel + λ_gen * gen + λ_loc * loc.

    gen can be mean or harmonic of text/image generality.
    """
    rel = reliability(model_new, edit_ds)
    tgen = text_generality(model_new, edit_ds, related_texts)
    igen = image_generality(model_new, edit_ds, related_images)

    if gen_agg == "harmonic":
        gen = 0.0 if (tgen == 0 or igen == 0) else 2.0 / (1.0 / tgen + 1.0 / igen)
    else:
        gen = 0.5 * (tgen + igen)

    loc = locality(model_base, model_new, unrelated_ds)
    score = rel + lambda_gen * gen + lambda_loc * loc

    return {
        "reliability": float(rel),
        "text_generality": float(tgen),
        "image_generality": float(igen),
        "locality": float(loc),
        "combined": float(score),
    }


def reliability(model_new: Any, edit_ds: Any) -> float:
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


def locality(model_old: Any, model_new: Any, unrelated_ds: Any) -> float:
    """Agreement between base and new models on unrelated dataset inputs.

    Uses batch generation on (image, prompt) pairs from unrelated_ds.
    """

    pairs_old = generation(model_old, unrelated_ds)
    pairs_new = generation(model_new, unrelated_ds)
    preds_old = [p for _, p in pairs_old]
    preds_new = [p for _, p in pairs_new]
    correct = sum(1 for a, b in zip(preds_old, preds_new) if a == b)
    return correct / len(preds_old)


def text_generality(model_new: Any, edit_ds: Any, related_texts: Dict[str, List[str]]) -> float:
    """Accuracy on paraphrased/related texts using the same images.

    related_texts: {"image_path": ["question_variant1", "question_variant2", ...]} aligned to edit_ds.data indices.
    """
    df = edit_ds._load_df()
    related_df = pd.DataFrame(
        (
            (image_path, question_variant)
            for image_path, variants in related_texts.items()
            for question_variant in variants
        ),
        columns=["image_path", "question"],
    )
    # merge related_df with df (without the "question" column) by image_path, keep all rows from related_df
    related_df = related_df.merge(
        df.drop(columns=["question"]),
        on="image_path",
        how="left",
    )
    related_df = pd.concat([related_df, df], axis=0, ignore_index=True)
    edit_ds.data = edit_ds.df2data(related_df) # convert to structured dataset of my project
    edit_ds.set_dataloader()
    return reliability(model_new, edit_ds)


def image_generality(model_new: Any, edit_ds: Any, related_images: Dict[str, List[str]]) -> float:
    """Accuracy on paraphrased/related texts using the same images.

    related_texts: {"question": ["image_path1", "image_path2", ...]} aligned to edit_ds.data indices.
    """
    df = edit_ds._load_df()
    related_df = pd.DataFrame(
        (
            (question, image_path_variant)
            for question, image_paths in related_images.items()
            for image_path_variant in image_paths
        ),
        columns=["question", "image_path"],
    )
    # merge related_df with df (without the "question" column) by image_path, keep all rows from related_df
    related_df = related_df.merge(
        df.drop(columns=["image_path"]),
        on="question",
        how="left",
    )
    related_df = pd.concat([related_df, df], axis=0, ignore_index=True)
    edit_ds.data = edit_ds.df2data(related_df) # convert to structured dataset of my project
    edit_ds.set_dataloader()
    return reliability(model_new, edit_ds)


