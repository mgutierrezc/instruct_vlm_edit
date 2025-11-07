from typing import Any, Dict, List, Tuple

# Customize your task-specific generation function here
# inputs: 
# - vlm: VLMModel
# - edit_ds: VQADataset (or your structured dataset that has samples of <"image", "prompt", "target">)
# output: 
# - list of (target, prediction) pairs. 
def generation(vlm: Any, edit_ds: Any) -> List[Tuple[str, str]]:
	# edit_ds.loader should be already set via set_dataloader(...)
	for batch in edit_ds.loader:
		edit_ds.task_generate(batch, vlm)
	edit_set, pred_set = get_edit_pred(edit_ds)
	return [(e["target"], p["pred"]) for e, p in zip(edit_set, pred_set)]


def get_edit_pred(vlm_dataset: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
	edit_set: List[Dict[str, Any]] = []
	pred_set: List[Dict[str, Any]] = []
	for ex in vlm_dataset.data:
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
	return edit_set, pred_set

