"""Build related inputs (generality) using dataset utilities."""

from typing import Any, Dict, List, MutableMapping, Sequence

def build_related_texts_with_taskengineers(
	edit_examples: Sequence[MutableMapping[str, Any]],
	engineer_confs: Sequence[Dict[str, Any]],
	default_task: str = "qa",
	max_per_edit: int = 0,
) -> Dict[int, List[str]]:
	"""Create related texts by re-running TaskEngineer variants on examples.

	- `edit_examples` align with your `edit_set` (same order).
	- Each config in `engineer_confs` is passed to `get_taskengineer(task, **conf)`.

	Example
	-------
	>>> es = [{"question": "What brand?", "answer": "lenovo", "idx_choices": "(A) hp\n(B) lenovo\n(C) dell\n(D) apple"}]
	>>> cfgs = [
	...   {"task": "qa", "with_rationale": False},
	...   {"task": "mci", "shuffle_choices": True},
	... ]
	>>> out = build_related_texts_with_taskengineers(es, cfgs, default_task="qa", max_per_edit=2)
	>>> isinstance(out, dict)
	True
	"""
	from revlm.dataset.utils.taskengineer import get_taskengineer

	texts: Dict[int, List[str]] = {}
	for i, ex in enumerate(edit_examples):
		variants: List[str] = []
		for conf in engineer_confs:
			task = str(conf.get("task", default_task)).lower()
			te = get_taskengineer(task, **{k: v for k, v in conf.items() if k != "task"})
			ex_tmp = dict(ex)
			te.eng_golds(ex_tmp)
			te.eng_prompt(ex_tmp)
			variants.append(ex_tmp.get("prompt", ""))
		if max_per_edit > 0:
			variants = variants[:max_per_edit]
		texts[i] = variants
	return texts


def build_related_images_by_same_label(
	all_examples: Sequence[MutableMapping[str, Any]],
	target_examples: Sequence[MutableMapping[str, Any]],
	label_key: str = "answer",
	max_per_edit: int = 1,
) -> Dict[int, List[Any]]:
	"""Create related images by retrieving other images with the same label.

	Assumes examples have keys 'image' and label under `label_key`.

	Example
	-------
	>>> exs = [
	...   {"image": object(), "answer": "lenovo"},
	...   {"image": object(), "answer": "lenovo"},
	... ]
	>>> out = build_related_images_by_same_label(exs, exs[:1], label_key="answer", max_per_edit=1)
	>>> len(out[0])
	1
	"""
	# Build index from label -> list of images
	from collections import defaultdict
	label_to_images: Dict[str, List[Any]] = defaultdict(list)
	for ex in all_examples:
		lab = str(ex.get(label_key, "")).strip().lower()
		img = ex.get("image")
		if lab and img is not None:
			label_to_images[lab].append(img)

	# For each target example, return up to k other images with same label
	result: Dict[int, List[Any]] = {}
	for i, ex in enumerate(target_examples):
		lab = str(ex.get(label_key, "")).strip().lower()
		candidates = [im for im in label_to_images.get(lab, []) if im is not ex.get("image")]
		if max_per_edit > 0:
			candidates = candidates[:max_per_edit]
		result[i] = candidates
	return result


