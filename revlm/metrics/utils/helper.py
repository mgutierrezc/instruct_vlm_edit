"""Helpers to adapt `VQAModel`-style objects and datasets for metrics."""

from typing import Any, Callable, Dict, List, Sequence


def make_model_callable_vqa(vqa_model: Any) -> Callable[[Any, str], str]:
	"""Return `(image, text) -> str` by wrapping `.generate(...)`.

	Example
	-------
	>>> class Dummy:
	...     def generate(self, images, prompts, **kw):
	...         return ["ok"]
	>>> make_model_callable_vqa(Dummy())(object(), "q")
	'ok'
	"""
	def _call(image: Any, text: str) -> str:
		out = vqa_model.generate(image, text)
		return out[0] if isinstance(out, list) else out

	return _call




def score_choices_single(
	vqa_model: Any,
	image: Any,
	text: str,
	choices: List[str],
	use_avg: bool = False,
	temperature: float = 1.0,
) -> Dict[str, Any]:
	"""Score multiple-choice labels using model APIs if available.

	Uses `VQAModel.score_choices_single` when present; else a simple
	generate-and-match fallback.

	Example
	-------
	>>> class Dummy:
	...     def score_choices_single(self, img, pr, lbls, use_avg=False, temperature=1.0):
	...         return {lbl: {"avg_nll": 1.0, "sum_nll": 1.0, "num_tokens": 1, "prob": 1/len(lbls)} for lbl in lbls}
	>>> sorted(score_choices_single(Dummy(), object(), "q", ["a", "b"]).keys())
	['a', 'b']
	"""
	if hasattr(vqa_model, "score_choices_single"):
		return vqa_model.score_choices_single(image, text, choices, use_avg=use_avg, temperature=temperature)

	# Fallback: generate once and give full weight to exact match
	out = vqa_model.generate(image, text)
	pred = out[0] if isinstance(out, list) else out
	res: Dict[str, Any] = {}
	for lbl in choices:
		p = 1.0 if str(pred).strip().lower() == str(lbl).strip().lower() else 0.0
		res[lbl] = {"avg_nll": 0.0, "sum_nll": 0.0, "num_tokens": 1, "prob": p}
	return res



def build_edit_set_from_dataset(vlm_dataset: Any, use_label_train: bool = True) -> List[Dict[str, Any]]:
	"""Build `edit_set` directly from a `VLMDataset` after `set_dataloader`.

	Uses engineered `prompt` and `gold` fields prepared by `TaskEngineer`.

	Example
	-------
	>>> class D:  # minimal duck-typed dataset
	...     data = [{"image": object(), "prompt": "q", "gold": {"label": "a", "label_train": "a"}}]
	>>> es = build_edit_set_from_dataset(D())
	>>> len(es)
	1
	"""
	text_key = "prompt"
	use_train = bool(use_label_train)
	edit_set: List[Dict[str, Any]] = []
	for ex in vlm_dataset.data:
		gold = ex.get("gold", {})
		target = str(gold.get("label_train" if use_train else "label", gold.get("label", "")))
		edit_set.append({"image": ex.get("image"), "text": ex.get(text_key, ""), "target": target})
	return edit_set
