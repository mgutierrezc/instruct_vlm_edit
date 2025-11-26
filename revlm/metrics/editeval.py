from typing import Any, Dict, List, Tuple, Mapping, Sequence
import pandas as pd
import copy
import time
import random
import gc
import torch


def _move_model_device(model: Any, device: str) -> None:
	"""Move a (possibly wrapped) model to the given device if supported."""
	if hasattr(model, "model") and hasattr(model.model, "to"):
		model.model.to(device)
	elif hasattr(model, "to"):
		model.to(device)


def _cuda_gc() -> None:
	"""Lightweight CUDA memory cleanup."""
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

# ! Customize your task-specific generation function here
# inputs: 
# - vlm: VLMModel
# - edit_ds: VQADataset (or your structured dataset that has samples of <"image", "prompt", "target">)
# output: 
# - list of (target, prediction) pairs. 
def generation(model: Any, edit_ds: Any) -> List[Tuple[str, str]]:
	edit_ds.task_generate(model, use_cache=True)
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
		model_old: Any,
		model_new: Any,
		edit_ds: Any,
		editor: Any,
		related_texts: Mapping[int, Sequence[str]],
		related_images: Mapping[int, Sequence[Any]],
		related_r_gen_df: pd.DataFrame,
		unrelated_ds=None,
		loc_sample_size=100,
		lambda_gen: float = 1.0,
		lambda_loc: float = 1.0,
		gen_agg: str = "harmonic",
	) -> Dict[str, float]:
	"""Combined metric: rel + λ_gen * gen + λ_loc * loc.

	gen can be mean or harmonic of text/image generality.
	"""
	t_rel = time.time()
	rel = reliability(model_new, edit_ds)
	print(f"[Timing] reliability: {time.time() - t_rel:.2f}s", flush=True)

	t_tgen = time.time()
	tgen = text_generality(model_new, edit_ds, related_texts)
	print(f"[Timing] text_generality: {time.time() - t_tgen:.2f}s", flush=True)

	t_igen = time.time()
	igen = image_generality(model_new, edit_ds, related_images)
	print(f"[Timing] image_generality: {time.time() - t_igen:.2f}s", flush=True)

	t_rgen = time.time()
	rgen = rationale_generality(model_new, edit_ds, related_r_gen_df)
	print(f"[Timing] rationale_generality: {time.time() - t_rgen:.2f}s", flush=True)

	t_edit1 = time.time()
	edit1 = 0.0
	# edit1 = edit1_generality(model_old, edit_ds, editor)
	print(f"[Timing] edit1_generality: {time.time() - t_edit1:.2f}s", flush=True)

	t_editk = time.time()
	editk = 0.0
	# editk = editk_boot_generality(model_old, edit_ds, editor)
	print(f"[Timing] editk_generality: {time.time() - t_editk:.2f}s", flush=True)

	t_loc = time.time()
	loc = locality(model_old, model_new, edit_ds, unrelated_ds=unrelated_ds, sample_size=loc_sample_size)
	print(f"[Timing] locality: {time.time() - t_loc:.2f}s", flush=True)

	if gen_agg == "harmonic":
		gen = 0.0 if (tgen == 0 or igen == 0 or rgen == 0) else 3.0 / (1.0 / tgen + 1.0 / igen + 1.0 / rgen)
	else:
		gen = 0.5 * (tgen + igen + rgen)

	score = rel + lambda_gen * gen + lambda_loc * loc

	return {
		"reliability": float(rel),
		"text_generality": float(tgen),
		"image_generality": float(igen),
		"rationale_generality": float(rgen),
		"locality": float(loc),
		"edit1_generality": float(edit1),
		"editk_generality": float(editk),
		"hm": float(score),
		"n_edits": float(len(edit_ds.data)),
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


def locality(model_old: Any, model_new: Any, edit_ds: Any, unrelated_ds=None, sample_size=None) -> float:
    """Agreement between base and new models on unrelated inputs.
    
    We form an unrelated set by excluding rows that share the same image or
    question as those in the current edit set, then sample.
    """
    if unrelated_ds is None: # generate unrelated_ds from edit_ds by sampling
        unrelated_ds = copy.deepcopy(edit_ds)
        full_df = edit_ds.load_df()
        used_images = {ex.get("image") for ex in edit_ds.data}
        used_questions = {ex.get("question") for ex in edit_ds.data}
        mask = ~full_df["image_path"].isin(used_images) & ~full_df["question"].isin(used_questions)
        pool_df = full_df.loc[mask].reset_index(drop=True)
        if pool_df.empty:
            raise ValueError("No unrelated inputs found")
        if sample_size is not None:
            pool_df = pool_df.sample(n=min(sample_size, len(pool_df)), random_state=getattr(edit_ds.config, "seed", 333))
        unrelated_ds.data = unrelated_ds.df2data(pool_df)
        unrelated_ds.set_dataloader(shuffle_choices=False)

    # evaluate locality
    pairs_old = generation(model_old, unrelated_ds)
    pairs_new = generation(model_new, unrelated_ds)
    preds_old = [p for _, p in pairs_old]
    preds_new = [p for _, p in pairs_new]
    correct = sum(1 for a, b in zip(preds_old, preds_new) if a == b)
    return correct / len(preds_old)

# def text_locality(model_old: Any, model_new: Any, edit_ds: Any, unrelated_texts: Dict[str, List[str]]) -> float:
#     """Accuracy on unrelated texts using the same images.

#     related_texts: {"image_path": [("unrelated_question1", "unrelated_question1_answer"), 
#                                    ("unrelated_question2", "unrelated_question2_answer"), ...]} aligned to edit_ds.data indices.
#     """

#     df = edit_ds.load_df()
#     unrelated_df = pd.DataFrame(
#         (
#             (image_path, unrelated_question, answer)
#             for image_path, questions in unrelated_texts.items()
#             for unrelated_question, answer in questions
#         ),
#         columns=["image_path", "question"],
#     )
#     # merge unrelated_df with df (without the "question" column) by image_path, keep all rows from unrelated_df
#     unrelated_df = unrelated_df.merge(
#         df.drop(columns=["question"]),
#         on="image_path",
#         how="left",
#     )
#     edit_ds.data = edit_ds.df2data(unrelated_df) # convert to structured dataset of my project
#     edit_ds.set_dataloader()

#     pairs_old = generation(model_old, edit_ds)
#     pairs_new = generation(model_new, edit_ds)
#     preds_old = [p for _, p in pairs_old]
#     preds_new = [p for _, p in pairs_new]
#     correct = sum(1 for a, b in zip(preds_old, preds_new) if a == b)
#     return correct / len(preds_old)

# def image_locality(model_old: Any, model_new: Any, edit_ds: Any, unrelated_images: Dict[str, List[str]]) -> float:
#     """Accuracy on unrelated images using the same texts.

#     unrelated_images: {"question": ["image_path1", "image_path2", ...]} aligned to edit_ds.data indices.
#     """
#     df = edit_ds.load_df()
#     unrelated_df = pd.DataFrame(
#         (
#             (question, image_path)
#             for question, image_paths in unrelated_images.items()
#             for image_path in image_paths
#         ),
#         columns=["question", "image_path"],
#     )
#     # merge unrelated_df with df (without the "text" column) by image_path, keep all rows from unrelated_df
#     unrelated_df = unrelated_df.merge(
#         df.drop(columns=["question"]),
#         on="image_path",
#         how="left",
#     )
#     edit_ds.data = edit_ds.df2data(unrelated_df) # convert to structured dataset of my project
#     edit_ds.set_dataloader()
#     pairs_old = generation(model_old, edit_ds)
#     pairs_new = generation(model_new, edit_ds)
#     preds_old = [p for _, p in pairs_old]
#     preds_new = [p for _, p in pairs_new]
#     correct = sum(1 for a, b in zip(preds_old, preds_new) if a == b)
#     return correct / len(preds_old)


def text_generality(model_new: Any, edit_ds: Any, related_texts: Dict[str, List[str]]) -> float:
    """Accuracy on paraphrased/related texts using the same images.

    related_texts: {"uid": ["question_variant1", "question_variant2", ...]} aligned to edit_ds.data indices.
    """
    df = edit_ds.load_df() # df is the full dataset from HF
    ds = copy.deepcopy(edit_ds) # do not change the original dataset
    related_df = pd.DataFrame(
        (
            (uid, question_variant)
            for uid, variants in related_texts.items()
            for question_variant in variants
        ),
        columns=["uid", "question"],
    )
    related_df = related_df.merge(
        df.drop(columns=["question"]),
        on="uid",
        how="left",
    )
    ds.data = ds.df2data(related_df)
    ds.set_dataloader(shuffle_choices=False)
    return reliability(model_new, ds)

def image_generality(model_new: Any, edit_ds: Any, related_images: Dict[str, List[str]]) -> float:
    """Accuracy on paraphrased/related texts using the same images.

    related_images: {"uid": ["image_path1", "image_path2", ...]} aligned to edit_ds.data indices.
    """
    df = edit_ds.load_df()
    ds = copy.deepcopy(edit_ds) # do not change the original dataset
    related_df = pd.DataFrame(
        (
            (uid, image_path_variant)
            for uid, image_paths in related_images.items()
            for image_path_variant in image_paths
        ),
        columns=["uid", "image_path"],
    )
    # merge related_df with df (without the "question" column) by image_path, keep all rows from related_df
    related_df = related_df.merge(
        df.drop(columns=["image_path"]),
        on="uid",
        how="left",
    )
    ds.data = ds.df2data(related_df) 
    ds.set_dataloader(shuffle_choices=False)
    return reliability(model_new, ds)


def rationale_generality(model_new: Any, edit_ds: Any, related_r_gen_df: pd.DataFrame) -> float:
    """Accuracy on new samples with the same rationale.
    related_r_gen_df: pd.DataFrame with "uid" and "rationale" columns
    """
    ds = copy.deepcopy(edit_ds)
    edit_uid = [str(ex["uid"]) for ex in edit_ds.data]
    related_r_gen_df = related_r_gen_df[related_r_gen_df["uid"].isin(edit_uid)]
    related_r_gen_df['uid'] = related_r_gen_df['sid'].astype(str)

    ds.data = ds.df2data(related_r_gen_df)
    ds.set_dataloader(shuffle_choices=False)
    return reliability(model_new, ds)


def edit1_generality(model_old: Any, edit_ds: Any, editor: Any) -> float:
	"""Leave-one-out generality: edit on one example, test on the rest."""
	n = len(edit_ds.data)
	if n == 0:
		return 0.0

	correct_total = 0
	num_total = 0
	config = edit_ds.config
	editor_name = getattr(config.editor, "_name", getattr(config, "editor", None))

	# Move base model to CPU so deepcopy does not allocate GPU tensors
	if torch.cuda.is_available():
		_move_model_device(model_old, "cpu")
		_cuda_gc()

	# For IKE: build corpus once from full edit_ds (same for all iterations)
	if editor_name == "ike":
		editor.build_corpus_from_dataset(edit_ds.data)

	for i in range(n):
		# fresh model copy for this edit
		new_model = copy.deepcopy(model_old)
		# move working copy to GPU for editing/eval
		if torch.cuda.is_available():
			_move_model_device(new_model, "cuda")
		if hasattr(editor, "model"):
			editor.model = new_model.model if hasattr(new_model, "model") else new_model
		editor.generate = new_model.model.generate if hasattr(new_model, "model") else new_model.generate

		# dataset with just example i
		single_ds = copy.deepcopy(edit_ds)
		single_ds.data = [edit_ds.data[i]]

		if editor_name == "ike":
			# IKE: retrieval-only, augment prompts via dataset API
			if hasattr(new_model, "model"):
				new_model.model.eval()
			editor.edit(config, edit_ds=single_ds, train_ds=edit_ds)
		else:
			# Weight-updating editors: train on a single batch
			if hasattr(new_model, "model"):
				new_model.model.train()
			single_ds.set_dataloader(
				with_rationale=getattr(config, "rationale", False),
				rationale_in_prompt=False,
				shuffle_choices=True,
			)
			batch = next(iter(single_ds.loader))
			tokens = new_model.prepare_training_batch(batch)
			editor.edit(config, tokens, batch_history=None)
			del tokens
			if hasattr(new_model, "model"):
				new_model.model.eval()

		# evaluate on remaining examples
		remain_examples = [edit_ds.data[j] for j in range(n) if j != i]
		if not remain_examples:
			continue
		ds_eval = copy.deepcopy(edit_ds)
		ds_eval.data = remain_examples
		ds_eval.set_dataloader(shuffle_choices=False)

		if hasattr(new_model, "model"):
			new_model.model.eval()
		pairs = generation(new_model, ds_eval)
		correct_total += sum(1 for t, p in pairs if p == t)
		num_total += len(pairs)

		# Clean up GPU memory before the next iteration
		if hasattr(editor, "model"):
			editor.model = None
		del new_model
		_cuda_gc()

	if num_total == 0:
		return 0.0
	return correct_total / num_total


def editk_boot_generality(
	model_old: Any,
	edit_ds: Any,
	editor: Any,
	B: int = 100,
	k: int = 10,
) -> float:
	"""Bootstrap generality: repeatedly edit on k samples, test on the rest.

	Args:
		(model_old, edit_ds, editor): as in edit1_generality.
		B: number of bootstrap rounds.
		k: number of edit samples per round.
	"""
	n = len(edit_ds.data)
	if n == 0 or k <= 0 or B <= 0:
		return 0.0
	k = min(k, n)

	config = edit_ds.config
	editor_name = getattr(config.editor, "_name", getattr(config, "editor", None))

	# For IKE: build corpus once from full edit_ds (same for all iterations)
	if editor_name == "ike":
		editor.build_corpus_from_dataset(edit_ds.data)

	# Pre-generate seeds for reproducible bootstrapping
	base_seed = getattr(config, "seed", 333)
	rng = random.Random(base_seed)
	seeds = [rng.randint(0, 2**31 - 1) for _ in range(1000)]
	B_eff = min(B, len(seeds))

	correct_total = 0
	num_total = 0

	# Move base model to CPU so deepcopy does not allocate GPU tensors
	if torch.cuda.is_available():
		_move_model_device(model_old, "cpu")
		_cuda_gc()

	for b in range(B_eff):
		rng_round = random.Random(seeds[b])
		edit_indices = rng_round.sample(range(n), k)

		# fresh model copy for this round
		new_model = copy.deepcopy(model_old)
		# move working copy to GPU for editing/eval
		if torch.cuda.is_available():
			_move_model_device(new_model, "cuda")
		if hasattr(editor, "model"):
			editor.model = new_model.model if hasattr(new_model, "model") else new_model
		editor.generate = new_model.model.generate if hasattr(new_model, "model") else new_model.generate

		# apply edits on the k sampled examples
		for i in edit_indices:
			single_ds = copy.deepcopy(edit_ds)
			single_ds.data = [edit_ds.data[i]]

			if editor_name == "ike":
				# IKE: retrieval-only, augment prompts via dataset API
				if hasattr(new_model, "model"):
					new_model.model.eval()
				editor.edit(config, edit_ds=single_ds, train_ds=edit_ds)
			else:
				# Weight-updating editors: train on a single batch
				if hasattr(new_model, "model"):
					new_model.model.train()
				single_ds.set_dataloader(
					with_rationale=getattr(config, "rationale", False),
					rationale_in_prompt=False,
					shuffle_choices=True,
				)
				batch = next(iter(single_ds.loader))
				tokens = new_model.prepare_training_batch(batch)
				editor.edit(config, tokens, batch_history=None)
				del tokens
				if hasattr(new_model, "model"):
					new_model.model.eval()

		# evaluate on remaining examples (complement of edit_indices)
		remain_examples = [edit_ds.data[j] for j in range(n) if j not in edit_indices]
		if not remain_examples:
			continue
		ds_eval = copy.deepcopy(edit_ds)
		ds_eval.data = remain_examples
		ds_eval.set_dataloader(shuffle_choices=False)

		if hasattr(new_model, "model"):
			new_model.model.eval()
		pairs = generation(new_model, ds_eval)
		correct_total += sum(1 for t, p in pairs if p == t)
		num_total += len(pairs)

		# Clean up GPU memory before the next bootstrap round
		if hasattr(editor, "model"):
			editor.model = None
		del new_model
		_cuda_gc()

	if num_total == 0:
		return 0.0
	return correct_total / num_total