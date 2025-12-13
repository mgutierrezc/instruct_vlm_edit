import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import torch
import time
import numpy as np

# Add project root to path so we can run as a module or script
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *  # noqa: F401,F403
from revlm.config_utils import configure_args
from revlm.editors import get_editor
from revlm.metrics.editeval import reliability


def print10(dataset, label):
    sample_size = min(10, len(dataset.data))
    sampled_dataset = copy.deepcopy(dataset)
    sampled_dataset.data = copy.deepcopy(dataset.data[:sample_size])
    print(f"\n{label} predictions:", flush=True)
    dataset.task_engineer.eval(sampled_dataset)


def run_edit(config):
    """Universal edit runner: find errors, edit with chosen editor, report reliability."""

    # early return if edit evaluation result already exists
    out_path = os.path.join(config.edit_dir, config.fname)
    if os.path.exists(out_path) and not config.overwrite:
        print(f"Edit evaluation result already exists at {out_path}. Skipping edit evaluation.", flush=True)
        print("-"*50, flush=True)
        return

    # Step 0: determinism + load model and dataset
    seed = getattr(config, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = VQAModel(config)
    pred_snapshot = getattr(config, "pred_path", None)
    if not pred_snapshot:
        pred_snapshot = os.path.join(config.pred_dir, config.fname)
    ds = VQADataset(config)

    # Simple: if COT mode is on, cache cot text from the fresh parquet-backed dataset
    use_cot = getattr(config, "cot", False)
    cot_map = ({str(ex.get("uid")): ex.get("cot", ex.get("rationale", "")) for ex in ds.data} if use_cot else None)
    
    # Step 1: run task generation / load snapshot
    print("="*50, flush=True)
    print("Step 1 (predictions)", flush=True)
    t1 = time.time()
    if os.path.exists(pred_snapshot):  # and not config.overwrite:
        with open(pred_snapshot, "r") as f:
            ds.data = json.load(f)
        print(f"Total samples {len(ds.data)} loaded from {pred_snapshot}", flush=True)
    else:
        if config.subsample and len(ds) > config.subsample:
            ds.data = random.sample(ds.data, config.subsample)
        ds.set_dataloader(
            with_rationale=False,
            use_cot=False,
            rationale_in_prompt=False,
            shuffle_choices=False,
            unpaired=True,
        )
        ds.task_generate(model, use_cache=False)
        out_dir = os.path.dirname(pred_snapshot)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        ds.snap(out_path=pred_snapshot)
        print(f"Total samples {len(ds.data)} saved to {pred_snapshot}", flush=True)

    # Build edit subset (only errors)
    edit_ds = ds.get_edits()  # ds is filtered to only include errors in place
    # In COT mode, ensure each edit example has 'cot' (from cached map or fallback to rationale)
    if use_cot and cot_map:
        for ex in edit_ds.data:
            if "cot" not in ex:
                uid = str(ex.get("uid"))
                ex["cot"] = cot_map.get(uid, ex.get("rationale", ""))

    # Configure its dataloader: choose between rationale vs COT in the target
    edit_ds.set_dataloader(
        with_rationale=config.rationale,
        use_cot=config.cot,
        rationale_in_prompt=False,
        shuffle_choices=False,
        unpaired=True,
    )
    print10(edit_ds, label="model_old")
    model_old = copy.deepcopy(model)
    print(f"Edit subset (errors): {len(edit_ds.data)}", flush=True)
    print(f"Total time: {time.time() - t1:.2f}s", flush=True)

    # Step 2: apply edits on edit_ds with chosen editor
    print("="*50, flush=True)
    print("Step 2 (editing)", flush=True)
    t2 = time.time()
    editor_name = getattr(config.editor, "_name", "")

    # For baseline, we skip constructing an editor and performing any edits; the model and prompts stay unchanged.
    editor = None
    if editor_name != "baseline":
        editor = get_editor(config, model)
        editor.generate = model.model.generate if hasattr(model, "model") else model.generate

    if editor_name == "baseline":
        if hasattr(model, "model"):
            model.model.eval()
    elif editor_name in {"ike", "ike_cot", "ike_clip"}:
        if hasattr(model, "model"):
            model.model.eval()
        editor.edit(config, edit_ds=edit_ds)
    else:
        if hasattr(model, "model"):
            model.model.train()
        print(f"Starting edits with editor='{config.editor._name}'...", flush=True)
        batch_history = []
        retrain_freq = getattr(config.editor, "retrain_frequency", 50)
        for batch_idx, batch in enumerate(edit_ds.loader):
            tokens = model.prepare_training_batch(batch)
            editor.edit(config, tokens, batch_history=batch_history)

            # Keep a lightweight history copy for methods that need replay/regularization
            tokens_copy = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in tokens.items()}
            batch_history.append(tokens_copy)

            # Periodic retraining for ft_retrain
            if editor_name == "ft_retrain" and retrain_freq > 0 and (batch_idx + 1) % retrain_freq == 0:
                print(f"[ft_retrain] periodic retrain at batch {batch_idx + 1}", flush=True)
                editor.retrain(config, batch_history)

            del tokens
            if (batch_idx + 1) % 10 == 0:
                print(f"Edited {batch_idx + 1} batches", flush=True)
        if hasattr(model, "model"):
            model.model.eval()
    edit_ds.task_generate(model, use_cache=False)
    print10(edit_ds, label="model_new")
    print(f"Total time: {time.time() - t2:.2f}s", flush=True)

    # Snap post-edit predictions on the edit set to a separate folder, analogous to `pred`.
    # `configure_args` already guarantees `config.pred_postedit_dir` is a valid directory,
    # so we can join directly here.
    pred_postedit_snapshot = os.path.join(config.pred_postedit_dir, config.fname)
    edit_ds.snap(out_path=pred_postedit_snapshot)
    print(f"Post-edit predictions saved to {pred_postedit_snapshot}", flush=True)

    # Step 3: evaluate the edited model
    print("="*50, flush=True)
    print("Step 3 (evaluation)", flush=True)
    t3 = time.time()
    model_new = model
    dataset_name = config.experiment.dataset_name
    related_texts = get_t_gen_input(dataset_name, edit_ds)
    related_images = get_i_gen_input(dataset_name, edit_ds, k_per_model=2)
    related_r_gen_df = get_r_gen_input(dataset_name, s=2) # has at least 2 sentences of shared rationale
    out_dict = editeval(
        model_old,
        model_new,
        edit_ds,
        editor,
        related_texts,
        related_images,
        related_r_gen_df,
    )

    # reliability() is side-effect free on edit_ds (operates on a deepcopy)
    rel_old = reliability(model_old, edit_ds)
    out_dict['reliability_old'] = rel_old
    
    # add a job finish time
    out_dict['finish_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print(f"Reliability (model_old, on edit set): {out_dict['reliability_old']:.4f}", flush=True)
    print(f"Reliability (model_new, on edit set): {out_dict['reliability']:.4f}", flush=True)
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"Total time: {time.time() - t3:.2f}s", flush=True)
    print(f"Saved edit-eval metrics to {out_path}", flush=True)
    print("="*50, flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Editing Evaluation")

    # Config
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml", help="Path to YAML config file (CLI overrides YAML)")
    parser.add_argument(
        "--editor",
        type=str,
        required=True,
        choices=["ft", "ft_ewc", "ft_retrain", "grace", "balancedit", "ike", "ike_cot", "ike_clip", "mend", "baseline"],
        help="Editor method to use",
    )
    parser.add_argument("--model_name", type=str, default=None, help="Short VLM name to map to full HF id (e.g., 'qwen3', 'qwen3_4b', 'llava', 'blip')")
    parser.add_argument("--dataset_name", type=str, default="", help="Dataset name (overrides YAML if provided)")
    parser.add_argument("--task", type=str, default="mc", choices=["mc", "mci", "qa"], help="Task type")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for edit dataloader")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"], help="Split to search for edit examples")
    parser.add_argument("--edit_dir", type=str, default=None, help="Edit evaluation result directory (overrides config.yaml if provided)")

    # Args
    parser.add_argument("--rationale", action="store_true", help="Append rationale/COT to targets (not prompts) when enabled")
    parser.add_argument("--cot", action="store_true", help="Use COT ('cot' field) instead of 'rationale' when rationale is enabled")
    parser.add_argument("--subsample", type=int, default=0, help="Evaluate on a random subset of this many examples (0=all)")
    parser.add_argument("--pred_path", type=str, default=None, help="Optional path to saved edit dataset. If it exists the file is loaded, otherwise it is written after error discovery.")
    parser.add_argument("--pred_postedit_dir", type=str, default=None, help="Optional directory for saving post-edit predictions on the edit set.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results if they exist")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.rationale:
        if args.cot:
            args.suffix = "_cot"
        else:
            args.suffix = "_rationale"
    else:
        args.suffix = ""
    config = configure_args(args, config_path=args.config)

    # current run-specific settings
    config.subsample = args.subsample
    config.rationale = args.rationale
    config.cot = args.cot
    config.pred_path = args.pred_path
    config.overwrite = args.overwrite
    run_edit(config)
