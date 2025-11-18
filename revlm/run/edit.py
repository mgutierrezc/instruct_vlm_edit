import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import torch

# Add project root to path so we can run as a module or script
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *  # noqa: F401,F403
from revlm.config_utils import configure_args
from revlm.editors import get_editor
from revlm.metrics.editeval import reliability


def run_edit(config):
    """Universal edit runner: find errors, edit with chosen editor, report reliability."""
    # Build model
    model = VQAModel(config)

    # Load dataset
    ds = VQADataset(config)
    if config.subsample and len(ds) > config.subsample:
        ds.data = random.sample(ds.data, config.subsample)

    # Step 1: find error subset (edit set) under base model
    ds.set_dataloader(
        with_rationale=config.rationale,
        rationale_in_prompt=True,
        shuffle_choices=True,
        unpaired=True,
    )
    ds.task_generate(model, use_cache=True)
    edit_ds = ds.get_edits()

    print(f"Total examples: {len(ds)}, edit subset (errors): {len(edit_ds.data)}", flush=True)

    # Save a copy of the unedited model
    model_old = copy.deepcopy(model)

    # Step 2: apply edits on edit_ds with chosen editor
    editor = get_editor(config, model)
    editor.generate = model.model.generate if hasattr(model, "model") else model.generate

    # Use a simple loop over the edit dataloader (no history by default)
    edit_ds.set_dataloader(
        with_rationale=config.rationale,
        rationale_in_prompt=False,
        shuffle_choices=True,
    )

    if hasattr(model, "model"):
        model.model.train()

    print(f"Starting edits with editor='{config.editor._name}'...", flush=True)
    for batch_idx, batch in enumerate(edit_ds.loader):
        tokens = model.prepare_training_batch(batch)
        editor.edit(config, tokens, batch_history=None)
        del tokens
        if (batch_idx + 1) % 10 == 0:
            print(f"Edited {batch_idx + 1} batches", flush=True)

    if hasattr(model, "model"):
        model.model.eval()

    model_new = model

    # Step 3: compute reliability before and after edits on the edit subset
    rel_old = reliability(model_old, edit_ds)
    rel_new = reliability(model_new, edit_ds)

    print(f"Reliability (model_old, on edit set): {rel_old:.4f}", flush=True)
    print(f"Reliability (model_new, on edit set): {rel_new:.4f}", flush=True)

    # Save simple edit-eval metrics under config.edit_dir
    out_path = os.path.join(config.edit_dir, config.fname)
    with open(out_path, "w") as f:
        json.dump(
            {
                "reliability_old": float(rel_old),
                "reliability_new": float(rel_new),
                "n_edit_examples": len(edit_ds.data),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Editing Evaluation")

    # Config
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml", help="Path to YAML config file (CLI overrides YAML)")
    parser.add_argument("--editor", type=str, required=True, choices=["ft", "ft_ewc", "ft_retrain", "mend", "grace", "rome", "memory", "defer", "balancedit"], help="Editor method to use")
    parser.add_argument("--model_name", type=str, default=None, help="Short VLM name to map to full HF id (e.g., 'qwen3', 'llava', 'blip')")
    parser.add_argument("--dataset_name", type=str, default="", help="Dataset name (overrides YAML if provided)")
    parser.add_argument("--task", type=str, default="mc", choices=["mc", "mci", "qa"], help="Task type")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for edit dataloader")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"], help="Split to search for edit examples")
    parser.add_argument("--task_dir", type=str, default=None, help="Result directory (overrides config.yaml if provided)")

    # Args
    parser.add_argument("--rationale", action="store_true", help="Append rationale to prompts if available")
    parser.add_argument("--subsample", type=int, default=0, help="Evaluate on a random subset of this many examples (0=all)")
    parser.add_argument("--overwrite", action="store_true", help="Unused here; kept for interface compatibility")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.suffix = "_rationale" if args.rationale else ""
    config = configure_args(args, config_path=args.config)

    # current run-specific settings
    config.overwrite = args.overwrite
    config.subsample = args.subsample
    config.rationale = args.rationale

    run_edit(config)
