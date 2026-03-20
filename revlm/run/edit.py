import argparse
import copy
import json
import os
import pickle as pkl
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm
import time
import numpy as np
import pandas as pd

# Add project root to path so we can run as a module or script
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *
from revlm.config_utils import configure_args
from .edit_utils import edit_n_eval_seq, edit_n_eval_all, edit_n_eval_all_loc, edit_n_eval_indep_all, find_errors

def run_edit(config, sequential=False, eval_every=200, subsample_path="", biases_path=""):
    """Universal edit runner: find errors, edit with chosen editor, report reliability."""
    
    # Enable wandb if sequential mode is used
    config.wandb = sequential

    # for replicability
    seed = getattr(config, "seed", 42)
    random.seed(seed)

    # early return if edit evaluation result already exists
    out_path = os.path.join(config.edit_dir, config.fname)
    if os.path.exists(out_path) and not config.overwrite:
        print(f"Edit evaluation result already exists at {out_path}. Skipping edit evaluation.", flush=True)
        print("-"*50, flush=True)
        return

    # Step 1: Find errors
    model, edit_ds = find_errors(config)
    
    # Subsample edit set if requested
    subsample_edits = getattr(config, "subsample_edits", 0)
    if subsample_edits and len(edit_ds.data) > subsample_edits:
        print(f"Subsampling edit set from {len(edit_ds.data)} to {subsample_edits} examples", flush=True)
        edit_ds.data = random.sample(edit_ds.data, subsample_edits)
        edit_ds.set_dataloader()
    
    # storing subsample
    if subsample_path != "":
        # creating parent dir if it doesn't exist
        parent_dir = os.path.dirname(subsample_path)
        os.makedirs(parent_dir, exist_ok=True)
        print(f"created parent_dir: {parent_dir}")

        if not os.path.exists(subsample_path):
            with open(subsample_path, "w") as f:
                json.dump(edit_ds.data, f, indent=2)
                print("stored edit_ds")
    # exit()

    # loading biases subsample
    if biases_path != "":
        with open(biases_path, "r") as f:
            biases_data = json.load(f)
            edit_ds.data = biases_data
            edit_ds.set_dataloader()
            print(f"Loaded {len(edit_ds.data)} entries from {biases_path}", flush=True)
        
    # Step 2-3: Edit and Evaluate on all errors
    if sequential:
        out_dict = edit_n_eval_seq(config, model, edit_ds, out_path, eval_every=eval_every)
    else:
        out_dict = edit_n_eval_all(config, model, edit_ds, out_path)

def run_edit_locality(config, sequential=False, eval_every=200, loc_sample_path=""):
    """Universal edit runner: find errors, edit with chosen editor, report reliability."""
    
    # Enable wandb if sequential mode is used
    config.wandb = sequential

    q_index = config.q_index

    # for replicability
    seed = getattr(config, "seed", 42)
    random.seed(seed)

    # early return if edit evaluation result already exists
    out_path = os.path.join(config.edit_dir, config.fname)
    if os.path.exists(out_path) and not config.overwrite:
        print(f"Edit evaluation result already exists at {out_path}. Skipping edit evaluation.", flush=True)
        print("-"*50, flush=True)
        return

    # Step 1: Find errors
    model, edit_ds = find_errors(config)
    
    # loading subsample
    with open(loc_sample_path, "rb") as f:
        ## edit sample
        loc_sample = pkl.load(f)
        loc_sample = loc_sample[f"q{q_index}"]
        edit_sample_data = loc_sample["edit"]
        edit_sample_data = edit_sample_data.to_dict(orient="records")
        edit_ds.data = edit_sample_data
        edit_ds.set_dataloader()
        print(f"Loaded {len(edit_ds.data)} entries from {loc_sample_path}", flush=True)

        ## locality sample
        unrelated_ds = copy.deepcopy(edit_ds)
        unrelated_sample_data = loc_sample["unrelated"]
        unrelated_sample_data = unrelated_sample_data.to_dict(orient="records")
        unrelated_ds.data = unrelated_sample_data
        unrelated_ds.set_dataloader()
        print(f"Loaded {len(unrelated_ds.data)} entries from {loc_sample_path}", flush=True)

    out_dict = edit_n_eval_all_loc(config, model, edit_ds, unrelated_ds, out_path)

def run_edit_indep(config):
    """Universal edit runner: find errors, edit with chosen editor, report reliability."""
    
    # we need a run that goes across indexes

    # for replicability
    seed = getattr(config, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # setting up model and datasets
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    eval_ds = VQADataset(config)
    
    # loading dataframes
    eval_df = pd.read_pickle(config.eval_df_path)
    correct_edit_df = pd.read_pickle(config.correct_edit_path)
    wrong_edit_df = pd.read_pickle(config.wrong_edit_path)

    def parse_range(s):
        if "-" in str(s):
            return s.split("-")
        else:
            return [s]
    
    ouputs = []

    indices_parsed = parse_range(config.indices)
    print(f"indices_parsed: {indices_parsed}")
    for index in tqdm(indices_parsed):
        model = VQAModel(config)
        
        # keeping current index rows
        print(f"current index: {index}")
        num_index = int(index)
        eval_df_filtered = eval_df[eval_df["qa_id"] == num_index]
        correct_edit_df_filtered = correct_edit_df[correct_edit_df["qa_id"] == num_index]
        wrong_edit_df_filtered = wrong_edit_df[wrong_edit_df["qa_id"] == num_index]
        print(f"filtered dsets, current length: {len(eval_df_filtered)}")

        # formatting for assignment
        eval_df_filtered = eval_df_filtered.to_dict(orient="records")
        correct_edit_df_filtered = correct_edit_df_filtered.to_dict(orient="records")
        wrong_edit_df_filtered = wrong_edit_df_filtered.to_dict(orient="records")
        print("formatted dfs for assignment")

        # setting dfs as data attribute for editing + eval
        eval_ds.data = eval_df_filtered
        eval_ds.set_dataloader()

        correct_edit_ds = copy.deepcopy(eval_ds)
        correct_edit_ds.data = correct_edit_df_filtered
        correct_edit_ds.set_dataloader()

        wrong_edit_ds = copy.deepcopy(eval_ds)
        wrong_edit_ds.data = wrong_edit_df_filtered
        wrong_edit_ds.set_dataloader()
        print("assigned dfs to ds.data attribute")

        current_outputs = edit_n_eval_indep_all(config, model, wrong_edit_ds, correct_edit_ds, eval_ds)
        current_outputs["qa_pair"] = num_index
        pd.DataFrame([current_outputs]).to_excel(config.output_path + str(index) + ".xlsx", index=False)

        # freeing up memory after every update
        del model

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Editing Evaluation")

    # Config
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml", help="Path to YAML config file (CLI overrides YAML)")
    parser.add_argument(
        "--editor",
        type=str,
        required=True,
        choices=["ft", "ft_retrain", "grace", "grace_cot", "balancedit", "ike", "ike_cot", "ike_chain", "mend", "mend_retrain", "liveedit", "liveedit_cot", "baseline"],
        help="Editor method to use",
    )
    parser.add_argument("--model_name", type=str, default=None, help="Short VLM name to map to full HF id (e.g., 'qwen3', 'qwen3_4b', 'llava', 'blip')")
    parser.add_argument("--dataset_name", type=str, default="", help="Dataset name (overrides YAML if provided)")
    parser.add_argument("--task", type=str, default="mc", choices=["mc", "mci", "qa"], help="Task type")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for edit dataloader")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"], help="Split to search for edit examples")
    parser.add_argument("--edit_dir", type=str, default=None, help="Edit evaluation result directory (overrides config.yaml if provided)")
    parser.add_argument("--sequential", action="store_true", help="Run sequential editing/eval (enables wandb logging)")

    # Args
    parser.add_argument("--rationale", action="store_true", help="Append rationale/COT to targets (not prompts) when enabled")
    parser.add_argument("--cot", action="store_true", help="Use COT ('cot' field) instead of 'rationale' when rationale is enabled")
    parser.add_argument("--subsample", type=int, default=0, help="Evaluate on a random subset of this many examples (0=all)")
    parser.add_argument("--pred_path", type=str, default=None, help="Optional path to saved edit dataset. If it exists the file is loaded, otherwise it is written after error discovery.")
    parser.add_argument("--pred_postedit_dir", type=str, default=None, help="Optional directory for saving post-edit predictions on the edit set.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results if they exist")
    parser.add_argument("--mode", type=str, default=None, choices=["vision", "language", "language_last", "dual_sbert"], help="Embedding mode for IKE_CHAIN")
    parser.add_argument("--pool_method", type=str, default=None, choices=["mean", "last"], help="Pooling method for IKE_CHAIN")
    parser.add_argument("--coe_pt", action="store_true", help="Enable COE question perturbation (default: disabled)")
    parser.add_argument("--subsample_edits", type=int, default=0, help="Subsample edit set to this many examples after error discovery (0=all)")
    parser.add_argument("--n_edit_cap", type=int, default=None, help="Cap number of edits (subsample edit set before training)")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max new tokens for VLM generation (default from config.yaml)")

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

    # Override editor settings if provided
    if args.mode:
        config.editor.mode = args.mode
    if args.pool_method:
        config.editor.pool_method = args.pool_method

    # current run-specific settings
    config.subsample = args.subsample
    config.rationale = args.rationale
    config.cot = args.cot
    config.pred_path = args.pred_path
    config.overwrite = args.overwrite
    config.coe_pt = args.coe_pt
    config.subsample_edits = args.subsample_edits
    config.n_edit_cap = args.n_edit_cap
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    run_edit(config, sequential=args.sequential)
