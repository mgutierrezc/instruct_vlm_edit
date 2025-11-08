import argparse
import copy
import json
import os
import random
import torch
from PIL import Image

from .config_utils import *
from .dataset import *
from .models import *
# from .metrics import *


def run_eval(config):

    out_path = os.path.join(config.task_dir, config.fname)
    if os.path.exists(out_path) and not config.overwrite:
        print(f"Results already exist at {out_path}. Use --overwrite to overwrite.")
        return
    
    # Build model
    vlm = VQAModel(config)

    # Load dataset
    ds = VQADataset(config)
    if config.subsample and len(ds) > config.subsample:
        ds.data = random.sample(ds.data, config.subsample)

    # ---- run -----
    ds.set_dataloader(
        with_rationale=config.rationale,
        rationale_in_prompt=True, # prompt model with "image + prompt + rationale" (if)
        shuffle_choices=True,
        unpaired=True
    )
    ds.task_generate(vlm)

    # ---- save predictions ----
    ds.snap()

    # ---- save task-based evaluation metrics ----
    ds.task_eval()
    
    # Explicit cleanup to free GPU memory before script exits
    del vlm
    del ds
    torch.cuda.empty_cache()
    print("Cleaned up model and freed GPU memory")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Evaluation")

    # Config
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml", help="Path to YAML config file (CLI overrides YAML)")
    parser.add_argument("--editor", type=str, default="raw", choices=["raw", "ft", "ft_ewc", "ft_retrain", "mend", "grace", "rome", "memory", "defer"], help="Editor method to use ('raw' = no editing)")
    parser.add_argument("--model_name", type=str, default=None, help="Short VLM name to map to full HF id (e.g., 'qwen3', 'llava', 'blip')")
    parser.add_argument("--dataset_name", type=str, default="", help="Dataset name (overrides YAML if provided)")
    parser.add_argument("--task", type=str, default="mc", choices=["mc", "mci", "qa"], help="Task to evaluate")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Split to evaluate on")
    parser.add_argument("--task_dir", type=str, default=None, help="Result directory (overrides config.yaml if provided)")
    
    # Args
    parser.add_argument("--rationale", action="store_true", help="Append rationale to prompts if available")
    parser.add_argument("--subsample", type=int, default=0, help="Evaluate on a random subset of this many examples (0=all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results if they exist")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.suffix = "_rationale" if args.rationale else ""
    config = configure_args(args, config_path=args.config)

    # current run-specific settings
    config.overwrite = args.overwrite
    config.subsample = args.subsample
    config.rationale = args.rationale
    run_eval(config)

