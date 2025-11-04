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
from .metrics import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_eval(config, args):

    # Build model
    vlm = get_model(config)

    # Load dataset (test split)
    ds = get_dataset(config, split=args.split)
    if args.subsample and len(ds) > args.subsample:
        ds.data = random.sample(ds.data, args.subsample)
    
    # # --- debug: sample 10 examples ---
    # sample_ds = copy.copy(ds)
    # sample_ds.data = [ds.data[i].copy() for i in range(min(10, len(ds.data)))]
    # sample_ds.set_dataloader(task=args.task, with_rationale=args.rationale, batch_size=10)
    # for batch in sample_ds.loader:
    #     sample_ds.task_generate(batch, vlm)
    #     break
    # print(sample_ds.data[:10])

    # ---- run -----
    ds.set_dataloader(
        task=args.task,
        with_rationale=args.rationale,
        rationale_in_prompt=True, # prompt model with "image + prompt + rationale" (if)
        unpaired=True,
        batch_size=10
    )
    for batch in ds.loader:
        ds.task_generate(batch, vlm)
    results = ds.task_engineer.eval(ds)

    # Save under res_dir
    if args.res_dir is not None:
        res_dir = os.path.join("results", args.res_dir)
    else:
        res_dir = getattr(config, "res_dir")
    os.makedirs(res_dir, exist_ok=True)
    out_path = os.path.join(res_dir, f"{args.task}{'_rationale' if args.rationale else ''}_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {out_path}")
    
    # Explicit cleanup to free GPU memory before script exits
    del vlm
    del ds
    torch.cuda.empty_cache()
    print("Cleaned up model and freed GPU memory")






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Evaluation")
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml", help="Path to YAML config file (CLI overrides YAML)")
    parser.add_argument("--editor", type=str, default="raw", choices=["raw", "ft", "ft_ewc", "ft_retrain", "mend", "grace", "rome", "memory", "defer"], help="Editor method to use ('raw' = no editing)")
    parser.add_argument("--model_name", type=str, default=None, help="Short VLM name to map to full HF id (e.g., 'qwen3', 'llava', 'blip')")
    parser.add_argument("--dataset_name", type=str, default="", help="Dataset name (overrides YAML if provided)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Split to evaluate on")
    parser.add_argument("--task", type=str, default="mc", choices=["mc", "mci", "qa"], help="Task to evaluate")
    parser.add_argument("--rationale", action="store_true", help="Append rationale to prompts if available")
    parser.add_argument("--subsample", type=int, default=0, help="Evaluate on a random subset of this many examples (0=all)")
    parser.add_argument("--res_dir", type=str, default=None, help="Result directory (overrides config.yaml if provided)")


    args = parser.parse_args()
    config = configure_args(args, config_path=args.config)
    setattr(config, "device", device)
    run_eval(config, args)

