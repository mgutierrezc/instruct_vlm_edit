import argparse
import os
import sys
from pathlib import Path

import torch

# Add project root to path so we can run as a module or script
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import VQAModel, VQADataset
from revlm.config_utils import configure_args
from revlm.editors import AutoLayer


def run_auto_layer(config, n_runs=10):
    """Run AutoLayer analysis k times for error bars."""
    
    # Build model & dataset
    model = VQAModel(config)
    dataset = VQADataset(config)
    print(f"Model: {config.model.name}, Dataset: {len(dataset)} samples", flush=True)
    
    # Initialize AutoLayer (uses class defaults: n_samples=100, n_aug=10)
    auto = AutoLayer(config, model)
    layers = auto.get_candidate_layers()
    
    # Run k times
    for run_id in range(n_runs):
        auto._samples = None  # Force new random samples each run
        best, scores = auto.find_best(dataset, layers, verbose=(run_id == 0))
        auto.save_results(best, scores, run_id=run_id)
        print(f"Run {run_id+1}/{n_runs} done", flush=True)
    
    # Load all runs and print aggregated results
    print(f"\n{'='*50}", flush=True)
    print("Aggregating results...", flush=True)
    agg_scores = auto.load_results_k()
    best = auto.get_best_from_agg(agg_scores)
    
    # Cleanup
    auto.cleanup()
    del model
    torch.cuda.empty_cache()
    print("Cleaned up model and freed GPU memory", flush=True)
    
    return agg_scores, best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoLayer Analysis")

    # Config
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml")
    parser.add_argument("--model_name", type=str, default="blip")
    parser.add_argument("--dataset_name", type=str, default="aokvqa")
    parser.add_argument("--task", type=str, default="mc", choices=["mc", "mci", "qa"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    parser.add_argument("--subsample", type=int, default=500)

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.editor = "ike_chain"
    args.rationale = False
    args.cot = False
    
    config = configure_args(args, config_path=args.config)
    config.subsample = args.subsample

    run_auto_layer(config)
