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


def run_auto_layer(config, n_runs=10, n_samples=100, n_aug=10):
    """Run AutoLayer analysis k times for error bars."""
    
    # Build model & dataset
    model = VQAModel(config)
    dataset = VQADataset(config)
    print(f"Model: {config.model.name}, Dataset: {len(dataset)} samples", flush=True)
    
    # Initialize AutoLayer
    auto = AutoLayer(config, model)
    layers = auto.get_candidate_layers()
    
    # Run k times
    for run_id in range(n_runs):
        # Check if this run already exists
        out_dir = auto.out_dir
        out_path = os.path.join(out_dir, f"scores_run{run_id}.json")
        if os.path.exists(out_path) and not config.overwrite:
            print(f"Run {run_id} already exists, skipping.", flush=True)
            continue
        
        auto._samples = None  # Force new random samples each run
        best, scores = auto.find_best(dataset, layers, n_samples=n_samples, n_aug=n_aug, verbose=(run_id == 0))
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
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml", help="Path to YAML config file")
    parser.add_argument("--model_name", type=str, default="blip", help="Short VLM name (e.g., 'qwen3', 'qwen3_4b', 'llava', 'blip')")
    parser.add_argument("--dataset_name", type=str, default="aokvqa", help="Dataset name")
    parser.add_argument("--task", type=str, default="mc", choices=["mc", "mci", "qa"], help="Task type")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"], help="Split")

    # AutoLayer-specific args
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs for error bars")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples per run")
    parser.add_argument("--n_aug", type=int, default=3, help="Number of augmentations per sample")

    # Other args
    parser.add_argument("--subsample", type=int, default=100, help="Subsample size for dataset (0=all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.editor = "ike_chain"  # Placeholder, not used but needed for configure_args
    args.rationale = False
    args.cot = False
    
    config = configure_args(args, config_path=args.config)
    config.subsample = args.subsample
    config.overwrite = args.overwrite

    run_auto_layer(config, n_runs=args.n_runs, n_samples=args.n_samples, n_aug=args.n_aug)

