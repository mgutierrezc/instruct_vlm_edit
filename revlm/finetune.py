import argparse
import logging
import os
import torch
import numpy as np
import json
import random
import time

from .models import *
from .dataset import *
from .editors import *
from .editors.utils import explore_layers, validate_and_correct_param_name
from .config_utils import configure_args

logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
# LOG = logging.getLogger(__name__)


def finetune(config):
    """Main finetuning function"""
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    print(f"model={config.model.name}, dataset={config.experiment.dataset_name}, batch_size={config.batch_size}, n_iter={config.n_iter}, "
          f"editor={config.editor._name}, rationale={getattr(config.experiment, 'with_rationale', False)}", flush=True)
    
    # Check if output files already exist (early exit to avoid loading model/datasets)
    with_rationale = getattr(config.experiment, 'with_rationale', False)
    task = getattr(config.experiment, 'task', 'mc')
    res_dir_arg = getattr(config, 'res_dir_arg', None)
    if res_dir_arg is not None:
        res_dir = os.path.join("results", res_dir_arg)
    else:
        res_dir = getattr(config, "res_dir")
    os.makedirs(res_dir, exist_ok=True)
    rationale_suffix = "_rationale" if with_rationale else ""
    out_path_test = os.path.join(res_dir, f"{task}{rationale_suffix}_test.json")
    out_path_train = os.path.join(res_dir, f"{task}{rationale_suffix}_train.json")
    
    overwrite = getattr(config, 'overwrite', False)
    if os.path.exists(out_path_test) and os.path.exists(out_path_train) and not overwrite:
        print(f"Results already exist at {out_path_test} and {out_path_train}. Use --overwrite to overwrite.", flush=True)
        return
    
    device = torch.device(config.device if isinstance(config.device, str) else config.device)
    
    # Load model
    print("Loading model...", flush=True)
    t0 = time.time()
    model = VQAModel(config).to(device)
    print(f"Model loaded in {time.time() - t0:.2f}s", flush=True)
    
    # Auto-select layer if not provided
    if not getattr(config.model, 'inner_params', []) or len(config.model.inner_params) == 0:
        print("Auto-selecting layer...", flush=True)
        t0 = time.time()
        suggestions = explore_layers(model.model)
        if suggestions:
            config.model.inner_params = [suggestions[0]]
            print(f"Auto-selected layer: {config.model.inner_params[0]} (took {time.time() - t0:.2f}s)", flush=True)
        else:
            raise ValueError("No suitable layers found and inner_params not provided")
    
    # Validate and correct parameter name (before creating editor)
    # COMMENTED OUT: Validation not needed - auto-selection/YAML configs provide correct layer names
    # validated_param = validate_and_correct_param_name(model.model, config.model.inner_params[0], logger=LOG)
    # if validated_param != config.model.inner_params[0]:
    #     config.model.inner_params[0] = validated_param
    #     LOG.info(f"Using validated parameter: {validated_param}")
    
    # Load datasets
    print("Loading datasets...", flush=True)
    t0 = time.time()
    train_dataset = VQADataset(config)
    test_dataset = VQADataset(config)
    print(f"Datasets loaded in {time.time() - t0:.2f}s (train: {len(train_dataset)}, test: {len(test_dataset)})", flush=True)
    
    subsample = getattr(config, 'subsample', 0)
    if subsample and len(train_dataset) > subsample:
        train_dataset.data = random.sample(train_dataset.data, subsample)
    if subsample and len(test_dataset) > subsample:
        test_dataset.data = random.sample(test_dataset.data, subsample)

    # Setup dataloaders
    print(f"Setting up train dataloader (processing {len(train_dataset)} examples)...", flush=True)
    t0 = time.time()
    train_dataset.set_dataloader(
        with_rationale=with_rationale,
        rationale_in_prompt=False, # image + prompt -> label + rationale
        shuffle_choices=True,
    )
    print(f"Train dataloader setup in {time.time() - t0:.2f}s", flush=True)
    
    print(f"Setting up test dataloader (processing {len(test_dataset)} examples)...", flush=True)
    t0 = time.time()
    test_dataset.set_dataloader(
        with_rationale=False,
        shuffle_choices=False,
    )
    print(f"Test dataloader setup in {time.time() - t0:.2f}s", flush=True)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}", flush=True)
    
    # Load editor
    editor = get_editor(config, model, device)
    editor.generate = model.model.generate if hasattr(model, 'model') else model.generate
    
    # Finetuning loop
    model.model.train()
    batch_history = []
    losses = []
    
    # Check if editor needs batch_history (e.g., ft_ewc, ft_retrain)
    # If so, we'll collect batches first before starting training
    editor_name = getattr(config.editor, '_name', '')
    needs_history = editor_name in ['ft_ewc', 'ft_retrain']
    prefill_size = 2 if needs_history else 0  # Pre-fill with at least 2 batches for history-based editors
    
    print("Starting finetuning...", flush=True)
    total_batches = len(train_dataset.loader)
    for batch_idx, batch in enumerate(train_dataset.loader):
        tokens = model.prepare_training_batch(batch)
        
        # For history-based editors, collect batches first before training
        if needs_history and len(batch_history) < prefill_size:
            batch_history.append(tokens)
            if len(batch_history) == prefill_size:
                print(f"Pre-populated batch_history with {len(batch_history)} batches for {editor_name}", flush=True)
            continue  # Skip editing until we have enough history
        
        # Edit (finetune) on this batch
        editor.edit(config, tokens, batch_history)
        
        # Track history only for editors that need it (avoid holding large tensors for ft)
        if needs_history:
            batch_history.append(tokens)
            max_history = getattr(config.editor, 'fisher_mem', 10) if hasattr(config.editor, 'fisher_mem') else \
                          getattr(config.editor, 'retrain_memory', 100) if hasattr(config.editor, 'retrain_memory') else 10
            if len(batch_history) > max_history:
                batch_history = batch_history[-max_history:]

        # Release per-batch tensors ASAP to reduce VRAM pressure when not needed
        del tokens
        
        # Track losses
        if hasattr(editor, 'losses') and editor.losses:
            losses.extend(editor.losses)
        
        # Periodic logging every 10 batches
        if (batch_idx + 1) % 10 == 0:
            recent_losses = losses[-10:] if len(losses) >= 10 else losses
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0
            print(f"Batch {batch_idx + 1}/{total_batches}, Avg loss (last 10): {avg_loss:.4f}", flush=True)
            # Periodically purge cached memory to smooth peak usage
            torch.cuda.empty_cache()
    
    print(f"Finetuning complete. Total batches: {total_batches}", flush=True)
    
    model.model.eval()
    with torch.no_grad():
        print("Evaluating on train set...", flush=True)
        train_dataset.task_generate(model)
        train_metrics = train_dataset.task_engineer.eval(train_dataset)
        print(f"Train metrics: {train_metrics}", flush=True)
        print("Evaluating on test set...", flush=True)
        test_dataset.task_generate(model)
        test_metrics = test_dataset.task_engineer.eval(test_dataset)
        print(f"Test metrics: {test_metrics}", flush=True)
    # Save evaluation metrics (using eval.py structure: nested folders in res_dir)
    # Note: res_dir and paths are already constructed at the start of the function
    with open(out_path_test, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    with open(out_path_train, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Saved evaluation metrics: {out_path_test} and {out_path_train}", flush=True)
    
    # Save checkpoint if requested
    if config.ckpt_dir:
        os.makedirs(config.ckpt_dir, exist_ok=True)
        model_tag = config.model.name.split("/")[-1].replace(" ", "_")
        dataset_tag = config.experiment.dataset_name
        editor_tag = config.editor._name
        rationale_tag = "rationale" if with_rationale else "norationale"
        ckpt_path = os.path.join(
            config.ckpt_dir, 
            f"{model_tag}_{dataset_tag}_{editor_tag}_{rationale_tag}.pt"
        )
        torch.save(model.model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}", flush=True)
    
    # Explicit cleanup to free GPU memory before script exits
    del model
    del editor
    del train_dataset
    del test_dataset
    torch.cuda.empty_cache()
    print("Cleaned up model and freed GPU memory", flush=True)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Finetuning")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--editor", type=str, required=True, choices=["ft", "ft_ewc", "ft_retrain"], help="Editor method")
    parser.add_argument("--model_name", type=str, default=None, help="Model name: 'qwen3', 'llava', 'blip'")
    parser.add_argument("--inner_params", type=str, nargs='+', default=[], help="Layer to finetune (auto-selected if empty)")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["aokvqa", "fvqa"], help="Dataset name")
    parser.add_argument("--task", type=str, default=None, choices=["mc", "mci", "qa"], help="Task type (uses config.yaml if not provided)")
    parser.add_argument("--with_rationale", action="store_true", help="Include rationale in prompts (uses config.yaml if not provided)")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--n_iter", type=int, default=5, help="Inner iterations per batch")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory to save checkpoints (overrides config.yaml)")
    parser.add_argument("--subsample", type=int, default=0, help="Evaluate on a random subset of this many examples (0=all)")
    parser.add_argument("--res_dir", type=str, default=None, help="Result directory (overrides config.yaml if provided)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results if they exist")
    
    args = parser.parse_args()
    
    # Create config
    cfg_path = args.config or os.path.join(
        os.path.dirname(__file__), "config", "config.yaml"
    )
    config = configure_args(args, config_path=cfg_path)
    
    # Override settings
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.subsample = args.subsample
    config.res_dir_arg = args.res_dir
    config.overwrite = args.overwrite
    if args.ckpt_dir is not None:
        config.ckpt_dir = args.ckpt_dir
    if args.with_rationale:
        config.experiment.with_rationale = True
    
    # Run finetuning
    finetune(config)

