import argparse
import logging
import os
import torch
import numpy as np

from .models import get_model
from .dataset import get_dataset
from .editors import get_editor
from .editors.utils import explore_layers, validate_and_correct_param_name
from .config_utils import configure_args

logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
LOG = logging.getLogger(__name__)


def finetune(config):
    """Main finetuning function"""
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    LOG.info(f"Starting finetuning: model={config.model.name}, dataset={config.experiment.dataset_name}, "
             f"editor={config.editor._name}, rationale={getattr(config.experiment, 'with_rationale', False)}")
    
    device = torch.device(config.device if isinstance(config.device, str) else config.device)
    
    # Load model
    model = get_model(config).to(device)
    
    # Auto-select layer if not provided
    if not getattr(config.model, 'inner_params', []) or len(config.model.inner_params) == 0:
        suggestions = explore_layers(model.model)
        if suggestions:
            config.model.inner_params = [suggestions[0]]
            LOG.info(f"Auto-selected layer: {config.model.inner_params[0]}")
        else:
            raise ValueError("No suitable layers found and inner_params not provided")
    
    # Validate and correct parameter name (before creating editor)
    # COMMENTED OUT: Validation not needed - auto-selection/YAML configs provide correct layer names
    # validated_param = validate_and_correct_param_name(model.model, config.model.inner_params[0], logger=LOG)
    # if validated_param != config.model.inner_params[0]:
    #     config.model.inner_params[0] = validated_param
    #     LOG.info(f"Using validated parameter: {validated_param}")
    
    # Load datasets
    train_dataset = get_dataset(config, split="train")
    test_dataset = get_dataset(config, split="test")
    
    with_rationale = getattr(config.experiment, 'with_rationale', False)
    task = getattr(config.experiment, 'task', 'mc')
    
    # Setup dataloaders
    train_dataset.set_dataloader(
        task=task,
        with_rationale=with_rationale,
        shuffle_choices=True if task in ("mc", "mci") else False,
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    test_dataset.set_dataloader(
        task=task,
        with_rationale=with_rationale,
        shuffle_choices=False,
        batch_size=config.batch_size,
        shuffle=False,
    )
    
    LOG.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
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
    
    LOG.info("Starting finetuning...")
    total_batches = len(train_dataset.loader)
    for batch_idx, batch in enumerate(train_dataset.loader):
        tokens = model.prepare_training_batch(batch)
        
        # For history-based editors, collect batches first before training
        if needs_history and len(batch_history) < prefill_size:
            batch_history.append(tokens)
            if len(batch_history) == prefill_size:
                LOG.info(f"Pre-populated batch_history with {len(batch_history)} batches for {editor_name}")
            continue  # Skip editing until we have enough history
        
        # Edit (finetune) on this batch
        editor.edit(config, tokens, batch_history)
        
        # Track history for EWC/retrain methods
        batch_history.append(tokens)
        max_history = getattr(config.editor, 'fisher_mem', 10) if hasattr(config.editor, 'fisher_mem') else \
                      getattr(config.editor, 'retrain_memory', 100) if hasattr(config.editor, 'retrain_memory') else 10
        if len(batch_history) > max_history:
            batch_history = batch_history[-max_history:]
        
        # Track losses
        if hasattr(editor, 'losses') and editor.losses:
            losses.extend(editor.losses)
        
        # Periodic logging every 20 batches
        if (batch_idx + 1) % 20 == 0:
            recent_losses = losses[-20:] if len(losses) >= 20 else losses
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0
            LOG.info(f"Batch {batch_idx + 1}/{total_batches}, Avg loss (last 20): {avg_loss:.4f}")
    
    LOG.info(f"Finetuning complete. Total batches: {total_batches}")
    
    # Evaluation on test set
    model.model.eval()
    LOG.info("Evaluating on test set...")
    
    for batch in test_dataset.loader:
        test_dataset.task_generate(batch, model)
    
    # Compute metrics
    metrics = test_dataset.task_engineer.eval(test_dataset)
    LOG.info(f"Test metrics: {metrics}")
    
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
        LOG.info(f"Saved checkpoint: {ckpt_path}")
    
    # Explicit cleanup to free GPU memory before script exits
    del model
    del editor
    del train_dataset
    del test_dataset
    torch.cuda.empty_cache()
    LOG.info("Cleaned up model and freed GPU memory")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Finetuning")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--editor", type=str, required=True, 
                        choices=["ft", "ft_ewc", "ft_retrain"], help="Editor method")
    parser.add_argument("--model_name", type=str, default=None, 
                        help="Model name: 'qwen3', 'llava', 'blip'")
    parser.add_argument("--inner_params", type=str, nargs='+', default=[], 
                        help="Layer to finetune (auto-selected if empty)")
    parser.add_argument("--dataset_name", type=str, required=True, 
                        choices=["aokvqa", "fvqa"], help="Dataset name")
    parser.add_argument("--task", type=str, default=None, 
                        choices=["mc", "mci", "qa"], help="Task type (uses config.yaml if not provided)")
    parser.add_argument("--with_rationale", action="store_true", 
                        help="Include rationale in prompts (uses config.yaml if not provided)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--n_iter", type=int, default=1, help="Inner iterations per batch")
    parser.add_argument("--ckpt_dir", type=str, default=None, 
                        help="Directory to save checkpoints (overrides config.yaml)")
    
    args = parser.parse_args()
    
    # Create config
    cfg_path = args.config or os.path.join(
        os.path.dirname(__file__), "config", "config.yaml"
    )
    
    ns = argparse.Namespace(
        config=cfg_path,
        editor=args.editor,
        inner_params=args.inner_params if args.inner_params else [],
        dataset_name=args.dataset_name,
        model_name=args.model_name,
    )
    
    config = configure_args(ns, config_path=cfg_path)
    
    # Override settings
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.batch_size = args.batch_size
    config.n_iter = args.n_iter
    if args.ckpt_dir is not None:
        config.ckpt_dir = args.ckpt_dir
    # Only override task from CLI if explicitly provided (otherwise uses config.yaml)
    if args.task is not None:
        config.experiment.task = args.task
    # Only override with_rationale from CLI if flag is explicitly provided
    # If flag not provided, config.yaml value (or default False) will be used
    if args.with_rationale:
        config.experiment.with_rationale = True
    
    # Run finetuning
    metrics = finetune(config)
    print(f"\nFinal Results: {metrics}")

