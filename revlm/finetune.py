import argparse
import logging
import os
import torch
from tqdm import tqdm
import numpy as np

from .models import get_model
from .dataset import get_dataset
from .editors import get_editor
from .config_utils import configure_args

logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
LOG = logging.getLogger(__name__)


def explore_layers(model, top_k=10):
    """Find suggested layer candidates for finetuning"""
    all_names = [n for n, p in model.named_parameters()]
    keywords = ['lm_head', 'embed_out', 'output', 'classifier', 'head', 
                'self_attn.q_proj', 'self_attn.v_proj', 'self_attn.k_proj',
                'mlp.c_fc', 'mlp.c_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    suggestions = []
    for name in all_names:
        for kw in keywords:
            if kw in name.lower():
                suggestions.append(name)
                break
    
    return suggestions[:top_k] if suggestions else all_names[:1]


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
    
    LOG.info("Starting finetuning...")
    for batch_idx, batch in enumerate(tqdm(train_dataset.loader, desc="Training")):
        tokens = model.prepare_training_batch(batch)
        
        # Edit (finetune) on this batch
        editor.edit(config, tokens, batch_history)
        
        # Track history for EWC/retrain methods
        batch_history.append(tokens)
        max_history = getattr(config.editor, 'fisher_mem', 10) if hasattr(config.editor, 'fisher_mem') else 10
        if len(batch_history) > max_history:
            batch_history = batch_history[-max_history:]
        
        # Track losses
        if hasattr(editor, 'losses') and editor.losses:
            losses.extend(editor.losses)
        
        # Periodic logging
        if (batch_idx + 1) % 10 == 0:
            avg_loss = np.mean(losses[-10:]) if losses else 0.0
            LOG.info(f"Batch {batch_idx + 1}/{len(train_dataset.loader)}, Avg loss: {avg_loss:.4f}")
    
    LOG.info(f"Finetuning complete. Total batches: {len(train_dataset.loader)}")
    
    # Evaluation on test set
    model.model.eval()
    LOG.info("Evaluating on test set...")
    
    for batch in tqdm(test_dataset.loader, desc="Evaluating"):
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
    parser.add_argument("--task", type=str, default="mc", 
                        choices=["mc", "mci", "qa"], help="Task type")
    parser.add_argument("--with_rationale", action="store_true", 
                        help="Include rationale in prompts")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--n_iter", type=int, default=1, help="Inner iterations per batch")
    
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
    config.experiment.task = args.task
    config.experiment.with_rationale = args.with_rationale
    
    # Run finetuning
    metrics = finetune(config)
    print(f"\nFinal Results: {metrics}")

