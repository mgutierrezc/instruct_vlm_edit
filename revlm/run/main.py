import argparse
import copy
import logging
import os
from tqdm import tqdm
from time import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .models import get_model
from .dataset import get_dataset, get_tokenize_fn
from .editors import get_editor
from .metrics import get_metric, get_error_fn
from .config_utils import configure_args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
LOG = logging.getLogger(__name__)


def main(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    LOG.info(f"Starting experiment with config: {config}")
    
    # --- Load Model ---
    model = get_model(config)
    model = model.to(device)
    
    # --- Load Dataset ---
    edit_dataset = get_dataset(config)
    
    # --- Get loaders ---
    edit_loader = DataLoader(edit_dataset, batch_size=config.batch_size, shuffle=True)
     
    # --- Define task-specific functions ---
    metric = get_metric(config.experiment.task)
    is_error = get_error_fn(config.experiment.task)
    tokenize = get_tokenize_fn(config.experiment.task)
    
    LOG.info(f"Loaded {len(edit_loader)} candidate edits.")
    
    # --- Load editor ---
    editor = get_editor(config, model)
    editor.generate = model.model.generate if hasattr(model, 'model') else model.generate
    
    # --- Begin editing ---
    unedited_model = copy.deepcopy(model)
    n_edits = 0
    batch_history = []
    
    for i, batch in tqdm(enumerate(edit_loader)):
        tokens = tokenize(batch, editor.tokenizer, device)
        
        # Check if edit is needed and within limit
        if is_error(editor, tokens) and (n_edits <= config.max_n_edits):
            n_edits += 1
            batch_history.append(tokens)
            
            # --- Perform edit ---
            start = time()
            editor.edit(config, tokens, batch_history)
            total_time = time() - start
            
            # --- Compute and log metrics ---
            log_dict = {}
            with torch.no_grad():
                ES = metric(editor, tokens)
                
                if i == 0:
                    ERR = ES
                
                # Periodically compute historical metrics
                if i > 0 and n_edits % 250 == 0:
                    ERR = torch.tensor([metric(editor, tokens) for tokens in batch_history]).nanmean()  
                    ES = ES.item() if isinstance(ES, torch.Tensor) else ES
                    log_dict["ERR"] = ERR.item() if isinstance(ERR, torch.Tensor) else ERR
                    log_dict["ES"] = ES
                    log_dict["train_time"] = total_time
                    log_dict["edit"] = batch.get("text", "") if isinstance(batch, dict) else ""
                    log_dict["edit_label"] = batch.get("label", "") if isinstance(batch, dict) else ""
                
                log_dict["n_edits"] = n_edits
                
                if hasattr(editor, "log_dict"):
                    log_dict.update(editor.log_dict)
                
                LOG.info(f"Edit {n_edits}: {log_dict}")
    
    # --- Save model if requested ---
    if config.ckpt_dir:
        os.makedirs(config.ckpt_dir, exist_ok=True)
        editor_name = config.editor._name if hasattr(config, "editor") and hasattr(config.editor, "_name") else getattr(config, "editor", "editor")
        ckpt_path = os.path.join(config.ckpt_dir, f"model_edited_{editor_name}.pt")
        torch.save(editor.model.state_dict(), ckpt_path)
        LOG.info(f"Saved edited model to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Model Editing")
    # most settings come from YAML file
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (CLI overrides YAML)")
    
    # CLI overrides
    parser.add_argument("--editor", type=str, required=True, choices=["ft", "ft_ewc", "ft_retrain", "mend", "grace", "rome", "memory", "defer"], help="Editor method to use")
    parser.add_argument("--model_name", type=str, default=None, help="Short VLM name to map to full HF id (e.g., 'qwen3', 'qwen3_4b', 'llava', 'blip')")
    parser.add_argument("--inner_params", type=str, nargs='+', default=[], help="Model parameters to edit (overrides YAML if provided)")
    parser.add_argument("--dataset_name", type=str, default="", help="Dataset name (overrides YAML if provided)")

    config = parser.parse_args()
    config = configure_args(config, config_path=config.config)
    config.device = device
    main(config)

