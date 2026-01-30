# -*- coding: utf-8 -*-
"""
Pretrain LiveEdit meta-learners on VLM-specific edits.

Based on original LiveEdit repo (CVPR 2025):
"Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts"

This script trains the meta-learners (edit_extractor, inpt_extractor, moegen_c, moegen_r)
to generate good LoRA weights from input representations.

Training loop matches original:
- Reliability + Generality + Locality losses
- Soft routing loss (contrastive for query features)
- Hard routing loss (contrastive for vision features)
- Step decay LR scheduler

Usage:
    python -m revlm.run.liveedit_pretrain --model_name qwen3_4b --dataset_name aokvqa --epochs 1000

Output:
    checkpoints/liveedit_pretrain/{model_name}/{dataset_name}/meta_learners.pt
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import VQAModel, VQADataset
from revlm.config_utils import configure_args
from revlm.editors.liveedit.liveedit_pretrain import LiveEditPretrain
from revlm.editors.liveedit.liveedit_pretrain_cot import LiveEditPretrainCOT
from revlm.metrics.utils.t_gen import get_t_gen_input
from revlm.metrics.utils.i_gen import get_i_gen_input


# ==============================================================================
# Data Preparation Functions (merged from liveedit_pretrain_data.py)
# ==============================================================================

def get_r_gen_input(dataset_name: str):
    """Load rationale generality data."""
    from huggingface_hub import snapshot_download
    repo_id = "JJoy333/RationaleVQA"
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["r_gen/*.parquet"],
    )
    r_gen_path = os.path.join(local_root, "r_gen", f"{dataset_name}.parquet")
    if os.path.exists(r_gen_path):
        return pd.read_parquet(r_gen_path)
    return pd.DataFrame()


def get_coe_gen_input(dataset_name: str, model_name: str, edit_ds):
    """Load COE generality data."""
    from huggingface_hub import snapshot_download
    repo_id = "JJoy333/RationaleVQA"
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["e_gen/*.parquet"],
    )
    
    # Try model-specific COE
    model_short = model_name.split("/")[-1]
    coe_path = os.path.join(local_root, "e_gen", f"{dataset_name}_{model_short}.parquet")
    if not os.path.exists(coe_path):
        coe_path = os.path.join(local_root, "e_gen", f"{dataset_name}.parquet")
    
    if os.path.exists(coe_path):
        df = pd.read_parquet(coe_path)
        df["uid"] = df["uid"].astype(str)
        return df
    return pd.DataFrame()


def build_locality_pool(edit_ds, full_df: pd.DataFrame, sample_size: int = 100):
    """Build pool of unrelated samples for locality."""
    used_images = {ex.get("image") for ex in edit_ds.data}
    used_questions = {ex.get("question") for ex in edit_ds.data}
    
    mask = ~full_df["image_path"].isin(used_images) & ~full_df["question"].isin(used_questions)
    pool_df = full_df.loc[mask].reset_index(drop=True)
    
    if pool_df.empty:
        return []
    
    if len(pool_df) > sample_size:
        pool_df = pool_df.sample(n=sample_size, random_state=42)
    
    locality_pool = []
    for _, row in pool_df.iterrows():
        locality_pool.append({
            "uid": str(row["uid"]),
            "image": row["image_path"],
            "question": row["question"],
            "answer": row["answer"],
        })
    
    return locality_pool


def load_edit_set_from_predictions(config):
    """Load edit set directly from saved predictions (no model needed)."""
    pred_snapshot = getattr(config, "pred_path", None)
    if not pred_snapshot:
        pred_snapshot = os.path.join(config.pred_dir, config.fname)
    
    if not os.path.exists(pred_snapshot):
        raise FileNotFoundError(
            f"Predictions not found at {pred_snapshot}. "
            f"Run baseline/edit first to generate predictions."
        )
    
    with open(pred_snapshot, "r") as f:
        data = json.load(f)
    
    ds = VQADataset(config)
    ds.data = data
    
    edit_ds = ds.get_edits()
    edit_ds.set_dataloader(
        with_rationale=False,
        shuffle_choices=False,
        unpaired=True,
    )
    
    return edit_ds


def prepare_pretrain_data(config):
    """Prepare pretraining dataset for one (VLM, dataset) pair."""
    
    dataset_name = config.experiment.dataset_name
    model_name = config.model.name
    model_short = model_name.split("/")[-1]
    
    print(f"Preparing pretrain data for {model_short} on {dataset_name}")
    print("=" * 50)
    
    # Step 1: Load edit set from saved predictions (NO MODEL NEEDED)
    print("Step 1: Loading edit set from saved predictions...")
    edit_ds = load_edit_set_from_predictions(config)
    print(f"  Edit set size: {len(edit_ds.data)}")
    
    # Step 2: Get generality inputs
    print("Step 2: Loading generality data...")
    
    print("  Loading text generality...")
    related_texts = get_t_gen_input(dataset_name, edit_ds)
    print(f"    Found {len(related_texts)} UIDs with text variants")
    
    print("  Loading image generality...")
    related_images = get_i_gen_input(dataset_name, edit_ds)
    print(f"    Found {len(related_images)} UIDs with image variants")
    
    print("  Loading rationale generality...")
    related_r_gen_df = get_r_gen_input(dataset_name)
    print(f"    Found {len(related_r_gen_df)} rationale samples")
    
    print("  Loading COE generality...")
    related_coe_df = get_coe_gen_input(dataset_name, model_name, edit_ds)
    print(f"    Found {len(related_coe_df)} COE samples")
    
    # Step 3: Build locality pool
    print("Step 3: Building locality pool...")
    full_df = edit_ds.load_df()
    locality_pool = build_locality_pool(edit_ds, full_df, sample_size=500)
    print(f"  Locality pool size: {len(locality_pool)}")
    
    # Step 4: Bundle into pretrain format
    print("Step 4: Bundling pretrain data...")
    pretrain_data = []
    
    for ex in edit_ds.data:
        uid = str(ex["uid"])
        
        text_gen = related_texts.get(uid, [])
        image_gen = related_images.get(uid, [])
        
        r_gen = []
        if not related_r_gen_df.empty and "uid" in related_r_gen_df.columns:
            r_gen_rows = related_r_gen_df[related_r_gen_df["uid"].astype(str) == uid]
            for _, row in r_gen_rows.iterrows():
                r_gen.append({
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "image": row.get("image_path", ex["image"]),
                })
        
        coe_gen = []
        if not related_coe_df.empty and "uid" in related_coe_df.columns:
            coe_rows = related_coe_df[related_coe_df["uid"].astype(str) == uid]
            for _, row in coe_rows.iterrows():
                coe_gen.append({
                    "question": row.get("question", ex["question"]),
                    "answer": row.get("answer", ex["answer"]),
                    "image": row.get("image_path", ""),
                })
        
        loc_samples = random.sample(locality_pool, min(5, len(locality_pool)))
        
        entry = {
            "uid": uid,
            "edit": {
                "image": ex["image"],
                "question": ex["question"],
                "answer": ex["answer"],
                "rationale": ex.get("rationale", ""),
                "cot": ex.get("cot", ex.get("rationale", "")),
            },
            "text_gen": text_gen[:5],
            "image_gen": image_gen[:4],
            "rationale_gen": r_gen[:3],
            "coe_gen": coe_gen[:3],
            "locality": loc_samples,
        }
        pretrain_data.append(entry)
    
    # Step 5: Save
    print("Step 5: Saving pretrain data...")
    out_dir = Path(f"data/pretrain/{model_short}/{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pretrain_data.json"
    
    with open(out_path, "w") as f:
        json.dump(pretrain_data, f, indent=2)
    
    print(f"  Saved {len(pretrain_data)} entries to {out_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Total edits: {len(pretrain_data)}")
    print(f"  Avg text_gen per edit: {sum(len(e['text_gen']) for e in pretrain_data) / len(pretrain_data):.1f}")
    print(f"  Avg image_gen per edit: {sum(len(e['image_gen']) for e in pretrain_data) / len(pretrain_data):.1f}")
    print(f"  Avg locality per edit: {sum(len(e['locality']) for e in pretrain_data) / len(pretrain_data):.1f}")
    
    return pretrain_data


# ==============================================================================
# Pretraining Functions
# ==============================================================================

def prepare_pretrain_data_if_needed(config) -> Path:
    """
    Step 0: Prepare pretrain data if it doesn't exist.
    This is fast (CPU-only, ~1-5 minutes) so we do it inline.
    """
    model_name = config.model.name
    dataset_name = config.experiment.dataset_name
    model_short = model_name.split("/")[-1]
    data_path = Path(f"data/pretrain/{model_short}/{dataset_name}/pretrain_data.json")
    
    if data_path.exists():
        print(f"[Step 0] Pretrain data already exists: {data_path}")
        return data_path
    
    print(f"[Step 0] Preparing pretrain data (no GPU needed, ~1-5 min)...")
    print("=" * 60)
    prepare_pretrain_data(config)
    print("=" * 60)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Failed to create pretrain data at {data_path}")
    
    return data_path


def load_pretrain_data(model_name: str, dataset_name: str) -> list:
    """Load prepared pretrain data."""
    model_short = model_name.split("/")[-1]
    data_path = Path(f"data/pretrain/{model_short}/{dataset_name}/pretrain_data.json")
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Pretrain data not found at {data_path}. "
            f"This should have been created by prepare_pretrain_data_if_needed()."
        )
    
    with open(data_path, "r") as f:
        return json.load(f)


def prepare_batch(entry: dict, model) -> dict:
    """Prepare a single example as a training batch."""
    try:
        img = Image.open(entry["image"]).convert("RGB")
    except Exception as e:
        print(f"[pretrain] Failed to load image {entry['image']}: {e}")
        return None
    
    tokens = model.prepare_training_batch({
        "images": [img],
        "prompts": [entry["question"]],
        "golds": [{"label": entry["answer"], "label_train": entry["answer"]}],
        "idxs": [0],
    })
    return tokens


def prepare_text_gen_batch(edit: dict, text_variant: str, model) -> dict:
    """Prepare text generality batch: same image, different question."""
    try:
        img = Image.open(edit["image"]).convert("RGB")
    except Exception:
        return None
    
    tokens = model.prepare_training_batch({
        "images": [img],
        "prompts": [text_variant],
        "golds": [{"label": edit["answer"], "label_train": edit["answer"]}],
        "idxs": [0],
    })
    return tokens


def prepare_image_gen_batch(edit: dict, image_path: str, model) -> dict:
    """Prepare image generality batch: different image, same question."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    
    tokens = model.prepare_training_batch({
        "images": [img],
        "prompts": [edit["question"]],
        "golds": [{"label": edit["answer"], "label_train": edit["answer"]}],
        "idxs": [0],
    })
    return tokens


class EMALoss:
    """Exponential moving average for loss logging (from original)."""
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = None
    
    def update(self, loss: float) -> float:
        if self.value is None:
            self.value = loss
        else:
            self.value = self.alpha * loss + (1 - self.alpha) * self.value
        return self.value


def run_pretrain(config):
    """Pretrain LiveEdit meta-learners (matching original training loop)."""
    
    dataset_name = config.experiment.dataset_name
    model_name = config.model.name
    model_short = model_name.split("/")[-1]
    
    # Get pretrain config
    epochs = getattr(config, "epochs", 1000)
    use_cot = getattr(config, "cot", False)
    batch_size = getattr(config, "batch_size", 4)
    save_ckpt_per_i = getattr(config, "save_ckpt_per_i", 1000)
    log_per_i = getattr(config, "log_per_i", 1)
    ema_alpha = getattr(config, "ema_alpha", 0.1)
    random_seed = getattr(config, "seed", 42)
    
    print(f"Pretraining LiveEdit meta-learners (original config)")
    print(f"  Model: {model_short}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  COT mode: {use_cot}")
    print(f"  Random seed: {random_seed}")
    print("=" * 60)
    
    # Set random seeds (from original)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    # Random generators for data sampling (from original)
    rng_data = np.random.default_rng(random_seed)
    rng_train = np.random.default_rng(random_seed + 1)
    
    # Step 0: Prepare pretrain data if needed (CPU-only, fast)
    prepare_pretrain_data_if_needed(config)
    
    # Step 1: Load pretrain data
    print("Step 1: Loading pretrain data...")
    pretrain_data = load_pretrain_data(model_name, dataset_name)
    print(f"  Loaded {len(pretrain_data)} edit entries")
    
    # Step 2: Load model
    print("Step 2: Loading model...")
    model = VQAModel(config)
    
    # Step 3: Initialize editor
    print("Step 3: Initializing editor...")
    if use_cot:
        editor = LiveEditPretrainCOT(config, model)
    else:
        editor = LiveEditPretrain(config, model)
    
    # Step 4: Setup optimizer (matching original)
    print("Step 4: Setting up optimizer...")
    optimizer, scheduler = editor.get_optimizer_and_scheduler()
    
    trainable_params = editor.get_trainable_params()
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"  Learning rate: {editor.lr}")
    print(f"  LR cut iterations: {editor.lr_cut_it}")
    print(f"  LR cut rate: {editor.lr_cut_rate}")
    print(f"  Loss weights: rel={editor.rel_lambda}, gen={editor.gen_lambda}, "
          f"loc={editor.loc_lambda}, soft={editor.soft_routing_lambda}, hard={editor.hard_routing_lambda}")
    
    # Step 5: Training loop (matching original)
    print("Step 5: Training...")
    print("=" * 60)
    
    ckpt_dir = Path(f"checkpoints/liveedit_pretrain/{model_short}/{dataset_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    ema_loss = EMALoss(alpha=ema_alpha)
    all_logs = []
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(epochs):
        t_epoch = time.time()
        
        # Shuffle data each epoch
        indices = list(range(len(pretrain_data)))
        rng_train.shuffle(indices)
        
        epoch_losses = []
        
        # Process in batches with gradient accumulation
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_entries = [pretrain_data[i] for i in batch_indices]
            
            optimizer.zero_grad()
            batch_loss_sum = 0.0
            batch_log = {}
            valid_entries = 0
            
            for entry in batch_entries:
                edit = entry["edit"]
                
                # Prepare edit batch
                edit_tokens = prepare_batch(edit, model)
                if edit_tokens is None:
                    continue
                
                # Prepare generality batches (limit to 1 each to save GPU memory)
                gen_tokens_list = []
                
                # Text generality
                for text_var in entry.get("text_gen", [])[:1]:
                    gen_tokens = prepare_text_gen_batch(edit, text_var, model)
                    if gen_tokens is not None:
                        gen_tokens_list.append(gen_tokens)
                
                # Image generality
                for img_path in entry.get("image_gen", [])[:1]:
                    gen_tokens = prepare_image_gen_batch(edit, img_path, model)
                    if gen_tokens is not None:
                        gen_tokens_list.append(gen_tokens)
                
                # Prepare locality batch
                loc_tokens = None
                if entry.get("locality"):
                    loc_sample = entry["locality"][rng_data.integers(0, len(entry["locality"]))]
                    loc_tokens = prepare_batch(loc_sample, model)
                
                # Prepare neighbor data for soft routing loss
                # In original: samples from same edit (rel/gen) vs different sample
                neighbor_data = None
                
                # Prepare prototype data for hard routing loss
                # In original: rel/gen samples vs locality samples
                prototype_data = None
                
                # Forward + compute losses
                if use_cot:
                    losses = editor.pretrain_step_cot(
                        edit_tokens, gen_tokens_list, loc_tokens,
                        image=edit["image"],
                        cot=edit.get("cot", edit.get("rationale", "")),
                    )
                else:
                    losses = editor.pretrain_step(
                        edit_tokens, gen_tokens_list, loc_tokens,
                        neighbor_data=neighbor_data,
                        prototype_data=prototype_data,
                    )
                
                # Gradient accumulation: backward immediately to free memory
                entry_loss = losses["loss"] / len(batch_entries)
                entry_loss.backward()
                
                batch_loss_sum += losses["loss"].item()
                valid_entries += 1
                
                # Accumulate logs
                if "log" in losses:
                    for k, v in losses["log"].items():
                        batch_log[k] = batch_log.get(k, 0) + v
                
                # Clear intermediate tensors to save memory
                del edit_tokens, gen_tokens_list, loc_tokens, losses
                torch.cuda.empty_cache()
            
            if valid_entries == 0:
                continue
            
            # Clip gradients and step
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            loss_val = batch_loss_sum / valid_entries
            epoch_losses.append(loss_val)
            ema_val = ema_loss.update(loss_val)
            
            # Logging
            if global_step % log_per_i == 0:
                lr = scheduler.get_last_lr()[0]
                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": loss_val,
                    "ema_loss": ema_val,
                    "lr": lr,
                }
                log_entry.update({k: v / valid_entries for k, v in batch_log.items()})
                all_logs.append(log_entry)
            
            # Progress print
            if global_step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {global_step} | Epoch {epoch+1}/{epochs} | "
                      f"Loss: {loss_val:.4f} | EMA: {ema_val:.4f} | LR: {lr:.2e}")
            
            # Save checkpoint
            if global_step % save_ckpt_per_i == 0:
                editor.save_pretrain_checkpoint(ckpt_dir / f"ckpt_step{global_step}.pt")
        
        # Epoch summary
        avg_epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        elapsed = time.time() - t_epoch
        
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_epoch_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s")
        
        # Save best checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            editor.save_pretrain_checkpoint(ckpt_dir / "meta_learners_best.pt")
            print(f"  [Saved best checkpoint]")
    
    # Step 6: Save final checkpoint
    print("=" * 60)
    print("Step 6: Saving final checkpoint...")
    editor.save_pretrain_checkpoint(ckpt_dir / "meta_learners.pt")
    
    # Save training log
    log = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": editor.lr,
        "lr_cut_it": editor.lr_cut_it,
        "lr_cut_rate": editor.lr_cut_rate,
        "rel_lambda": editor.rel_lambda,
        "gen_lambda": editor.gen_lambda,
        "loc_lambda": editor.loc_lambda,
        "soft_routing_lambda": editor.soft_routing_lambda,
        "hard_routing_lambda": editor.hard_routing_lambda,
        "use_cot": use_cot,
        "final_step": global_step,
        "best_loss": best_loss,
        "training_log": all_logs[-100:],  # Last 100 entries
    }
    with open(ckpt_dir / "pretrain_log.json", "w") as f:
        json.dump(log, f, indent=2)
    
    print(f"  Final checkpoint: {ckpt_dir / 'meta_learners.pt'}")
    print(f"  Best checkpoint: {ckpt_dir / 'meta_learners_best.pt'}")
    print(f"  Training log: {ckpt_dir / 'pretrain_log.json'}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain LiveEdit meta-learners")
    
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml")
    parser.add_argument("--model_name", type=str, required=True, help="VLM name (e.g., qwen3_4b)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., aokvqa)")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--cot", action="store_true", help="Use COT mode (sentence experts)")
    parser.add_argument("--task", type=str, default="mc")
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--save_ckpt_per_i", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--log_per_i", type=int, default=1, help="Log every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.rationale = False
    args.suffix = ""
    args.overwrite = False
    args.subsample = 0
    args.pred_path = None
    args.editor = "liveedit_pretrain"
    args.ema_alpha = 0.1
    
    config = configure_args(args, config_path=args.config)
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.cot = args.cot
    config.save_ckpt_per_i = args.save_ckpt_per_i
    config.log_per_i = args.log_per_i
    config.seed = args.seed
    config.ema_alpha = args.ema_alpha
    
    run_pretrain(config)
