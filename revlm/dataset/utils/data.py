import os
import logging
from typing import List, Dict, Optional, Tuple
import re

import pandas as pd
from huggingface_hub import snapshot_download

LOG = logging.getLogger(__name__)


def data_download_parquet_splits(repo_id: str, path_in_repo: str, cache_dir: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Download train/val/test parquet files from a HF dataset directory."""
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{path_in_repo}/*.parquet"],
        cache_dir=cache_dir,
    )
    base_dir = os.path.join(local_root, path_in_repo)
    return {
        split: os.path.join(base_dir, f"{split}.parquet") if os.path.exists(os.path.join(base_dir, f"{split}.parquet")) else None
        for split in ("train", "val", "test")
    }


def data_load_split_df(parquet_path: Optional[str]) -> pd.DataFrame:
    return (
        pd.DataFrame(columns=["image_path", "question", "answer", "rationale", "choices", "idx_choices"]) if parquet_path is None
        else pd.read_parquet(parquet_path)
    )


def data_rows_to_examples(df: pd.DataFrame) -> List[Dict]:
    """Convert a dataframe to trainer-ready dicts.

    Required columns: image_path, question, answer, rationale, choices, idx_choices
    """
    cols = ["image_path", "question", "answer", "rationale", "choices", "idx_choices"]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}")
    if df.empty:
        return []

    records = df[cols].to_dict(orient="records")
    examples: List[Dict] = []
    for r in records:
        ex: Dict[str, object] = {
            "image": r["image_path"],
            "question": r["question"],
            "answer": r["answer"],
            "rationale": r["rationale"],
            "choices": r["choices"],
            "idx_choices": r["idx_choices"],
        }
        examples.append(ex)
    return examples


# Tokenization utilities moved from top-level utils
def tokenize_vlm(batch, tokenizer, device, test=False):
    """
    Tokenize VLM input batch.
    Accepts either:
    - collated batch with keys: 'prompts' (list[str]), 'golds' (list[dict])
    - single example with 'prompt' or 'question', and optional 'answer'/'label'
    """
    # Questions/prompts
    if isinstance(batch, dict):
        if "prompts" in batch:
            questions = batch["prompts"]
        else:
            q = batch.get("prompt", batch.get("question", ""))
            questions = q if isinstance(q, list) else [q]
    else:
        questions = [str(batch)]

    tokens = tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Labels (optional)
    if isinstance(batch, dict):
        labels = None
        if "golds" in batch:
            golds = batch["golds"]
            if isinstance(golds, list):
                labels = [g.get("label") for g in golds]
            elif isinstance(golds, dict):
                labels = [golds.get("label")]
        if labels is None:
            if "label" in batch:
                labels = batch["label"] if isinstance(batch["label"], list) else [batch["label"]]
            elif "answer" in batch:
                labels = batch["answer"] if isinstance(batch["answer"], list) else [batch["answer"]]

        if labels is not None and not test:
            label_tokens = tokenizer(
                labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            tokens["label"] = label_tokens["input_ids"]
            if tokenizer.pad_token_id is not None:
                tokens["label"][tokens["label"] == tokenizer.pad_token_id] = -100
        elif not test:
            tokens["label"] = tokens["input_ids"].clone()
            if tokenizer.pad_token_id is not None:
                tokens["label"][tokens["input_ids"] == tokenizer.pad_token_id] = -100

    tokens = {k: v.to(device) for k, v in tokens.items()}
    return tokens


def get_tokenize_fn(task):
    """Get tokenization function for given task"""
    return tokenize_vlm

