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


# def data_rows_to_examples(df: pd.DataFrame) -> List[Dict]:
#     """Convert a dataframe to trainer-ready dicts.

#     Required columns: image_path, question, answer, rationale, choices, idx_choices
#     """
#     cols = ["image_path", "question", "answer", "rationale", "choices", "idx_choices"]
#     missing = set(cols) - set(df.columns)
#     if missing:
#         raise ValueError(f"Parquet missing required columns: {missing}")
#     if df.empty:
#         return []

#     records = df[cols].to_dict(orient="records")
#     examples: List[Dict] = []
#     for r in records:
#         ex: Dict[str, object] = {
#             "image": r["image_path"],
#             "question": r["question"],
#             "answer": r["answer"],
#             "rationale": r["rationale"],
#             "choices": r["choices"],
#             "idx_choices": r["idx_choices"],
#         }
#         examples.append(ex)
#     return examples

