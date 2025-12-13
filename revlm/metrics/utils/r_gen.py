import pandas as pd
import os
from huggingface_hub import snapshot_download
import string

def get_r_gen_input(dataset_name):
    """Load caption dataframe from HuggingFace dataset."""
    repo_id = "JJoy333/RationaleVQA"
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["r_gen/qa/*.parquet"],
    )
    r_gen_df = pd.read_parquet(os.path.join(local_root, "r_gen", "qa", f"{dataset_name}.parquet"))
    r_gen_df = to_mc_format(r_gen_df)
    # Derive image_path directly from sid (deterministic path pattern)
    r_gen_df["image_path"] = f"data/r_gen/image/{dataset_name}/" + r_gen_df["sid"].astype(str) + ".png"
    return r_gen_df

def to_mc_format(r_gen_df: pd.DataFrame) -> pd.DataFrame:
    # split "answers" into list of options
    opts = (
        r_gen_df["answers"]
        .astype(str)
        .str.split("|")
        .apply(lambda xs: [x.strip() for x in xs])
    )

    # ground-truth answer = first option
    r_gen_df["answer"] = opts.str[0]

    # all choices in one string, separated by "; "
    r_gen_df["choices"] = opts.apply(lambda xs: "; ".join(xs))

    # "(A) choice1\n(B) choice2\n..." format
    letters = string.ascii_uppercase
    def make_idx(xs):
        return "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(xs))

    r_gen_df["idx_choices"] = opts.apply(make_idx)

    # final columns in the same order as your second dataframe
    wanted_cols = ["uid", "sid", "question",
                   "answer", "rationale", "choices", "idx_choices"]
    # Keep 'reason' column if it exists (contains generated COT for r_gen QA pairs)
    if "reason" in r_gen_df.columns:
        wanted_cols.append("reason")
    return r_gen_df[wanted_cols].copy()

