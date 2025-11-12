#!/usr/bin/env bash
# Reusable script to run finetune tasks for a dataset
# Usage: MODEL_NAME=... DATA=... BATCH_SIZE=... bash main.sh
# Requires env: MODEL_NAME, DATA, BATCH_SIZE

set -euo pipefail

python -m revlm.finetune --editor ft --model_name "$MODEL_NAME" --dataset_name "$DATA" \
       --split "all" --n_iter 1 --task mc \
       --batch_size "$BATCH_SIZE" --subsample 20 --ckpt_dir "ckpts_new/ft_sim" \
       --overwrite || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
