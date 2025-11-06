#!/usr/bin/env bash
# Reusable script to run finetune tasks for a dataset
# Usage: MODEL_NAME=... DATA=... BATCH_SIZE=... bash main.sh
# Requires env: MODEL_NAME, DATA, BATCH_SIZE

set -euo pipefail

python -m revlm.finetune --editor ft --model_name "$MODEL_NAME" --dataset_name "$DATA" --task mc --batch_size "$BATCH_SIZE" || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft --model_name "$MODEL_NAME" --dataset_name "$DATA" --task mc --with_rationale --batch_size "$BATCH_SIZE" || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft --model_name "$MODEL_NAME" --dataset_name "$DATA" --task mci --batch_size "$BATCH_SIZE" || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft --model_name "$MODEL_NAME" --dataset_name "$DATA" --task mci --with_rationale --batch_size "$BATCH_SIZE" || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft --model_name "$MODEL_NAME" --dataset_name "$DATA" --task qa --batch_size "$BATCH_SIZE" || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft --model_name "$MODEL_NAME" --dataset_name "$DATA" --task qa --with_rationale --batch_size "$BATCH_SIZE" || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
