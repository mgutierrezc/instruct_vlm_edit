#!/bin/bash
# Submit cot_pred jobs for all models x datasets
set -euo pipefail
cd "$(dirname "$0")"

for DATA in aokvqa fvqa; do
  for MODEL in blip llava qwen qwen_4b; do
    echo "Submitting cot_pred: ${DATA}/${MODEL}"
    sbatch "${DATA}/${MODEL}.sbatch"
  done
done
