#!/bin/bash
# Submit all auto_scaler jobs
# 100 bootstrap runs × 10 samples each

cd "$(dirname "$0")"
mkdir -p log

sbatch blip.sbatch
sbatch llava.sbatch
sbatch qwen3.sbatch
sbatch qwen3_4b.sbatch

echo "Submitted 4 auto_scaler jobs (100 runs × 10 samples each)"

