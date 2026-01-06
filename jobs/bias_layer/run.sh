#!/bin/bash
# Submit all bias_layer jobs

cd "$(dirname "$0")"
mkdir -p log

sbatch blip.sbatch
sbatch llava.sbatch
sbatch qwen3.sbatch
sbatch qwen3_4b.sbatch

echo "Submitted 4 bias_layer jobs"

