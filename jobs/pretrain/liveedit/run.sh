#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Submit pretraining jobs for all datasets
for dataset_dir in aokvqa fvqa; do
  if [ -d "$dataset_dir" ] && [ -f "$dataset_dir/run.sh" ]; then
    echo "Submitting pretrain jobs for dataset: $dataset_dir"
    cd "$dataset_dir"
    bash run.sh
    cd ..
  fi
done
