#!/usr/bin/env bash
# Submit all ablation jobs for Qwen3-VL-4B
set -euo pipefail
cd "$(dirname "$0")"

echo "=========================================="
echo "Submitting Qwen3-VL-4B ablation jobs"
echo "=========================================="

for PARAM in cap_keys mode merge_keys pair_rationale_w reject_threshold_pct hubness_keys; do
  SBATCH="aokvqa/$PARAM/qwen3_4b.sbatch"
  if [[ -f "$SBATCH" ]]; then
    echo "Submitting $SBATCH"
    sbatch "$SBATCH"
  else
    echo "Skipping $PARAM (no qwen3_4b.sbatch found)"
  fi
done
