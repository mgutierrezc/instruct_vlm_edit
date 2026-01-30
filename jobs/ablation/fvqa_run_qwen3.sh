#!/usr/bin/env bash
# Submit all FVQA ablation jobs for Qwen3-VL-8B
set -euo pipefail
cd "$(dirname "$0")"

echo "=========================================="
echo "Submitting FVQA Qwen3-VL-8B ablation jobs"
echo "=========================================="

for PARAM in cap_keys mode merge_keys pair_rationale_w reject_threshold_pct hubness_keys; do
  SBATCH="fvqa/$PARAM/qwen3.sbatch"
  if [[ -f "$SBATCH" ]]; then
    echo "Submitting $SBATCH"
    sbatch "$SBATCH"
  else
    echo "Skipping $PARAM (no qwen3.sbatch found)"
  fi
done
