#!/usr/bin/env bash
# Submit all ablation jobs for LLaVA-1.5-7B
set -euo pipefail
cd "$(dirname "$0")"

echo "=========================================="
echo "Submitting LLaVA-1.5-7B ablation jobs"
echo "=========================================="

for PARAM in cap_keys mode merge_keys pair_rationale_w reject_threshold_pct hubness_keys; do
  SBATCH="aokvqa/$PARAM/llava.sbatch"
  if [[ -f "$SBATCH" ]]; then
    echo "Submitting $SBATCH"
    sbatch "$SBATCH"
  else
    echo "Skipping $PARAM (no llava.sbatch found)"
  fi
done
