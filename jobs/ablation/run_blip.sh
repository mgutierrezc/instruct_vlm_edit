#!/usr/bin/env bash
# Submit all ablation jobs for InstructBLIP-7B
set -euo pipefail
cd "$(dirname "$0")"

echo "=========================================="
echo "Submitting InstructBLIP-7B ablation jobs"
echo "=========================================="

for PARAM in cap_keys mode merge_keys pair_rationale_w reject_threshold_pct hubness_keys; do
  SBATCH="aokvqa/$PARAM/blip.sbatch"
  if [[ -f "$SBATCH" ]]; then
    echo "Submitting $SBATCH"
    sbatch "$SBATCH"
  else
    echo "Skipping $PARAM (no blip.sbatch found)"
  fi
done
