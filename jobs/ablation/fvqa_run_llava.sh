#!/usr/bin/env bash
# Submit all FVQA ablation jobs for LLaVA-1.5-7B
set -euo pipefail
cd "$(dirname "$0")"

echo "=========================================="
echo "Submitting FVQA LLaVA-1.5-7B ablation jobs"
echo "=========================================="

for PARAM in cap_keys mode merge_keys pair_rationale_w reject_threshold_pct hubness_keys; do
  SBATCH="fvqa/$PARAM/llava.sbatch"
  if [[ -f "$SBATCH" ]]; then
    echo "Submitting $SBATCH"
    sbatch "$SBATCH"
  else
    echo "Skipping $PARAM (no llava.sbatch found)"
  fi
done
