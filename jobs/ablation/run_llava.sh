#!/usr/bin/env bash
# Submit all ablation jobs for LLaVA-1.5-7B
set -euo pipefail
cd "$(dirname "$0")"

echo "=========================================="
echo "Submitting LLaVA-1.5-7B ablation jobs"
echo "=========================================="

for PARAM in cap_keys mode radius_area_pct pair_rationale_w reject_threshold_pct; do
  SBATCH="aokvqa/$PARAM/llava.sbatch"
  if [[ -f "$SBATCH" ]]; then
    echo "Submitting $SBATCH"
    sbatch "$SBATCH"
  else
    echo "Skipping $PARAM (no llava.sbatch found)"
  fi
done

echo ""
echo "Done!"
