#!/usr/bin/env bash
# Master script to submit all FVQA ablation study jobs
set -euo pipefail
cd "$(dirname "$0")"

echo "=========================================="
echo "Submitting all FVQA IKE_CHAIN ablation studies"
echo "=========================================="

for PARAM in cap_keys mode merge_keys pair_rationale_w reject_threshold_pct hubness_keys; do
  echo ""
  echo "=== Submitting $PARAM ablation jobs ==="
  (cd "fvqa/$PARAM" && bash run.sh)
done

echo ""
echo "=========================================="
echo "All FVQA ablation jobs submitted!"
echo "=========================================="
