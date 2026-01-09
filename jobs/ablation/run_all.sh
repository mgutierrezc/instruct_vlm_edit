#!/usr/bin/env bash
# Master script to submit all ablation study jobs
set -euo pipefail
cd "$(dirname "$0")"

echo "=========================================="
echo "Submitting all IKE_CHAIN ablation studies"
echo "=========================================="

for PARAM in cap_keys mode radius_area_pct top_k_patches reject_threshold_pct; do
  echo ""
  echo "=== Submitting $PARAM ablation jobs ==="
  (cd "aokvqa/$PARAM" && bash run.sh)
done

echo ""
echo "=========================================="
echo "All ablation jobs submitted!"
echo "=========================================="

