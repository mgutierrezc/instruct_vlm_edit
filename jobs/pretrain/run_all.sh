#!/usr/bin/env bash
# Master script for LiveEdit pretraining
# 
# Step 1: Prepare data (run first, wait for completion)
# Step 2: Pretrain meta-learners
#
# Usage:
#   # Submit data prep jobs (run first)
#   bash jobs/pretrain/run_all.sh data
#
#   # After data prep completes, submit pretrain jobs
#   bash jobs/pretrain/run_all.sh pretrain
#
#   # Or submit all (pretrain will fail if data not ready)
#   bash jobs/pretrain/run_all.sh all

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-help}"

case "$MODE" in
  data)
    echo "=== Step 1: Submitting data preparation jobs ==="
    cd liveedit_data
    bash run.sh
    cd ..
    echo "Data prep jobs submitted. Wait for completion before running pretrain."
    ;;
  pretrain)
    echo "=== Step 2: Submitting pretraining jobs ==="
    cd liveedit
    bash run.sh
    cd ..
    echo "Pretrain jobs submitted."
    ;;
  all)
    echo "=== Submitting ALL jobs ==="
    echo ""
    echo "Step 1: Data preparation"
    cd liveedit_data
    bash run.sh
    cd ..
    echo ""
    echo "Step 2: Pretraining (will fail if data not ready)"
    cd liveedit
    bash run.sh
    cd ..
    echo ""
    echo "All jobs submitted. Monitor with: squeue -u \$USER"
    ;;
  help|*)
    echo "Usage: bash run_all.sh [data|pretrain|all]"
    echo ""
    echo "  data     - Submit data preparation jobs (Step 1)"
    echo "  pretrain - Submit pretraining jobs (Step 2, requires data)"
    echo "  all      - Submit all jobs"
    echo ""
    echo "Workflow:"
    echo "  1. bash run_all.sh data      # Prepare data"
    echo "  2. Wait for data jobs to complete"
    echo "  3. bash run_all.sh pretrain  # Run pretraining"
    ;;
esac
