#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_all_rvqa_reports_and_analysis.sh [options]

Submits RVQA independent report jobs for all 4 models and all 4 iterations
(original, iterfix_1, iterfix_2, iterfix_3), checks report completeness, and
optionally waits until all folders have 1000 reports before running analysis.

Options:
  --submit              Submit all report jobs. Default if no action is given.
  --wait                Poll report counts until all expected folders reach 1000.
  --analyze             Run scripts/analyze_rvqa_results.py.
  --all                 Submit, wait, then analyze.
  --check-only          Only print report counts.
  --poll-seconds N      Poll interval for --wait. Default: 300.
  --mem-gb N            Override Slurm memory for all submitted jobs.
  --analysis-output P   Analysis output path.
  --help                Show this help.

Common examples:
  # Submit everything, then return.
  scripts/run_all_rvqa_reports_and_analysis.sh

  # Submit everything, wait for 1000 reports in every folder, then analyze.
  scripts/run_all_rvqa_reports_and_analysis.sh --all

  # Only check completeness.
  scripts/run_all_rvqa_reports_and_analysis.sh --check-only

  # Run analysis after jobs are already complete.
  scripts/run_all_rvqa_reports_and_analysis.sh --analyze
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-/sfs/gpfs/tardis/home/xxxxx/.conda/envs/revlm/bin/python}"
REPORT_ROOT="${REPORT_ROOT:-$REPO_ROOT/indep_runs/temp_reports}"
EVAL_DF="${EVAL_DF:-/scratch/xxxxx/eval_df.pkl}"
WRONG_EDIT_DF="${WRONG_EDIT_DF:-/scratch/xxxxx/rvqa_wrong_edit_df.pkl}"
CORRECT_EDIT_DF="${CORRECT_EDIT_DF:-/scratch/xxxxx/rvqa_correct_edit_df.pkl}"
NAMESPACE_CONFIG="${NAMESPACE_CONFIG:-$REPO_ROOT/revlm/config/config.yaml}"
ANALYSIS_OUTPUT="${ANALYSIS_OUTPUT:-$REPO_ROOT/all_4_models_originals_iterfixes_rvqa_analysis.txt}"

DO_SUBMIT=0
DO_WAIT=0
DO_ANALYZE=0
CHECK_ONLY=0
POLL_SECONDS=300
MEM_GB=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)
      DO_SUBMIT=1
      shift
      ;;
    --wait)
      DO_WAIT=1
      shift
      ;;
    --analyze)
      DO_ANALYZE=1
      shift
      ;;
    --all)
      DO_SUBMIT=1
      DO_WAIT=1
      DO_ANALYZE=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --mem-gb)
      MEM_GB="$2"
      shift 2
      ;;
    --analysis-output)
      ANALYSIS_OUTPUT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$DO_SUBMIT" -eq 0 && "$DO_WAIT" -eq 0 && "$DO_ANALYZE" -eq 0 && "$CHECK_ONLY" -eq 0 ]]; then
  DO_SUBMIT=1
fi

cd "$REPO_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable is not available or executable: $PYTHON" >&2
  echo "Set PYTHON=/path/to/env/bin/python and rerun." >&2
  exit 1
fi

if [[ ! -f "$EVAL_DF" || ! -f "$WRONG_EDIT_DF" || ! -f "$CORRECT_EDIT_DF" ]]; then
  echo "Missing one or more RVQA input dataframe files:" >&2
  echo "  EVAL_DF=$EVAL_DF" >&2
  echo "  WRONG_EDIT_DF=$WRONG_EDIT_DF" >&2
  echo "  CORRECT_EDIT_DF=$CORRECT_EDIT_DF" >&2
  exit 1
fi

RVQA_INDICES="$(
  grep '^[[:space:]]*+indices:' config/indep_runs/rvqa_sample_qwen3_4b.yaml \
    | sed 's/^[^:]*:[[:space:]]*//' \
    | tr -d ' '
)"

MODELS=(
  "qwen3_4b:qwen3_4b"
  "qwen3_8b:qwen3"
  "llava:llava"
  "blip:blip"
)

SUFFIXES=(
  "original:"
  "iterfix_1:1"
  "iterfix_2:2"
  "iterfix_3:3"
)

expected_run_names() {
  local spec model_label model_name suffix_spec suffix ignored
  for spec in "${MODELS[@]}"; do
    IFS=: read -r model_label model_name <<< "$spec"
    for suffix_spec in "${SUFFIXES[@]}"; do
      IFS=: read -r suffix ignored <<< "$suffix_spec"
      printf '%s_%s\n' "$model_label" "$suffix"
    done
  done
}

count_reports() {
  local run_name="$1"
  find "$REPORT_ROOT/$run_name" -maxdepth 1 -type f -name 'report_*.xlsx' 2>/dev/null | wc -l
}

print_counts() {
  local run_name count
  expected_run_names | while read -r run_name; do
    count="$(count_reports "$run_name")"
    printf '%5d %s\n' "$count" "$run_name"
  done | sort -n
}

all_complete() {
  local run_name count
  while read -r run_name; do
    count="$(count_reports "$run_name")"
    if [[ "$count" -ne 1000 ]]; then
      return 1
    fi
  done < <(expected_run_names)
  return 0
}

submit_rvqa_run() {
  local model_label="$1"
  local model_name="$2"
  local suffix="$3"
  local max_fix_iters="${4:-}"
  local output_prefix="$REPORT_ROOT/${model_label}_${suffix}/report_"
  local mode_args=()
  local mem_args=()

  mkdir -p "$(dirname "$output_prefix")"

  if [[ "$suffix" == "original" ]]; then
    mode_args=(+indep_mode=original)
  else
    mode_args=(+indep_mode=iterfix +max_fix_iters="$max_fix_iters")
  fi

  if [[ -n "$MEM_GB" ]]; then
    mem_args=(hydra.launcher.mem_gb="$MEM_GB")
  fi

  echo "Submitting ${model_label}_${suffix}..."
  "$PYTHON" main_gpu.py -m hydra/launcher=submitit_slurm \
    +run_name=independent_run \
    +log_filename=independent_run \
    +repo_root="$REPO_ROOT" \
    +eval_df_path="$EVAL_DF" \
    +wrong_edit_path="$WRONG_EDIT_DF" \
    +correct_edit_path="$CORRECT_EDIT_DF" \
    +output_path="$output_prefix" \
    +editor_name=ike_chain \
    +model_name="$model_name" \
    +dataset_name=aokvqa \
    +task=mc \
    +indices="$RVQA_INDICES" \
    +namespace_config_path="$NAMESPACE_CONFIG" \
    "${mode_args[@]}" \
    "${mem_args[@]}"
}

submit_all() {
  local spec suffix_spec model_label model_name suffix max_fix_iters
  for spec in "${MODELS[@]}"; do
    IFS=: read -r model_label model_name <<< "$spec"
    for suffix_spec in "${SUFFIXES[@]}"; do
      IFS=: read -r suffix max_fix_iters <<< "$suffix_spec"
      submit_rvqa_run "$model_label" "$model_name" "$suffix" "$max_fix_iters"
    done
  done
}

wait_for_completion() {
  echo "Waiting for all expected report folders to reach 1000 files."
  while true; do
    print_counts
    if all_complete; then
      echo "All expected report folders have 1000 reports."
      return 0
    fi
    echo "Not complete yet. Sleeping ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
  done
}

run_analysis() {
  echo "Running RVQA analysis -> $ANALYSIS_OUTPUT"
  "$PYTHON" scripts/analyze_rvqa_results.py \
    --reports-root "$REPORT_ROOT" \
    --wrong-edit-path "$WRONG_EDIT_DF" \
    --output-path "$ANALYSIS_OUTPUT"
}

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  print_counts
  exit 0
fi

if [[ "$DO_SUBMIT" -eq 1 ]]; then
  submit_all
fi

if [[ "$DO_WAIT" -eq 1 ]]; then
  wait_for_completion
fi

if [[ "$DO_ANALYZE" -eq 1 ]]; then
  run_analysis
fi
