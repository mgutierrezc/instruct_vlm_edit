#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"  # run relative to this script's directory

DIR="."

# List and submit all .sbatch files (handles weird filenames)
find "$DIR" -maxdepth 1 -type f -name '*.sbatch' -print0 | sort -z | while IFS= read -r -d '' f; do
  echo "Submitting $f"
  sbatch "$f"
done