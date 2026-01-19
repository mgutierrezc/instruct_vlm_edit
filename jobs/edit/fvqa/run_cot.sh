#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"  # run relative to this script's directory

# Submit all editor subfolders
for editor_dir in balancedit_cot grace_cot mend_cot ft_cot; do
  if [ -d "$editor_dir" ] && [ -f "$editor_dir/run.sh" ]; then
    echo "Submitting jobs for editor: $editor_dir"
    cd "$editor_dir"
    bash run.sh
    cd ..
  fi
done

