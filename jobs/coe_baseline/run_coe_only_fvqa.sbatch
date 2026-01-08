#!/bin/bash
#SBATCH --job-name=coe_only_fvqa
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -A netlab
#SBATCH --export=NONE
#SBATCH --output=log/slurm-%x.out
#SBATCH --error=log/slurm-%x.err

set -euo pipefail

module purge
module load cuda/12.8.0
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate revlm

echo "Node: $(hostname)"
nvidia-smi || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd /scratch/jq2uw/MME/instruct_vlm_edit

MODELS=("blip" "llava" "qwen_4b" "qwen3")

for MODEL in "${MODELS[@]}"; do
  echo "=========================================="
  echo "COE only: model=${MODEL}, dataset=fvqa"
  echo "=========================================="
  python -m revlm.run.run_coe_only \
    --model_name "${MODEL}" \
    --dataset_name "fvqa"
done

echo "Done!"

