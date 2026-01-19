#!/bin/bash
#SBATCH --job-name=edit_fvqa_grace_cot_qwen_4b
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH -A hartvigsen_lab
#SBATCH --export=NONE
#SBATCH --output=log/slurm-%x.out
#SBATCH --error=log/slurm-%x.err

set -euo pipefail

module purge
module load cuda/12.8.0
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate revlm

echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd /scratch/jq2uw/MME/instruct_vlm_edit

MODELS=("qwen3_4b")
DATASETS=("fvqa")
EDITOR="grace_cot"
BATCH_SIZE=1

for MODEL_NAME in "${MODELS[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    echo "Running ${EDITOR}: model=${MODEL_NAME}, dataset=${DATA}"
    python -m revlm.run.edit \
      --editor "${EDITOR}" \
      --model_name "${MODEL_NAME}" \
      --dataset_name "${DATA}" \
      --task mc \
      --batch_size "${BATCH_SIZE}" \
      --overwrite
  done
done
