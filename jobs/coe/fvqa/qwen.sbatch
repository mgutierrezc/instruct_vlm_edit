#!/bin/bash
#SBATCH --job-name=coe_fvqa_qwen
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH -t 10:00:00
#SBATCH -A cama
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

python -m revlm.run.coe_pred \
  --model_name qwen3 \
  --dataset_name fvqa \
  --task mc

