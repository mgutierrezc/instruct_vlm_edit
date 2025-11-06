#!/bin/bash
#SBATCH --job-name=ft_qwen3
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1  
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH -t 36:00:00
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


# Finetune Qwen3 with ft editor (basic finetuning)
MODEL_NAME=qwen3
BATCH_SIZE=5 
export MODEL_NAME BATCH_SIZE

export DATA=aokvqa
bash jobs/ft/main.sh

export DATA=fvqa
bash jobs/ft/main.sh