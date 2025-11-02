#!/bin/bash
#SBATCH --job-name=ft_ewc_blip
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1  
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH -t 24:00:00
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


# Finetune BLIP with ft_ewc editor (Elastic Weight Consolidation)
# Each command runs in a separate Python process which should free memory on exit
# Added explicit cleanup in finetune.py, but adding extra safety here
python -m revlm.finetune --editor ft_ewc --model_name blip --dataset_name aokvqa --task mc --batch_size 2 --n_iter 1 || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft_ewc --model_name blip --dataset_name aokvqa --task mc --with_rationale --batch_size 2 --n_iter 1 || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft_ewc --model_name blip --dataset_name aokvqa --task mci --batch_size 2 --n_iter 1 || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft_ewc --model_name blip --dataset_name aokvqa --task mci --with_rationale --batch_size 2 --n_iter 1 || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft_ewc --model_name blip --dataset_name aokvqa --task qa --batch_size 2 --n_iter 1 || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft_ewc --model_name blip --dataset_name aokvqa --task qa --with_rationale --batch_size 2 --n_iter 1 || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }

python -m revlm.finetune --editor ft_ewc --model_name blip --dataset_name fvqa --task mc --batch_size 2 --n_iter 1 || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft_ewc --model_name blip --dataset_name fvqa --task mc --with_rationale --batch_size 2 --n_iter 1 || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft_ewc --model_name blip --dataset_name fvqa --task mci --batch_size 2 --n_iter 1 || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft_ewc --model_name blip --dataset_name fvqa --task mci --with_rationale --batch_size 2 --n_iter 1 || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft_ewc --model_name blip --dataset_name fvqa --task qa --batch_size 2 --n_iter 1 || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.finetune --editor ft_ewc --model_name blip --dataset_name fvqa --task qa --with_rationale --batch_size 2 --n_iter 1 || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }

