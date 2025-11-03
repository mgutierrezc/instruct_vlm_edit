#!/bin/bash
#SBATCH --job-name=eval_qwen3_train
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=250G
#SBATCH -t 8:00:00
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

# Evaluate Qwen3 on train sets
# Each command runs in a separate Python process which should free memory on exit
# Added explicit cleanup in eval.py for safety
python -m revlm.eval  --model_name qwen3 --dataset_name fvqa --split train --task mci || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.eval  --model_name qwen3 --dataset_name fvqa --split train --task mci --rationale || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.eval  --model_name qwen3 --dataset_name fvqa --split train --task mc || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.eval  --model_name qwen3 --dataset_name fvqa --split train --task mc --rationale || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.eval  --model_name qwen3 --dataset_name fvqa --split train --task qa || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.eval  --model_name qwen3 --dataset_name fvqa --split train --task qa --rationale || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }

python -m revlm.eval  --model_name qwen3 --dataset_name aokvqa --split train --task mci || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.eval  --model_name qwen3 --dataset_name aokvqa --split train --task mci --rationale || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.eval  --model_name qwen3 --dataset_name aokvqa --split train --task mc || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.eval  --model_name qwen3 --dataset_name aokvqa --split train --task mc --rationale || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.eval  --model_name qwen3 --dataset_name aokvqa --split train --task qa || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }
python -m revlm.eval  --model_name qwen3 --dataset_name aokvqa --split train --task qa --rationale || { echo "Command failed, checking GPU memory"; nvidia-smi; exit 1; }