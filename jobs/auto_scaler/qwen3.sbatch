#!/bin/bash
#SBATCH --job-name=auto_scaler_qwen3
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1  
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 24:00:00
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
echo "GPU:"
nvidia-smi || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd /scratch/jq2uw/MME/instruct_vlm_edit

python -m revlm.run.auto_scaler --model_name qwen3 --n_samples 5
python -m revlm.run.auto_scaler --model_name qwen3 --n_samples 10
python -m revlm.run.auto_scaler --model_name qwen3 --n_samples 20

