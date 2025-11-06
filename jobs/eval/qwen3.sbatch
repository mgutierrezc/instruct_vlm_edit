#!/bin/bash
#SBATCH --job-name=eval_qwen3_test
#SBATCH -p gpu
#SBATCH --gres=gpu:1  
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

# Example command - adjust parameters based on your needs
python -m revlm.eval  --model_name qwen3 --dataset_name fvqa --split test --task mci
python -m revlm.eval  --model_name qwen3 --dataset_name fvqa --split test --task mci --rationale
python -m revlm.eval  --model_name qwen3 --dataset_name fvqa --split test --task mc
python -m revlm.eval  --model_name qwen3 --dataset_name fvqa --split test --task mc --rationale
python -m revlm.eval  --model_name qwen3 --dataset_name fvqa --split test --task qa
python -m revlm.eval  --model_name qwen3 --dataset_name fvqa --split test --task qa --rationale

python -m revlm.eval  --model_name qwen3 --dataset_name aokvqa --split test --task mci
python -m revlm.eval  --model_name qwen3 --dataset_name aokvqa --split test --task mci --rationale
python -m revlm.eval  --model_name qwen3 --dataset_name aokvqa --split test --task mc
python -m revlm.eval  --model_name qwen3 --dataset_name aokvqa --split test --task mc --rationale
python -m revlm.eval  --model_name qwen3 --dataset_name aokvqa --split test --task qa
python -m revlm.eval  --model_name qwen3 --dataset_name aokvqa --split test --task qa --rationale