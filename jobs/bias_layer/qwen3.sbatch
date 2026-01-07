#!/bin/bash
#SBATCH --job-name=bias_layer_qwen3
#SBATCH -p gpu
#SBATCH --gres=gpu:1  
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 2:00:00
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

# BiasLayer: compute layer-wise bias, saves to results/bias_layer/
python -m revlm.run.bias_layer --model_name qwen3 --n_runs 10 --pool_method mean
python -m revlm.run.bias_layer --model_name qwen3 --n_runs 10 --pool_method last
