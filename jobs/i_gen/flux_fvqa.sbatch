#!/bin/bash
#SBATCH --job-name=i_gen_flux_fvqa
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


# generate images with flux for fvqa
python -m revlm.run.i_gen \
    --dataset_name fvqa \
    --model_name flux \
    --num_images 2 \
    --device cuda