#!/bin/bash
#SBATCH --job-name=i_gen_sd3_aokvqa_3000_6000
#SBATCH -p gpu
#SBATCH --gres=gpu:1  
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH -t 12:00:00
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

# Generate images with sd3 for aokvqa (rows 3000:6000)
python -m revlm.run.i_gen \
    --dataset_name aokvqa \
    --model_name sd3 \
    --num_images 2 \
    --device cuda \
    --start_idx 3000 \
    --end_idx 6000

