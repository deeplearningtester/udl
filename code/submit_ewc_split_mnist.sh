#!/bin/bash
#SBATCH --job-name=vcl_experiments
#SBATCH --output=/scratch/shared/beegfs/user/experiments/vcl/custom/ewc/split-mnist/%A_%a.out
#SBATCH --error=/scratch/shared/beegfs/user/experiments/vcl/custom/ewc/split-mnist/%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=low-prio-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5

# Create logs directory if it doesn't exist
mkdir -p "/scratch/shared/beegfs/user/experiments/vcl/custom/ewc/split-mnist"

# Initialize 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vcl

declare -a SEEDS=(0 17 42 137 256 420)
declare -a SEEDS=(12 512 1024 106 611 768)

# Total number of seeds
NUM_SEEDS=${#SEEDS[@]}

# Get method index and seed index from task ID
METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))

METHOD="ewc"
LAMBD="1"
SEED=${SEEDS[$SEED_IDX]}

python experiment.py \
    --model discriminative_naive \
    --method $METHOD \
    --benchmark split_mnist \
    --experiment-path "/scratch/shared/beegfs/user/experiments/vcl/custom/ewc/split-mnist" \
    --device 0 \
    --lambd $LAMBD \
    --batch_size 40000 \
    --seed $SEED
