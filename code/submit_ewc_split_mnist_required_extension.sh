#!/bin/bash
#SBATCH --job-name=vcl_ewc_experiments
#SBATCH --output=/scratch/shared/beegfs/user/experiments/required-extension/logs/%A_%a.out
#SBATCH --error=/scratch/shared/beegfs/user/experiments/required-extension/logs/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=low-prio-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-14  # 3 lambd values * 5 seeds = 15 combinations

# Create logs directory if it doesn't exist
EXPERIMENT_PATH="/scratch/shared/beegfs/user/experiments/required-extension"
mkdir -p $EXPERIMENT_PATH

# Initialize 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vcl

# Define lambd values and seed values
declare -a LAMBD_VALUES=(1.0 10.0 100.0)
declare -a SEEDS=(17 137 256 42 0)

# Calculate lambd index and seed index based on SLURM_ARRAY_TASK_ID
LAMBDA_IDX=$((SLURM_ARRAY_TASK_ID / 5))  # Integer division by number of seeds
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))   # Remainder for the seed index

LAMBDA=${LAMBD_VALUES[$LAMBDA_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Method and model configuration
METHOD="ewc"
MODEL="discriminative_naive"

echo "Running experiment with method: $METHOD, model: $MODEL, lambd: $LAMBDA, seed: $SEED"

# Run the experiment for the ewc method with varying lambd and seed
python experiment.py \
    --model $MODEL \
    --method $METHOD \
    --benchmark split_mnist \
    --experiment-path $EXPERIMENT_PATH \
    --device 0 \
    --seed $SEED \
    --required_extension \
    --lambd $LAMBDA
