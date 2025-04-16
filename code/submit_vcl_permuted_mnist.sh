#!/bin/bash
#SBATCH --job-name=vcl_experiments
#SBATCH --output=/scratch/shared/beegfs/user/experiments/vcl/custom/%A_%a.out
#SBATCH --error=/scratch/shared/beegfs/user/experiments/vcl/custom/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=low-prio-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-17  # Doubled since we run both balanced and imbalanced for coreset methods

# Create logs directory if it doesn't exist
mkdir -p "/scratch/shared/beegfs/user/experiments/vcl/custom"

# Initialize 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vcl

# Define experiment configurations
declare -a METHODS=("vcl" "vcl+random_coreset" "vcl+k_center_coreset" "random_coreset" "random_coreset" "random_coreset" "random_coreset" "random_coreset" "k_center_coreset")
declare -a CORESET_SIZES=(0 200 200 200 400 1000 2500 5000 200)

# Adjust for doubling array size
IDX=$((SLURM_ARRAY_TASK_ID / 2))
IS_BALANCED=$((SLURM_ARRAY_TASK_ID % 2))

METHOD=${METHODS[$IDX]}
CORESET_SIZE=${CORESET_SIZES[$IDX]}

# Determine if coreset-balanced flag should be added
if [[ "$METHOD" == *"coreset"* ]]; then
    if [[ $IS_BALANCED -eq 1 ]]; then
        CORESET_BALANCED="--coreset-balanced"
        BALANCED_TAG="balanced"
    else
        CORESET_BALANCED=""
        BALANCED_TAG="imbalanced"
    fi
else
    CORESET_BALANCED=""
    BALANCED_TAG="N/A"
fi

echo "Running experiment with method: $METHOD, coreset size: $CORESET_SIZE, coreset balanced: $BALANCED_TAG"

# Run the experiment for the specific method and coreset size
python experiment.py \
    --model discriminative_mean_field \
    --method $METHOD \
    --benchmark permuted_mnist \
    --experiment-path "/scratch/shared/beegfs/user/experiments/vcl/custom/" \
    --device 0 \
    --coreset-size $CORESET_SIZE \
    --seed 17 \
    $CORESET_BALANCED
