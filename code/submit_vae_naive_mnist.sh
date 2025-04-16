#!/bin/bash
#SBATCH --job-name=vcl_experiments
#SBATCH --output=/scratch/shared/beegfs/user/experiments/naive/vae/%A_%a.out
#SBATCH --error=/scratch/shared/beegfs/user/experiments/naive/vae/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=low-prio-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-8


# Define parameter arrays
SEEDS=(123 17 42)
BATCH_SIZES=(32 64 128)

# Calculate the indices for the current array job
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 3))
BATCH_SIZE_IDX=$((SLURM_ARRAY_TASK_ID % 3))

# Extract the actual parameters
SEED=${SEEDS[$SEED_IDX]}
BATCH_SIZE=${BATCH_SIZES[$BATCH_SIZE_IDX]}

# Fixed parameters
DEVICE="0"
METHOD="naive"
BENCHMARK="mnist"
EXPERIMENT_PATH="/scratch/shared/beegfs/user/experiments/naive/vae"

mkdir -p $EXPERIMENT_PATH


echo "SEED=$SEED and BATCH_SIZE=$BATCH_SIZE"

# Initialize 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vcl

# Run the experiment
python experiment.py \
  --model generative_naive \
  --method $METHOD \
  --benchmark $BENCHMARK \
  --experiment-path $EXPERIMENT_PATH \
  --device $DEVICE \
  --seed $SEED \
  --batch_size $BATCH_SIZE