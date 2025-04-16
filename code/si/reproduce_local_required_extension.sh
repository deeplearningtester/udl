#!/bin/bash

CONFIGS=(
  "split_mnist 0"
  "split_mnist 17"
  "split_mnist 42"
  "split_mnist 137"
  "split_mnist 256"
  "split_mnist 420"
  "split_mnist 676"
  "split_mnist 711"
  "split_mnist 913"
  "split_mnist 1024"
  "permuted_mnist 0"
  "permuted_mnist 17"
  "permuted_mnist 42"
  "permuted_mnist 137"
  "permuted_mnist 256"
  "permuted_mnist 420"
  "permuted_mnist 676"
  "permuted_mnist 711"
  "permuted_mnist 913"
  "permuted_mnist 1024"
)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vcl-tf

LOG_DIR="/scratch/shared/beegfs/user/experiments/si/logs/local"
mkdir -p "$LOG_DIR"

# Run each configuration sequentially
for i in "${!CONFIGS[@]}"; do
  CONFIG="${CONFIGS[$i]}"
  read BENCHMARK SEED <<< "$CONFIG"
  
  echo "Running config $i: benchmark=$BENCHMARK, seed=$SEED"

  python si.py \
    --benchmark "$BENCHMARK" \
    --seed "$SEED" \
    --objective "regression" \
    --experiment_dir "/scratch/shared/beegfs/user/experiments/si/local/required-extension" \
    > "${LOG_DIR}/si_${i}.out" \
    2> "${LOG_DIR}/si_${i}.err"
done
