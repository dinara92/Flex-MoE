#!/bin/bash

# Parallel Hyperparameter Tuning Script for FlexMoE
# This script runs Optuna tuning for different seeds in parallel across multiple GPUs

# Configuration
N_TRIALS=200
MODALITY="AMD"
MAX_EPOCHS=100
EARLY_STOPPING_PATIENCE=15

# Function to run tuning for a single seed
run_seed() {
    local seed=$1
    local device=$2
    
    echo "Starting seed $seed on GPU $device"
    
    python optuna_tune_single_seed.py \
        --seed $seed \
        --device $device \
        --n_trials $N_TRIALS \
        --max_epochs $MAX_EPOCHS \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
        --modality $MODALITY \
        > logs/optuna_seed_${seed}.log 2>&1
    
    echo "Completed seed $seed on GPU $device"
}

# Create logs directory
mkdir -p logs

# Check available GPUs
echo "Checking available GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Make sure CUDA is installed."
    exit 1
fi

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $NUM_GPUS GPUs"

if [ $NUM_GPUS -eq 0 ]; then
    echo "ERROR: No GPUs found!"
    exit 1
fi

# Run seeds in parallel across available GPUs
echo ""
echo "Starting parallel hyperparameter tuning..."
echo "Configuration:"
echo "  - Number of trials per seed: $N_TRIALS"
echo "  - Modality: $MODALITY"
echo "  - Max epochs per trial: $MAX_EPOCHS"
echo "  - Early stopping patience: $EARLY_STOPPING_PATIENCE"
echo ""

# Launch background processes for each seed
for seed in {0..4}; do
    # Assign GPU in round-robin fashion
    device=$((seed % NUM_GPUS))
    
    # Run in background
    run_seed $seed $device &
    
    # Store process ID
    PIDS[$seed]=$!
    
    # Sleep briefly to stagger starts
    sleep 5
done

echo "All seeds launched. Waiting for completion..."
echo "Monitor progress with: tail -f logs/optuna_seed_*.log"
echo ""

# Wait for all background processes to complete
for seed in {0..4}; do
    wait ${PIDS[$seed]}
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Seed $seed completed successfully"
    else
        echo "✗ Seed $seed failed with exit code $exit_code"
    fi
done

echo ""
echo "=================================="
echo "Hyperparameter tuning complete!"
echo "=================================="
echo ""

# Check if all results were generated
missing_results=0
for seed in {0..4}; do
    if [ ! -f "optuna_results/best_params_seed_${seed}.json" ]; then
        echo "WARNING: Missing results for seed $seed"
        missing_results=$((missing_results + 1))
    fi
done

if [ $missing_results -eq 0 ]; then
    echo "✓ All 5 seeds completed successfully"
    echo ""
    echo "Results summary:"
    python -c "
import json
import numpy as np
from pathlib import Path

results_dir = Path('optuna_results')
val_f1_scores = []

for seed in range(5):
    params_file = results_dir / f'best_params_seed_{seed}.json'
    if params_file.exists():
        with open(params_file, 'r') as f:
            params = json.load(f)
            val_f1_scores.append(params['best_val_f1'])
            print(f'  Seed {seed}: Val F1 = {params[\"best_val_f1\"]:.4f}')

if val_f1_scores:
    print(f'')
    print(f'  Mean ± Std: {np.mean(val_f1_scores):.4f} ± {np.std(val_f1_scores):.4f}')
"
    echo ""
    echo "Next step: Run final training"
    echo "  python final_training.py --device 0 --max_epochs 150"
else
    echo "⚠ Warning: $missing_results seed(s) did not complete"
    echo "Check logs/optuna_seed_*.log for errors"
fi