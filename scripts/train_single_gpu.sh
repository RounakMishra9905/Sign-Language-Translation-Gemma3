#!/bin/bash

# Single-GPU Training Script for Sign Language Translation
# Usage: bash scripts/train_single_gpu.sh <config_file>

set -e  # Exit on error

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/train_single_gpu.sh <config_file>"
    echo "Example: bash scripts/train_single_gpu.sh configs/t5_base_isign.yaml"
    exit 1
fi

CONFIG_FILE=$1

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

echo "=========================================="
echo "Starting Single-GPU Training"
echo "Config: $CONFIG_FILE"
echo "=========================================="

# 🔑 Ensure src/ and project root are visible
export PYTHONPATH="$(pwd)"

# --- SPECIFIC GPU SELECTION ---
# Modify this if you want a different GPU
export CUDA_VISIBLE_DEVICES=1

# --- TIMEOUT AND STABILITY SETTINGS ---
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO

# Launch training
python train.py --config "$CONFIG_FILE"

echo "=========================================="
echo "Training Completed!"
echo "=========================================="