#!/bin/bash

# Single GPU Training Script for Sign Language Translation
# Usage: bash train_single_gpu.sh <config_file>

set -e  # Exit on error

# Check arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: bash train_single_gpu.sh <config_file>"
    echo "Example: bash train_single_gpu.sh configs/t5_base_isign.yaml"
    exit 1
fi

CONFIG_FILE=$1

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

echo "=========================================="
echo "Starting Single GPU Training"
echo "Config: $CONFIG_FILE"
echo "=========================================="

# Set CUDA device (change if needed)
export CUDA_VISIBLE_DEVICES=0

# Run training
python train.py --config "$CONFIG_FILE"

echo "=========================================="
echo "Training Completed!"
echo "=========================================="
