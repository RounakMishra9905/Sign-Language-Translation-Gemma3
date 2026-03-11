#!/bin/bash

# Multi-GPU Training Script for Sign Language Translation
# Uses PyTorch DistributedDataParallel (DDP)
# Usage: bash scripts/train_multi_gpu.sh <config_file> <num_gpus>
export HF_HOME=/DATACSEShare/sanjeet/ugp-26/Teja/hf_cache
export TRANSFORMERS_CACHE=/DATACSEShare/sanjeet/ugp-26/Teja/hf_cache
set -e  # Exit on error

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/train_multi_gpu.sh <config_file> [num_gpus]"
    echo "Example: bash scripts/train_multi_gpu.sh configs/t5_base_isign.yaml 3"
    exit 1
fi

CONFIG_FILE=$1
NUM_GPUS=${2:-3}  # Defaults to 3 based on your specific request

echo "=========================================="
echo "Starting Multi-GPU Training"
echo "Config: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="

# 🔑 CRITICAL: make src/ visible to all DDP processes
export PYTHONPATH="$(pwd)"

# --- SPECIFIC GPU SELECTION ---
# You requested to run on GPUs 1, 2, and 3 specifically
export CUDA_VISIBLE_DEVICES=1,2,3

# --- TIMEOUT AND STABILITY FIXES ---
# Extends the NCCL timeout from the default 30 mins to 2 hours (7200s)
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO

# Launch training using torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train.py --config "$CONFIG_FILE"

echo "=========================================="
echo "Training Completed!"
echo "=========================================="