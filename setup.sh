#!/bin/bash

# Automated Setup Script for Sign Language Translation Pipeline
# This script sets up everything you need to start training

set -e  # Exit on error

echo "=========================================="
echo "Sign Language Translation Pipeline Setup"
echo "=========================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if [[ $(echo "$python_version" | cut -d. -f1) -lt 3 ]] || [[ $(echo "$python_version" | cut -d. -f2) -lt 8 ]]; then
    echo "Error: Python 3.8+ required"
    exit 1
fi

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p src/dataloaders
mkdir -p src/models
mkdir -p src/trainers
mkdir -p src/utils
mkdir -p configs
mkdir -p checkpoints
mkdir -p predictions
mkdir -p logs

# Create __init__.py files
touch src/__init__.py
touch src/dataloaders/__init__.py
touch src/models/__init__.py
touch src/trainers/__init__.py
touch src/utils/__init__.py

echo "✓ Directory structure created"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt

echo "✓ Dependencies installed"

# Check CUDA availability
echo ""
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "✓ CUDA available"
    echo "  GPUs detected: $num_gpus"
    for i in $(seq 0 $((num_gpus-1))); do
        gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name($i))")
        echo "  GPU $i: $gpu_name"
    done
else
    echo "⚠ CUDA not available - will use CPU (slow!)"
fi

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x scripts/train_single_gpu.sh
chmod +x scripts/train_multi_gpu.sh

echo "✓ Scripts are executable"

# Verify installation
echo ""
echo "Verifying installation..."

python -c "
import torch
import transformers
import pandas
import numpy
print('✓ All imports successful')
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
"

echo ""
echo "=========================================="
echo "Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit configs/t5_base_isign.yaml with your data paths"
echo "2. Run training:"
echo "   Single GPU:  bash scripts/train_single_gpu.sh configs/t5_base_isign.yaml"
echo "   Multi-GPU:   bash scripts/train_multi_gpu.sh configs/t5_base_isign.yaml 4"
echo ""
echo "Happy training! 🚀"
