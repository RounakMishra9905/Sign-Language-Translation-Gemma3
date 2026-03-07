#!/bin/bash
# Quick Test Script for Model Swapping
# Usage: bash test_model_swap.sh <config_file>

set -e

CONFIG_FILE=$1

if [ -z "$CONFIG_FILE" ]; then
    echo "❌ Usage: bash test_model_swap.sh <config_file>"
    echo "   Example: bash test_model_swap.sh configs/qwen2.5_7b_instruct_isign.yaml"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "🧪 Testing Model Swap with: $CONFIG_FILE"
echo "=========================================="
echo ""

# Extract model name from config
MODEL_NAME=$(grep "name:" "$CONFIG_FILE" | grep -v "run_name" | grep -v "project_name" | head -1 | awk '{print $2}' | tr -d '"')

echo "📦 Detected Model: $MODEL_NAME"
echo ""

# Create test directory
TEST_DIR="test_model_swap_$(date +%s)"
mkdir -p "$TEST_DIR"

echo "✅ Step 1: Config file loaded successfully"
echo ""

# Test Python imports
echo "🐍 Step 2: Testing Python environment..."
python3 << EOF
import sys
try:
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    print("   ✅ PyTorch version:", torch.__version__)
    print("   ✅ Transformers version:", transformers.__version__)
    print("   ✅ CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("   ✅ CUDA devices:", torch.cuda.device_count())
except ImportError as e:
    print("   ❌ Missing dependency:", str(e))
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Python environment test failed!"
    exit 1
fi
echo ""

# Test model loading (just config, not full model)
echo "🤖 Step 3: Testing model availability..."
python3 << EOF
import sys
from transformers import AutoConfig, AutoTokenizer

model_name = "$MODEL_NAME"

try:
    # Try to load config (lightweight test)
    print(f"   Testing: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    
    # Detect architecture type
    if hasattr(config, 'is_encoder_decoder') and config.is_encoder_decoder:
        arch_type = "Seq2Seq (Encoder-Decoder)"
    else:
        arch_type = "Causal LM (Decoder-Only)"
    
    print(f"   ✅ Model found on HuggingFace Hub")
    print(f"   ✅ Architecture: {arch_type}")
    print(f"   ✅ Model type: {config.model_type}")
    
    # Try tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"   ✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
except Exception as e:
    print(f"   ❌ Error loading model: {str(e)}")
    print(f"   💡 Tip: Check if '{model_name}' exists on HuggingFace Hub")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Model loading test failed!"
    exit 1
fi
echo ""

# Test config parsing
echo "📋 Step 4: Testing config parsing..."
python3 << EOF
import yaml
import sys

try:
    with open("$CONFIG_FILE", 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required fields
    required = ['data', 'model', 'training']
    for key in required:
        if key not in config:
            print(f"   ❌ Missing required section: {key}")
            sys.exit(1)
    
    print("   ✅ Config structure valid")
    print(f"   ✅ Model name: {config['model']['name']}")
    print(f"   ✅ Batch size: {config['training']['batch_size']}")
    print(f"   ✅ Learning rate: {config['training']['learning_rate']}")
    print(f"   ✅ WandB run name: {config['training']['run_name']}")
    
    # Check if LoRA is configured
    if config['model'].get('use_lora', False):
        print(f"   ✅ LoRA enabled (r={config['model']['lora_config']['r']})")
    else:
        print("   ℹ️  LoRA disabled (full fine-tuning)")
    
    # Check quantization
    if config['model'].get('load_in_4bit', False):
        print("   ✅ 4-bit quantization enabled")
    elif config['model'].get('load_in_8bit', False):
        print("   ✅ 8-bit quantization enabled")
    else:
        print("   ℹ️  No quantization (FP16/FP32)")
    
except Exception as e:
    print(f"   ❌ Config parsing error: {str(e)}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Config parsing test failed!"
    exit 1
fi
echo ""

# Cleanup
rm -rf "$TEST_DIR"

echo "=========================================="
echo "✅ ALL TESTS PASSED!"
echo "=========================================="
echo ""
echo "Your config is ready to use! 🎉"
echo ""
echo "To train with this config, run:"
echo "  bash train_multi_gpu.sh $CONFIG_FILE 4"
echo ""
echo "Or for single GPU:"
echo "  bash train_single_gpu.sh $CONFIG_FILE"
echo ""
