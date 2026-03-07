#!/bin/bash
# Quick Setup Script - Configure your data paths
# Usage: bash quick_setup.sh

echo "=========================================="
echo "Sign Language Translation - Quick Setup"
echo "=========================================="
echo ""

# Ask for paths
echo "Please provide your data paths:"
echo ""

echo "1. Training CSV file:"
echo "   Example: /DATACSEShare/sanjeet/.../train_split_unicode_filtered_matched.csv"
read -p "   Your path: " TRAIN_PATH

echo ""
echo "2. Validation CSV file:"
echo "   Example: /DATACSEShare/sanjeet/.../val_split_unicode_filtered_matched.csv"
read -p "   Your path: " VAL_PATH

echo ""
echo "3. Test CSV file:"
echo "   Example: /DATACSEShare/sanjeet/.../test_split_unicode_filtered_matched.csv"
read -p "   Your path: " TEST_PATH

echo ""
echo "4. Pose files directory:"
echo "   Example: /DATACSEShare/sanjeet/.../performance/"
read -p "   Your path: " POSE_DIR

echo ""
echo "5. Experiment name (for tracking):"
read -p "   Your name: " EXP_NAME

# Create config from template
CONFIG_FILE="configs/${EXP_NAME}.yaml"

echo ""
echo "Creating config file: ${CONFIG_FILE}"

# Copy template and replace paths
cp configs/TEMPLATE.yaml "${CONFIG_FILE}"

# Use sed to replace paths (works on both Linux and Mac)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|train_path: \"/PATH/TO/YOUR/train.csv\"|train_path: \"${TRAIN_PATH}\"|" "${CONFIG_FILE}"
    sed -i '' "s|val_path: \"/PATH/TO/YOUR/val.csv\"|val_path: \"${VAL_PATH}\"|" "${CONFIG_FILE}"
    sed -i '' "s|test_path: \"/PATH/TO/YOUR/test.csv\"|test_path: \"${TEST_PATH}\"|" "${CONFIG_FILE}"
    sed -i '' "s|pose_dir: \"/PATH/TO/YOUR/POSE/FILES/\"|pose_dir: \"${POSE_DIR}\"|" "${CONFIG_FILE}"
    sed -i '' "s|checkpoint_dir: \"checkpoints/my_experiment\"|checkpoint_dir: \"checkpoints/${EXP_NAME}\"|" "${CONFIG_FILE}"
    sed -i '' "s|run_name: \"my_experiment\"|run_name: \"${EXP_NAME}\"|" "${CONFIG_FILE}"
else
    # Linux
    sed -i "s|train_path: \"/PATH/TO/YOUR/train.csv\"|train_path: \"${TRAIN_PATH}\"|" "${CONFIG_FILE}"
    sed -i "s|val_path: \"/PATH/TO/YOUR/val.csv\"|val_path: \"${VAL_PATH}\"|" "${CONFIG_FILE}"
    sed -i "s|test_path: \"/PATH/TO/YOUR/test.csv\"|test_path: \"${TEST_PATH}\"|" "${CONFIG_FILE}"
    sed -i "s|pose_dir: \"/PATH/TO/YOUR/POSE/FILES/\"|pose_dir: \"${POSE_DIR}\"|" "${CONFIG_FILE}"
    sed -i "s|checkpoint_dir: \"checkpoints/my_experiment\"|checkpoint_dir: \"checkpoints/${EXP_NAME}\"|" "${CONFIG_FILE}"
    sed -i "s|run_name: \"my_experiment\"|run_name: \"${EXP_NAME}\"|" "${CONFIG_FILE}"
fi

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Config file created: ${CONFIG_FILE}"
echo ""
echo "Next steps:"
echo "1. Verify paths in ${CONFIG_FILE}"
echo "2. (Optional) Change model in ${CONFIG_FILE}"
echo "3. Train:"
echo "   bash scripts/train_multi_gpu.sh ${CONFIG_FILE} 4"
echo ""
echo "=========================================="
