# Setup Guide - Customize Data Paths 📂

**Problem:** Everyone has data in different locations. Your mentor's paths won't work for you!

**Solution:** Simple 3-minute setup to use YOUR paths.

---

## 🚀 Quick Setup (Recommended)

### Method 1: Interactive Script

```bash
# Run the interactive setup
bash quick_setup.sh
```

It will ask you for:
1. Training CSV path
2. Validation CSV path  
3. Test CSV path
4. Pose directory path
5. Experiment name

Then automatically creates your config file! ✨

**Example interaction:**
```
Please provide your data paths:

1. Training CSV file:
   Example: /DATACSEShare/sanjeet/.../train_split_unicode_filtered_matched.csv
   Your path: /DATACSEShare/yourname/data/train.csv

2. Validation CSV file:
   Example: /DATACSEShare/sanjeet/.../val_split_unicode_filtered_matched.csv
   Your path: /DATACSEShare/yourname/data/val.csv

...
```

Result: Creates `configs/yourname.yaml` with YOUR paths!

---

## 📝 Method 2: Manual Setup

### Step 1: Copy Template

```bash
cp configs/TEMPLATE.yaml configs/my_experiment.yaml
```

### Step 2: Edit Paths

Open `configs/my_experiment.yaml` and replace:

**From (Placeholders):**
```yaml
data:
  train_path: "/PATH/TO/YOUR/train.csv"  # ← CHANGE THIS
  val_path: "/PATH/TO/YOUR/val.csv"      # ← CHANGE THIS
  test_path: "/PATH/TO/YOUR/test.csv"    # ← CHANGE THIS
  pose_dir: "/PATH/TO/YOUR/POSE/FILES/"  # ← CHANGE THIS
```

**To (Your actual paths):**
```yaml
data:
  train_path: "/DATACSEShare/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/translationexperiment/tokenization/train_split_unicode_filtered_matched.csv"
  val_path: "/DATACSEShare/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/translationexperiment/tokenization/val_split_unicode_filtered_matched.csv"
  test_path: "/DATACSEShare/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/translationexperiment/tokenization/test_split_unicode_filtered_matched.csv"
  pose_dir: "/DATACSEShare/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/performance/"
```

### Step 3: Update Experiment Name

```yaml
training:
  checkpoint_dir: "checkpoints/my_experiment"  # ← Change this
  run_name: "my_experiment"  # ← Change this (for WandB)
```

### Step 4: Train!

```bash
bash scripts/train_multi_gpu.sh configs/my_experiment.yaml 4
```

---

## 🌟 Method 3: Environment Variables

For quick tests or one-time runs:

```bash
# Set your paths as environment variables
export TRAIN_CSV="/DATACSEShare/yourname/data/train.csv"
export VAL_CSV="/DATACSEShare/yourname/data/val.csv"
export TEST_CSV="/DATACSEShare/yourname/data/test.csv"
export POSE_DIR="/DATACSEShare/yourname/data/poses/"

# Then train (uses env vars automatically)
bash scripts/train_multi_gpu.sh configs/TEMPLATE.yaml 4
```

Or create a script:

```bash
# Create set_my_paths.sh
cat > set_my_paths.sh << 'EOF'
#!/bin/bash
export TRAIN_CSV="/DATACSEShare/yourname/data/train.csv"
export VAL_CSV="/DATACSEShare/yourname/data/val.csv"
export TEST_CSV="/DATACSEShare/yourname/data/test.csv"
export POSE_DIR="/DATACSEShare/yourname/data/poses/"
echo "Paths set successfully!"
EOF

chmod +x set_my_paths.sh

# Use it
source set_my_paths.sh
bash scripts/train_multi_gpu.sh configs/TEMPLATE.yaml 4
```

---

## 📊 Real Examples from Team

### Example 1: Sanjeet's Paths (from your mentor)

```yaml
data:
  train_path: "/DATACSEShare/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/translationexperiment/tokenization/train_split_unicode_filtered_matched.csv"
  val_path: "/DATACSEShare/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/translationexperiment/tokenization/val_split_unicode_filtered_matched.csv"
  test_path: "/DATACSEShare/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/translationexperiment/tokenization/test_split_unicode_filtered_matched.csv"
  pose_dir: "/DATACSEShare/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/performance/"
```

### Example 2: Ashish's Paths

```yaml
data:
  train_path: "/DATA405/ashishu23/SURGE/iSign-videos_v1.1/tokenization/train.csv"
  val_path: "/DATA405/ashishu23/SURGE/iSign-videos_v1.1/tokenization/val.csv"
  test_path: "/DATA405/ashishu23/SURGE/iSign-videos_v1.1/tokenization/test.csv"
  pose_dir: "/DATA405/ashishu23/SURGE/iSign-videos_v1.1/poses/"
```

### Example 3: Your Paths

```yaml
data:
  train_path: "/DATA7/YOUR_USERNAME/project/train.csv"  # ← Fill this in
  val_path: "/DATA7/YOUR_USERNAME/project/val.csv"
  test_path: "/DATA7/YOUR_USERNAME/project/test.csv"
  pose_dir: "/DATA7/YOUR_USERNAME/project/poses/"
```

---

## ✅ Verify Your Setup

### Check if paths exist:

```bash
# Quick check
ls -lh /DATACSEShare/yourname/data/train.csv
ls -lh /DATACSEShare/yourname/data/val.csv
ls -lh /DATACSEShare/yourname/data/test.csv
ls -ld /DATACSEShare/yourname/data/poses/
```

### Test your config:

```bash
# Dry run (checks paths without training)
python train.py --config configs/my_experiment.yaml --validate_only
```

---

## 🎯 What to Change in Configs

### Always Change:
1. **Data paths** (train_path, val_path, test_path, pose_dir)
2. **Experiment name** (run_name, checkpoint_dir)

### Optional Changes:
3. **Model** (change to different model from HuggingFace)
4. **Batch size** (if you have different GPU memory)
5. **Learning rate** (for experimentation)

### Never Change (unless you know what you're doing):
- `num_keypoints: 152` (MediaPipe standard)
- `max_frames: 300` (reasonable default)
- `seed: 42` (for reproducibility)

---

## 🔍 Path Formats

Your mentor's format shows:
```python
train_csv = '/DATACSEShare/sanjeet/...'
POSE_DIR_ISIGN = "/DATACSEShare/sanjeet/.../performance/"
```

In YAML config, use:
```yaml
data:
  train_path: "/DATACSEShare/sanjeet/..."
  pose_dir: "/DATACSEShare/sanjeet/.../performance/"
```

**Key differences:**
- Python: `train_csv` → YAML: `train_path`
- Python: `POSE_DIR_ISIGN` → YAML: `pose_dir`
- Both use same actual path strings!

---

## 🐛 Troubleshooting

### "File not found" error

**Problem:** Path doesn't exist or has typo

**Solutions:**
1. Double-check path with `ls <path>`
2. Make sure you have read permissions
3. Use absolute paths (starting with `/`)
4. Check for extra spaces in YAML

### "Permission denied"

**Problem:** You don't have access to that directory

**Solutions:**
1. Check with `ls -l <path>`
2. Ask admin for permissions
3. Copy data to your own directory

### Paths with spaces

**Problem:** Path contains spaces

**Solution:** Use quotes in YAML:
```yaml
train_path: "/path/with spaces/train.csv"  # ✓ Good
train_path: /path/with spaces/train.csv    # ✗ Bad
```

---

## 👥 Team Workflow

### When you get paths from your mentor:

1. **Receive paths** (like in your screenshot)
   ```
   train_csv = '/DATACSEShare/sanjeet/.../train.csv'
   ```

2. **Copy them to your config**
   ```bash
   cp configs/TEMPLATE.yaml configs/yourname.yaml
   nano configs/yourname.yaml
   # Paste paths
   ```

3. **Train with your config**
   ```bash
   bash scripts/train_multi_gpu.sh configs/yourname.yaml 4
   ```

4. **Share your results** (not your paths!)
   - Share WandB link
   - Share config file (others replace paths)
   - Share BLEU scores

### When working with multiple people:

Everyone has their own config:
```
configs/
  ├── sanjeet_experiment.yaml     # Sanjeet's paths
  ├── ashish_experiment.yaml      # Ashish's paths
  ├── yourname_experiment.yaml    # Your paths
  └── TEMPLATE.yaml               # Template for new people
```

**Important:** Add your personal configs to `.gitignore`:
```bash
# In .gitignore
configs/*_experiment.yaml
configs/my_*.yaml
```

Only commit `TEMPLATE.yaml` to git!

---

## 📋 Quick Reference

### I want to...

**Just get started quickly:**
```bash
bash quick_setup.sh
```

**Copy my mentor's exact paths:**
```bash
cp configs/TEMPLATE.yaml configs/my_exp.yaml
# Edit and paste paths
nano configs/my_exp.yaml
```

**Test on different data temporarily:**
```bash
export TRAIN_CSV="/scratch/test/train.csv"
bash scripts/train_multi_gpu.sh configs/my_exp.yaml 4
```

**Switch between datasets:**
```bash
# Create multiple configs
configs/isign_experiment.yaml
configs/how2sign_experiment.yaml
# Train with specific one
bash scripts/train_multi_gpu.sh configs/isign_experiment.yaml 4
```

---

## ✨ Summary

**For most users:**
1. Run `bash quick_setup.sh`
2. Enter your paths when prompted
3. Train with generated config

**For manual control:**
1. Copy `configs/TEMPLATE.yaml`
2. Replace `/PATH/TO/YOUR/...` with your paths
3. Train

**Remember:** The pipeline doesn't care WHERE your data is, just that the paths in your config point to the right place! 🎯

---

## 📞 Need Help?

- Check if paths exist: `ls -lh <your_path>`
- Validate config: `python train.py --config <your_config> --validate_only`
- See template: `cat configs/TEMPLATE.yaml`
- Ask teammate: "What paths did you use?"

Happy training! 🚀
