# Quick Start Guide 🚀

## 1. Setup (One Time)

```bash
# Run setup script
bash setup.sh

# Install dependencies
pip install -r requirements.txt
```

## 2. Configure YOUR Data Paths ⭐ IMPORTANT

**Your mentor's paths won't work for you! Set YOUR paths:**

### Option A: Interactive Setup (Easiest)

```bash
bash quick_setup.sh
# Enter your paths when prompted
# Creates configs/yourname.yaml automatically
```

### Option B: Manual Setup

```bash
# Copy template
cp configs/TEMPLATE.yaml configs/my_experiment.yaml

# Edit and replace /PATH/TO/YOUR/... with your actual paths
nano configs/my_experiment.yaml
```

**Example paths to replace:**
```yaml
# Replace these placeholders:
train_path: "/PATH/TO/YOUR/train.csv"

# With your actual paths:
train_path: "/DATACSEShare/yourname/data/train.csv"
```

See `docs/SETUP_GUIDE.md` for detailed instructions!

## 3. Test Your Config (Optional but Recommended)

```bash
bash scripts/test_model_swap.sh configs/my_experiment.yaml
```

## 4. Train!

**Single GPU:**
```bash
bash scripts/train_single_gpu.sh configs/my_experiment.yaml
```

**Multi-GPU (Recommended):**
```bash
bash scripts/train_multi_gpu.sh configs/my_experiment.yaml 4  # 4 GPUs
```

## 5. Monitor Progress

Check WandB dashboard for real-time metrics!

---

## 📚 Documentation

- **README.md** - Main overview
- **docs/TEAM_WORKFLOW.md** - Team collaboration guide
- **docs/QWEN_MODELS_GUIDE.md** - Qwen models explained
- **docs/OPENSOURCE_MODELS_GUIDE.md** - All LLM models
- **docs/README_HUGGINGFACE.md** - Seq2Seq models

---

## 🎯 Recommended Configs

**Best Overall:**
```bash
bash scripts/train_multi_gpu.sh configs/qwen2.5_7b_instruct_isign.yaml 4
```

**Best Quality:**
```bash
bash scripts/train_multi_gpu.sh configs/qwen2.5_14b_isign.yaml 4
```

**Fastest:**
```bash
bash scripts/train_multi_gpu.sh configs/t5_small_isign.yaml 2
```

Happy training! 🎓
