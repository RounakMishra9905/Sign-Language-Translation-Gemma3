# Team Collaboration Guide 👥

**How to run experiments with different models WITHOUT touching code**

---

## 🎯 Core Principle

**ONE SIMPLE RULE:** Change model in config YAML, not in code!

```yaml
# This is ALL you need to change
model:
  name: "your-model-here"  # ← ONLY edit this
  tokenizer: "your-model-here"
```

---

## 🚀 Quick Team Workflow

### Step 1: Copy a Template Config

```bash
# Copy existing config as starting point
cp configs/qwen2.5_7b_instruct_isign.yaml configs/my_experiment.yaml
```

### Step 2: Edit ONLY the Model Name

```yaml
# configs/my_experiment.yaml

# Change this line:
model:
  name: "any-huggingface-model"  # ← Your model here
  tokenizer: "any-huggingface-model"
  
# Everything else stays the same!
```

### Step 3: Train

```bash
bash train_multi_gpu.sh configs/my_experiment.yaml 4
```

**That's it! No code changes needed!** ✅

---

## 📋 Team Experiment Examples

### Example 1: Alice Tests T5-Large

```yaml
# configs/alice_t5_large.yaml
model:
  name: "t5-large"
  tokenizer: "t5-large"
  use_lora: false  # Seq2Seq doesn't need LoRA
  
training:
  run_name: "alice_t5_large_experiment"  # WandB tracking
```

```bash
bash train_multi_gpu.sh configs/alice_t5_large.yaml 4
```

### Example 2: Bob Tests Qwen2.5-14B

```yaml
# configs/bob_qwen14b.yaml
model:
  name: "Qwen/Qwen2.5-14B"
  tokenizer: "Qwen/Qwen2.5-14B"
  use_lora: true
  load_in_4bit: true
  
training:
  run_name: "bob_qwen14b_experiment"
```

```bash
bash train_multi_gpu.sh configs/bob_qwen14b.yaml 4
```

### Example 3: Carol Tests Gemma-7B

```yaml
# configs/carol_gemma.yaml
model:
  name: "google/gemma-7b"
  tokenizer: "google/gemma-7b"
  use_lora: true
  load_in_4bit: true
  
training:
  run_name: "carol_gemma_experiment"
```

```bash
bash train_multi_gpu.sh configs/carol_gemma.yaml 4
```

### Example 4: Dave Tests Custom Model

```yaml
# configs/dave_custom.yaml
model:
  name: "bigscience/bloom-7b1"  # ANY HuggingFace model!
  tokenizer: "bigscience/bloom-7b1"
  use_lora: true
  load_in_4bit: true
  
training:
  run_name: "dave_bloom_experiment"
```

```bash
bash train_multi_gpu.sh configs/dave_custom.yaml 4
```

---

## 🔄 Supported Model Types

### ✅ The Pipeline Automatically Handles:

**Seq2Seq Models (Encoder-Decoder):**
- T5, mT5, BART, mBART, Pegasus, M2M100, NLLB
- **Auto-detected** - no special config needed

**Causal LM Models (Decoder-Only):**
- Qwen, Gemma, Llama, Mistral, Phi, Bloom
- **Auto-detected** - no special config needed

**You don't need to specify architecture type!** 🎉

---

## 📝 Model Swap Checklist

When swapping models, change these in your config:

```yaml
model:
  name: "new-model-name"           # ✅ REQUIRED
  tokenizer: "new-model-name"      # ✅ REQUIRED
  
  # Optional (adjust based on model size):
  use_lora: true/false             # ⚙️ true for 7B+
  load_in_4bit: true/false         # ⚙️ true for memory efficiency
  lora_config:                     # ⚙️ adjust if using LoRA
    r: 16
    lora_alpha: 32
    
training:
  run_name: "your_experiment_name" # ✅ REQUIRED (for WandB tracking)
  batch_size: 4                    # ⚙️ adjust based on GPU memory
  learning_rate: 1e-4              # ⚙️ adjust based on model
```

---

## 🎓 Best Practices for Teams

### 1. Use Descriptive Config Names

```bash
# ❌ BAD
configs/test1.yaml
configs/experiment.yaml

# ✅ GOOD
configs/alice_t5_large_baseline.yaml
configs/bob_qwen2.5_14b_lora.yaml
configs/carol_gemma_7b_4bit.yaml
```

### 2. Use Unique WandB Run Names

```yaml
training:
  run_name: "alice_t5_large_jan31"  # Include: name, model, date
  run_name: "bob_qwen14b_lora_jan31"
```

### 3. Document Your Experiment

Add comments to your config:

```yaml
# Alice's experiment - Testing T5-Large as baseline
# Date: 2026-01-31
# Goal: Establish Seq2Seq baseline for comparison

model:
  name: "t5-large"
  # Using default settings for fair comparison
```

### 4. Track Everything in Git

```bash
# Create experiment branch
git checkout -b alice/t5-large-baseline

# Add your config
git add configs/alice_t5_large.yaml

# Commit with details
git commit -m "Add T5-Large baseline experiment config"
```

### 5. Share Results via WandB

Everyone uses same project:
```yaml
training:
  use_wandb: true
  project_name: "sign-language-translation"  # Same for all
  run_name: "unique_name_here"               # Different for each
```

---

## 🔍 Comparing Multiple Models (Team Experiment)

### Scenario: Team wants to compare 5 models

**Setup:**
```bash
# Each person creates their config
alice:  configs/compare_t5_base.yaml
bob:    configs/compare_t5_large.yaml
carol:  configs/compare_qwen7b.yaml
dave:   configs/compare_gemma7b.yaml
eve:    configs/compare_llama8b.yaml
```

**All configs use same data and training params:**
```yaml
# SAME FOR EVERYONE
data:
  train_path: "/DATA7/vaibhav/tokenization/train_split_unicode_filtered.csv"
  # ... same paths

training:
  num_epochs: 30        # Same
  batch_size: 4         # Same (adjusted for model size)
  learning_rate: 1e-4   # Same (adjusted for model type)
  seed: 42              # Same for reproducibility!
  
  project_name: "model_comparison_jan2026"  # Same project
  run_name: "t5_base" / "qwen7b" / etc      # Different names
```

**Only difference:**
```yaml
model:
  name: "different-model-per-person"
```

**Everyone trains:**
```bash
bash train_multi_gpu.sh configs/compare_*.yaml 4
```

**Results automatically compared in WandB!** 📊

---

## 🛠️ Common Team Scenarios

### Scenario 1: Quick Model Swap Mid-Experiment

```bash
# Your current experiment isn't working well
# Want to try different model WITHOUT changing code

# Step 1: Copy existing config
cp configs/current_experiment.yaml configs/new_model_try.yaml

# Step 2: Edit ONLY model name
nano configs/new_model_try.yaml
# Change: name: "old-model" → "new-model"

# Step 3: Train
bash train_multi_gpu.sh configs/new_model_try.yaml 4
```

### Scenario 2: Testing Same Model, Different Hyperparameters

```bash
# Create multiple configs for same model
configs/qwen_lr1e4.yaml   # learning_rate: 1e-4
configs/qwen_lr2e4.yaml   # learning_rate: 2e-4
configs/qwen_lr5e4.yaml   # learning_rate: 5e-4

# All use same model, different LR
# Run all in parallel on different GPUs!
```

### Scenario 3: Ablation Study (LoRA vs Full Fine-tuning)

```yaml
# configs/qwen_full_finetune.yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  use_lora: false  # ← Full fine-tuning

# configs/qwen_lora.yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  use_lora: true   # ← LoRA only
```

---

## 📊 Team Dashboard (WandB)

All experiments automatically tracked:

```
Project: sign-language-translation
│
├─ alice_t5_large_baseline
│  ├─ BLEU-4: 4.2
│  ├─ Training time: 24h
│  └─ Memory: 24GB
│
├─ bob_qwen14b_lora
│  ├─ BLEU-4: 8.5 ⭐
│  ├─ Training time: 26h
│  └─ Memory: 16GB
│
├─ carol_gemma_7b
│  ├─ BLEU-4: 6.8
│  ├─ Training time: 20h
│  └─ Memory: 10GB
│
└─ dave_bloom_7b1
   ├─ BLEU-4: 5.9
   ├─ Training time: 22h
   └─ Memory: 14GB
```

**Compare all runs with one click!**

---

## 🚨 Important Notes

### ✅ What You CAN Change (No Code Modification Needed)

- ✅ Model name
- ✅ Batch size
- ✅ Learning rate
- ✅ LoRA settings
- ✅ Quantization (8-bit, 4-bit)
- ✅ Number of epochs
- ✅ Any training hyperparameter
- ✅ Data paths
- ✅ WandB settings

### ⚠️ What You SHOULD NOT Change (Unless You Know What You're Doing)

- ⚠️ Core Python files (`model_factory.py`, `trainer.py`, etc.)
- ⚠️ Directory structure
- ⚠️ Training script logic

**If you need to change core code, discuss with team first!**

---

## 🎯 Model Recommendation by Team Member

### For Beginners / Quick Tests
```yaml
model:
  name: "t5-base"  # Fast, simple, proven
```

### For Best Quality
```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"  # Current best
  use_lora: true
  load_in_4bit: true
```

### For Research Papers (Comparison)
```yaml
# Need multiple models:
- t5-base (baseline)
- t5-large (Seq2Seq best)
- Qwen/Qwen2.5-7B-Instruct (LLM best)
- Qwen/Qwen2.5-14B (SOTA)
```

---

## 📋 Team Experiment Template

```yaml
# configs/TEMPLATE.yaml
# Copy this and fill in your details!

# Experiment Info (add as comments)
# Team Member: Your Name
# Date: YYYY-MM-DD
# Goal: What are you testing?
# Expected Outcome: What do you hope to find?

# Data Configuration (usually same for everyone)
data:
  train_path: "/DATA7/vaibhav/tokenization/train_split_unicode_filtered.csv"
  val_path: "/DATA7/vaibhav/tokenization/val_split_unicode_filtered.csv"
  test_path: "/DATA7/vaibhav/tokenization/test_split_unicode_filtered.csv"
  pose_dir: "/DATA7/vaibhav/isign/Data/iSign-poses_v1.1/"
  max_frames: 300
  max_length: 128
  step_frames: 5
  num_keypoints: 152

# Model Configuration (CHANGE THIS!)
model:
  name: "your-model-here"        # ← EDIT
  tokenizer: "your-model-here"   # ← EDIT
  
  dropout: 0.1
  freeze_encoder: false
  freeze_decoder: false
  
  use_lora: true                 # ← EDIT based on model size
  load_in_4bit: true             # ← EDIT based on GPU memory
  
  lora_config:                   # ← EDIT if using LoRA
    r: 16
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    lora_dropout: 0.1

# Training Configuration
training:
  num_epochs: 30
  batch_size: 4                  # ← EDIT based on GPU memory
  learning_rate: 1e-4            # ← EDIT based on model
  weight_decay: 0.01
  
  max_grad_norm: 1.0
  gradient_accumulation_steps: 8 # ← EDIT based on batch size
  mixed_precision: true
  warmup_ratio: 0.1
  
  checkpoint_dir: "checkpoints/your_experiment"  # ← EDIT
  save_every: 5
  eval_every: 1
  
  num_beams: 5
  max_gen_length: 128
  
  use_wandb: true
  project_name: "sign-language-translation"
  run_name: "yourname_model_date"  # ← EDIT (unique!)
  
  num_workers: 4

seed: 42  # Keep same for reproducibility
```

---

## ✅ Final Checklist Before Training

- [ ] Copied template config
- [ ] Changed `model.name` to your model
- [ ] Adjusted `use_lora` and `load_in_4bit` for your model
- [ ] Set unique `run_name` for WandB
- [ ] Verified data paths are correct
- [ ] Committed config to git
- [ ] Ready to run: `bash train_multi_gpu.sh configs/your_config.yaml 4`

---

## 💡 Pro Tips

1. **Start small:** Test with 1 epoch first to verify config works
2. **Monitor WandB:** Check if training starts properly
3. **Share configs:** Push to git so team can reproduce
4. **Document surprises:** If model behaves unexpectedly, document it
5. **Compare fairly:** Use same seed (42) for all experiments

---

## 🚀 You're Ready!

The pipeline is designed so **ANYONE can swap models by just editing YAML**.

**No code changes. No Python knowledge needed. Just edit config and train!**

Questions? Check the model guides:
- `README_FINAL.md` - Overview
- `QWEN_MODELS_GUIDE.md` - Qwen models
- `OPENSOURCE_MODELS_GUIDE.md` - All LLMs
- `README_HUGGINGFACE.md` - Seq2Seq models

Happy experimenting! 🎓
