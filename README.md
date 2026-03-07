

# Sign Language Translation Pipeline
A configuration-driven, production-ready framework for translating sign language pose sequences into spoken language text.

This framework supports both **Seq2Seq** (encoder–decoder) and **Causal LM** (decoder-only) transformer models via HuggingFace—enabling seamless architecture swaps without code changes.



##  Key Highlights
* **Model-agnostic:** Swap architectures by simply editing a YAML file.
* **Robust Dataloader:** Handles ragged, noisy, and inconsistent pose inputs out of the box.
* **Memory-efficient:** Native support for **LoRA** and **4-bit/8-bit quantization**.
* **Scalable:** Built on PyTorch DDP for multi-GPU training.
* **Collaborative:** Integrated with Weights & Biases (WandB) for experiment tracking.
* **Research-ready:** Reproducible configurations with comprehensive logging and safe fallbacks.

---

##  Repository Structure
```text
sign-language-translation/
├── configs/               # Experiment YAMLs (ONE config = ONE experiment)
├── src/
│   ├── dataloaders/       # Sign language pose loaders & processors
│   ├── models/            # Model factory (auto-detects architecture)
│   ├── trainers/          # Training loops (single + multi-GPU)
│   └── utils/             # Metrics, logging, and helpers
├── scripts/               # Utility scripts for multi-GPU execution
├── train.py               # Main entry point
├── requirements.txt
└── README.md
```
##  Environment Setup

We recommend using Conda for better CUDA compatibility.
```
conda create -n signlang python=3.10 -y
conda activate signlang
pip install -r requirements.txt
```
Verify Hardware Acceleration:
```
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```
##  Quick Start
1. Prepare Data

Ensure you have:

    CSV files for train/val/test splits.

    Pose files (.pkl) stored in a central directory.

CSV Format (Required): | uid | text | | :--- | :--- | | 8GOiooYQskQ--12 | the person is walking | | bSs2OKpB2Vc--3 | hello everyone |
2. Pose Data Format

The dataloader searches for {uid}.pkl in your designated pose directory.

Supported Keypoint Structures:

    Fixed-size: (T, K, 2) or (T, 1, K, 2)

    Ragged: List of frames with variable keypoints [(K0, 2), (K1, 2), ...]

    Flattened: [x1, y1, x2, y2, ..., xK, yK]

    [!TIP] Normalization: All inputs are automatically converted to input_ids (max_frames, num_keypoints) and attention_mask (max_frames,). Padding and truncation are handled based on your config.

##  Model Support
Model Category	Recommended Models	Notes
Encoder-Decoder	t5-base, bart-large, mbart-large-50	Best for multilingual & traditional Seq2Seq tasks.
Causal LM	Qwen2.5-7B, Llama-3.1-8B, Mistral-7B	Recommended for high-quality, large-scale translation.
🧪 Training Configuration
All experiment parameters, including the **Experiment Name**, are managed in the **YAML** files.
All experiments are driven by YAML. To change a model, you only need to update the model block:
# YAML
```
# configs/my_experiment.yaml
model:
  name: "t5-base"
  tokenizer: "t5-base"
  
training:
  batch_size: 16
  learning_rate: 3e-4
  use_wandb: true
  project_name: "sign-language-translation"
  run_name: "qwen2.5_7b_lora_isign"  # <--- Change this for every new experiment
```
Execution Commands

# Single GPU:
```
bash scripts/train_single_gpu.sh configs/my_experiment.yaml
```
# Multi-GPU :
```
bash scripts/train_multi_gpu.sh configs/my_experiment.yaml 4
```
##  Experiment Tracking

We use Weights & Biases for collaborative monitoring.

    Run wandb login.

    All results are logged under the sign-language-translation project.

    Compare BLEU scores, loss curves, and GPU utilization across different model architectures in real-time.

 Troubleshooting

    Out of Memory (OOM): Enable load_in_4bit: true in your config or reduce batch_size.

    Corrupted Data: The pipeline is fault-tolerant; missing/corrupted .pkl files will default to zero tensors to prevent training crashes.

    Sanity Check: Run python sanity_check_dataloader.py to verify data integrity before starting long jobs.

    Multi-GPU same speed as single GPU: Ensure multiple GPUs are visible (check CUDA_VISIBLE_DEVICES). Keep batch_size as per-GPU to scale throughput; logs show effective batch size and steps per epoch.

Maintained by: Sign Language Translation Team

Last Updated: February 2026

Status: Stable
