## Running Experiments with tmux (Remote SSH)

Use `tmux` to keep experiments running on the remote server even after closing VS Code or disconnecting SSH.

---

## Start a tmux Session

Create and enter a tmux session:
```bash
tmux new -s slt_exp
```
---

## Clone the Repository
```bash
git clone https://github.com/mrinalchaturvedi27/Sign-Language-Translation-Benchmark
cd Sign-Language-Translation-Benchmark
```
 ## Setup Environment
 ```bash
pip install -r requirements.txt
bash setup.sh
```
## Setup Weights & Biases
Login to Weights & Biases:
```bash
wandb login
```
All experiments automatically log to:

project_name: sign-language-translation
## Repository Structure
```bash
sign-language-translation/
├── configs/               # ONLY configs/ files you should modify
├── src/
│   ├── dataloaders/
│   ├── models/
│   ├── trainers/
│   └── utils/
├── scripts/
├── train.py
├── requirements.txt
└── README.md
```
## Run Your Own Experiment
Create a new config:
```bash
cp configs/qwen2.5_7b_instruct_isign.yaml configs/<yourname>_<model>.yaml
nano configs/<yourname>_<model>.yaml
```
Example:
```bash
cp configs/t5_base_isign.yaml configs/t5_base.yaml
```
## What You Are Allowed to Change
```bash
Model (HuggingFace names only)
model:
  name: "t5-base"
  tokenizer: "t5-base"
  ```
Examples:
```bash
t5-large
Qwen/Qwen2.5-7B-Instruct
google/gemma-7b
meta-llama/Llama-3.1-8B
 ```
Training Parameters
```bash
training:
  num_epochs: 30
  batch_size: 8
  learning_rate: 3e-4

  use_wandb: true
  project_name: "sign-language-translation"
  run_name: "yourname_model_name"   # CHANGE THIS EVERY TIME
  ```
## Important Constants (Do Not Change)
```bash
num_keypoints: 266   # MUST ALWAYS BE 266
 ```
## Batch Size Rule
batch_size × gradient_accumulation_steps = 32
## Run Training (Inside tmux)
Single GPU
```bash
bash scripts/train_single_gpu.sh configs/my_experiment.yaml
```
Multi-GPU
```bash
bash scripts/train_multi_gpu.sh configs/my_experiment.yaml 
```

Detach tmux (Keep Training Running)
```bash
Detach without stopping training:
Ctrl + B, then D
```
You can now safely close VS Code or disconnect SSH.

Reattach to tmux Session
```bash
tmux attach -t slt_exp
```
List tmux Sessions
```bash
tmux ls
```
Kill tmux Session (If Needed)
```bash
tmux kill-session -t slt_exp
```
## Notes
Always run training scripts inside tmux

Closing VS Code does NOT stop training

Only modify YAML files inside configs/

Everything else is fully automated
