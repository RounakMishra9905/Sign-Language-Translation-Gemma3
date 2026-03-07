import torch
import numpy as np
from transformers import AutoTokenizer
from src.dataloaders.sign_dataloader import SignLanguageDataset

DATA_PATH = "/data/dept_share/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/translationexperiment/tokenization/train_split_unicode_filtered_matched.csv"
POSE_DIR = "/data/dept_share/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/performance"

tokenizer = AutoTokenizer.from_pretrained("t5-base")

dataset = SignLanguageDataset(
    data_path=DATA_PATH,
    pose_dir=POSE_DIR,
    tokenizer=tokenizer,
    max_frames=300,
    step_frames=5,
)

print("Dataset length:", len(dataset))

bad = 0
for i in range(10):
    sample = dataset[i]

    x = sample["input_ids"]
    mask = sample["attention_mask"]

    print(f"\nSample {i}")
    print("  input_ids shape:", x.shape)
    print("  attention sum:", mask.sum().item())
    print("  NaNs:", torch.isnan(x).any().item())

    if mask.sum() == 0 or torch.isnan(x).any():
        bad += 1

print("\nBad samples:", bad)

