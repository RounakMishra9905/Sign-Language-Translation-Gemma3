import torch

# We add weights_only=False to bypass the security check
ckpt = torch.load("checkpoints/gemma_3_4b_isign/checkpoint_epoch1.pt", map_location='cpu', weights_only=False)

print("Keys in checkpoint:", ckpt.keys())

if 'epoch' in ckpt:
    print(f"Saved Epoch: {ckpt['epoch']}")
else:
    print("WARNING: 'epoch' key is MISSING from the checkpoint.")