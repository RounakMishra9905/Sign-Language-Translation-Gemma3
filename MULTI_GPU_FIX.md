# Quick Fix Summary: Multi-GPU Performance Issue

## Problem
Multi-GPU and single-GPU training taking the same time (~25 mins/epoch).

## Solution
File locking in data loading was causing serialization. Fixed by:

1. **Removed file locking** from `src/dataloaders/sign_dataloader.py`
   - Deleted `fcntl.flock()` calls
   - Read-only pickle files don't need locks

2. **Added persistent workers** to DataLoader
   - Workers reused between epochs
   - Less overhead

## Expected Result
- **Before:** 1 GPU = 25 min, 4 GPUs = 25 min ❌
- **After:** 1 GPU = 25 min, 4 GPUs = 6-7 min ✅

## Test It
```bash
# Multi-GPU (should be ~4× faster now)
bash scripts/train_multi_gpu.sh configs/t5_base_isign.yaml 4
```

## Details
See `docs/MULTI_GPU_PERFORMANCE_FIX.md` for complete explanation.

## Files Changed
- `src/dataloaders/sign_dataloader.py` - Removed file locking, added persistent_workers
- `train.py` - Clarified batch size documentation
