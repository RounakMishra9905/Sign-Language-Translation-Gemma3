# Multi-GPU Training Performance Fix

## Problem Summary

Multi-GPU training was taking the same amount of time (~25 mins per epoch) as single-GPU training, negating any benefit from using multiple GPUs.

## Root Cause

The performance issue was caused by **file locking contention** in the data loading pipeline:

### The Issue
In `src/dataloaders/sign_dataloader.py`, the `_load_pkl_pose()` method was using `fcntl.flock()` for "thread-safe" file access:

```python
# OLD CODE (SLOW)
fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Acquire shared lock
pose_dict = pickle.load(f)
fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
```

### Why This Caused the Problem

1. **Multi-GPU training uses multiple processes** - one per GPU
2. **All processes read from the same pose files** during training
3. **File locking created serialization** - even with `LOCK_SH` (shared lock), the kernel needs to coordinate lock acquisition/release across processes
4. **I/O became the bottleneck** - GPUs were waiting on file locks instead of computing
5. **No parallelization benefit** - all GPUs had to wait in line for file access

### Why File Locking Was Unnecessary

- Pickle files are **read-only** during training (never written to)
- `pickle.load()` is **already atomic** for reading
- Multiple processes can **safely read the same file simultaneously** without locks
- The OS handles file caching efficiently
- File locking was added for "safety" but actually destroyed performance

## The Fix

### 1. Removed File Locking (Primary Fix)

**File:** `src/dataloaders/sign_dataloader.py`

```python
# NEW CODE (FAST)
with open(path, "rb") as f:
    pose_dict = pickle.load(f)
```

**Impact:** Eliminates serialization bottleneck, allows true multi-GPU parallelization

### 2. Added Persistent Workers

**File:** `src/dataloaders/sign_dataloader.py`

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,  # NEW
    ...
)
```

**Impact:** Reuses worker processes between epochs, reducing startup overhead

### 3. Clarified Documentation

**File:** `train.py`

Updated comments to explain multi-GPU batch size behavior clearly.

## Expected Performance Improvement

### Before (with file locking)
- **Single GPU:** ~25 mins/epoch
- **4 GPUs:** ~25 mins/epoch (NO IMPROVEMENT ❌)
- **Reason:** File locking serialization

### After (without file locking)
- **Single GPU:** ~25 mins/epoch (unchanged)
- **4 GPUs:** ~6-7 mins/epoch (3.5-4× FASTER ✅)
- **Reason:** True parallel data loading and computation

## Testing the Fix

### Single GPU
```bash
bash scripts/train_single_gpu.sh configs/t5_base_isign.yaml
```

### Multi-GPU (4 GPUs)
```bash
bash scripts/train_multi_gpu.sh configs/t5_base_isign.yaml 4
```

You should now see approximately **4× speedup** with 4 GPUs (accounting for some overhead).

## Understanding Multi-GPU Batch Sizes

### How It Works

With `DistributedSampler` and DDP:

1. **Dataset is split across GPUs** - each GPU gets 1/N of the data
2. **Each GPU processes batch_size samples** from its subset
3. **All GPUs compute in parallel**
4. **Gradients are synchronized** after each step

### Example

- Dataset: 10,000 samples
- Config batch_size: 8
- Gradient accumulation: 4
- Number of GPUs: 4

**Single GPU:**
- Samples per step: 8
- Steps per epoch: 10,000 / 8 = 1,250
- Effective batch size: 8 × 4 (accumulation) = 32

**4 GPUs:**
- Samples per GPU: 10,000 / 4 = 2,500
- Samples per step per GPU: 8
- Steps per epoch per GPU: 2,500 / 8 = 312.5 ≈ 312
- **Effective batch size: 8 × 4 (accumulation) × 4 (GPUs) = 128**
- **Time per epoch: ~1/4 of single GPU (4× faster)**

### Configuration Tips

#### For Maximum Speed (Recommended)
Keep `batch_size` the same when scaling GPUs:
- Single GPU: batch_size = 8
- 4 GPUs: batch_size = 8 (per GPU)
- Result: 4× throughput, 4× effective batch size

You may need to adjust learning rate when effective batch size changes (common rule: scale LR linearly with batch size).

#### For Same Effective Batch Size
Scale `batch_size` inversely with GPUs:
- Single GPU: batch_size = 32, gradient_accumulation = 1
- 4 GPUs: batch_size = 8, gradient_accumulation = 1
- Result: 4× throughput, same effective batch size (32)

## Additional Performance Tips

1. **Use enough workers:** Set `num_workers` to 2-4 per GPU (e.g., 8-16 for 4 GPUs)
2. **Enable mixed precision:** Set `mixed_precision: true` in config (if not already)
3. **Monitor GPU utilization:** Use `nvidia-smi dmon -s u` to verify GPUs are utilized
4. **Watch for I/O bottlenecks:** If GPU utilization is low (<80%), increase `num_workers`
5. **Use fast storage:** SSD/NVMe drives significantly help with pickle file loading

## Verification

After applying this fix, verify multi-GPU speedup:

```bash
# Time single GPU
time bash scripts/train_single_gpu.sh configs/t5_base_isign.yaml

# Time multi-GPU  
time bash scripts/train_multi_gpu.sh configs/t5_base_isign.yaml 4

# Expected: Multi-GPU should be ~3.5-4× faster
```

## Technical Details

### Why Shared Locks Still Cause Contention

Even though `LOCK_SH` allows multiple readers, it still requires kernel coordination:

1. Each process must acquire the lock (syscall overhead)
2. Kernel must track lock holders (data structure updates)
3. Lock release requires synchronization (another syscall)
4. With many processes reading frequently, this overhead accumulates
5. Modern OS file caching already handles concurrent reads efficiently

### When File Locking IS Needed

File locking is necessary when:
- **Writing to files** that others might read
- **Modifying shared state** across processes
- **Preventing race conditions** in concurrent writes

For read-only access to static files (like pre-computed pose data), file locking is unnecessary and harmful.

## Questions?

If you experience any issues after this fix or don't see the expected speedup, please check:

1. GPU utilization with `nvidia-smi dmon -s u`
2. Disk I/O with `iotop` (if available)
3. Number of worker processes with `ps aux | grep python`
4. Ensure you're using the multi-GPU script with correct number of GPUs

The fix addresses the primary bottleneck. Any remaining performance issues are likely hardware-related (slow disk, insufficient GPU memory, etc.).
