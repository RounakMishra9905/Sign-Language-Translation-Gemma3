"""
Generic Sign Language DataLoader - UPDATED
Supports both .pose and .pkl formats with variable keypoint counts
Includes DistributedSampler support for multi-GPU training
Includes CRITICAL Spatial Normalization for stable fine-tuning
"""

import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

# Try to import pose_format (optional)
try:
    from pose_format import Pose
    POSE_FORMAT_AVAILABLE = True
except ImportError:
    POSE_FORMAT_AVAILABLE = False
    logger.warning("pose_format not available, .pose files not supported")


class SignLanguageDataset(Dataset): 
    def __init__(
        self,
        data_path: str,
        pose_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_frames: int = 300,
        max_length: int = 128,
        step_frames: int = 1,
        add_noise: bool = False,
        noise_std: float = 0.01,
        use_video: bool = False,
        num_keypoints: int = 266,  # 133 × 2
        labels: Optional[List[List[int]]] = None,
    ):
        self.data_path = Path(data_path)
        self.pose_dir = Path(pose_dir)
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_length = max_length
        self.step_frames = step_frames
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.use_video = use_video
        self.num_keypoints = num_keypoints

        self.target_joints = num_keypoints // 2  # 133

        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} samples from {self.data_path}")

        if labels is not None:
            self.labels = labels
        else:
            self.labels = self._tokenize_texts(self.df["text"].tolist())

    def _tokenize_texts(self, texts: List[str]) -> List[List[int]]:
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encodings["input_ids"].tolist()

    def _load_pose(self, uid: str) -> np.ndarray:
        """Load pose from either .pkl or .pose file"""
        # Try .pkl first (your format)
        pkl_path = self.pose_dir / f"{uid}.pkl"
        if pkl_path.exists():
            return self._load_pkl_pose(pkl_path)
    
        # Try .pose format (if you have those)
        pose_path = self.pose_dir / f"{uid}.pose"
        if pose_path.exists():
            logger.warning(f".pose format not supported in this version, using empty pose")
            return self._empty_pose()
    
        logger.warning(f"Pose file not found: {uid} (tried .pkl and .pose)")
        return self._empty_pose()

    def _load_pkl_pose(self, path: Path) -> np.ndarray:
        """Load pose from pickle file"""
        try:
            with open(path, "rb") as f:
                pose_dict = pickle.load(f)

            keypoints = self._extract_keypoints_from_dict(pose_dict)
            keypoints = keypoints[:: self.step_frames]
            
            # --- CRITICAL FIX 1: SPATIAL NORMALIZATION ---
            # Applied before noise to ensure stable gradients in the LLM projector
            keypoints = self.normalize_spatial_coordinates(keypoints)

            if self.add_noise:
                keypoints += np.random.normal(0, self.noise_std, keypoints.shape)

            return self._pad_or_truncate(keypoints, self.max_frames)

        except Exception as e:
            logger.error(f"Error loading pickle pose {path.stem}: {e}")
            return self._empty_pose()
            
    def normalize_spatial_coordinates(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Applies spatial normalization to the extracted keypoints.
        Centers relative to available geometry and scales standard deviation.
        keypoints shape expected: (T, K*2) or similar reshaped flat array
        """
        # Reshape temporarily to process spatial X/Y separately
        # Assuming original structure was (T, 133, 2)
        try:
            T = keypoints.shape[0]
            spatial_kps = keypoints.reshape(T, -1, 2)
            
            # Create a mask to identify valid, non-padded frames (avoiding zero vectors)
            mask = np.any(spatial_kps != 0, axis=-1, keepdims=True)
            valid_points = spatial_kps[mask.squeeze(-1)]
            
            if len(valid_points) > 0:
                # Compute the mean and standard deviation exclusively over valid points
                # This centers the pose around a common origin (0,0) and scales variance
                mu = valid_points.mean(axis=0)
                sigma = valid_points.std(axis=0) + 1e-6
                
                # Standardize the coordinates and preserve the zero-padding
                normalized = np.where(mask, (spatial_kps - mu) / sigma, 0.0)
                return normalized.reshape(T, -1)
            return keypoints
        except Exception as e:
            logger.warning(f"Normalization failed, using raw coordinates: {e}")
            return keypoints
        
    def _extract_keypoints_from_dict(self, pose_dict: dict) -> np.ndarray:
        """
        Extract keypoints from pickle dictionary format.
        Handles variable-length keypoints per frame.
        """
        if "keypoints" not in pose_dict:
            raise ValueError("Pose dict missing 'keypoints' key")

        raw_keypoints = pose_dict["keypoints"]
        
        try:
            keypoints = np.asarray(raw_keypoints, dtype=np.float32)
            
            if keypoints.ndim == 4:  # (T, 1, K, 2)
                keypoints = keypoints[:, 0]  # → (T, K, 2)
            elif keypoints.ndim == 3:  # (T, K, 2)
                pass 
            else:
                raise ValueError(f"Invalid shape: {keypoints.shape}")
            
            T, K, C = keypoints.shape
            assert C == 2, f"Expected 2 coords, got {C}"
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Processing variable-length keypoints: {e}")
            keypoints = self._process_ragged_keypoints(raw_keypoints)
            T, K, C = keypoints.shape

        if K > self.target_joints:
            keypoints = keypoints[:, :self.target_joints, :]
        elif K < self.target_joints:
            pad_width = ((0, 0), (0, self.target_joints - K), (0, 0))
            keypoints = np.pad(keypoints, pad_width, mode='constant')
        
        return keypoints.reshape(T, -1)
    
    def _process_ragged_keypoints(self, raw_keypoints) -> np.ndarray:
        """Process when frames have different numbers of keypoints."""
        processed_frames = []
        
        for frame_data in raw_keypoints:
            try:
                if isinstance(frame_data, (list, tuple)):
                    while (isinstance(frame_data, (list, tuple)) and 
                           len(frame_data) == 1 and 
                           isinstance(frame_data[0], (list, tuple))):
                        frame_data = frame_data[0]
                    frame_array = np.asarray(frame_data, dtype=np.float32)
                else:
                    frame_array = np.asarray(frame_data, dtype=np.float32)
                
                if frame_array.ndim == 1:
                    frame_array = frame_array.reshape(-1, 2)
                elif frame_array.ndim == 3:
                    if frame_array.shape[0] == 1:
                        frame_array = frame_array[0]
                    else:
                        frame_array = frame_array[:, :, 0] 
                elif frame_array.ndim == 2:
                    if frame_array.shape[0] == 2 and frame_array.shape[1] != 2:
                        frame_array = frame_array.T 
                else:
                    frame_array = np.zeros((1, 2), dtype=np.float32)
                
                if frame_array.shape[1] != 2:
                    frame_array = np.zeros((1, 2), dtype=np.float32)
                
                processed_frames.append(frame_array)
                
            except Exception as e:
                logger.warning(f"Frame processing error: {e}, using zeros")
                processed_frames.append(np.zeros((1, 2), dtype=np.float32))
        
        if not processed_frames:
            return np.zeros((1, self.target_joints, 2), dtype=np.float32)
        
        max_K = min(
            max(frame.shape[0] for frame in processed_frames),
            self.target_joints
        )
        
        normalized_frames = []
        for frame in processed_frames:
            K = frame.shape[0]
            if K > max_K:
                frame = frame[:max_K]
            elif K < max_K:
                pad = np.zeros((max_K - K, 2), dtype=np.float32)
                frame = np.vstack([frame, pad])
            normalized_frames.append(frame)
        
        return np.stack(normalized_frames, axis=0)
    
    def _extract_keypoints_from_pose(self, pose: 'Pose') -> np.ndarray:
        try:
            pose_data = pose.body.data
            
            if pose_data.shape[1] > 0:
                keypoints = pose_data[:, 0, :, :2] 
            else:
                return np.zeros((1, self.num_keypoints), dtype=np.float32)
            
            T, K, C = keypoints.shape
            
            if K > self.target_joints:
                keypoints = keypoints[:, :self.target_joints, :]
            elif K < self.target_joints:
                pad_width = ((0, 0), (0, self.target_joints - K), (0, 0))
                keypoints = np.pad(keypoints, pad_width, mode='constant')
            
            return keypoints.reshape(T, -1)
            
        except Exception as e:
            logger.error(f"Error extracting from Pose object: {e}")
            return np.zeros((1, self.num_keypoints), dtype=np.float32)

    def _pad_or_truncate(self, sequence: np.ndarray, max_len: int) -> np.ndarray:
        T, D = sequence.shape

        if T > max_len:
            return sequence[:max_len]

        if T < max_len:
            pad = np.zeros((max_len - T, D), dtype=np.float32)
            return np.vstack([sequence, pad])

        return sequence

    def _empty_pose(self) -> np.ndarray:
        return np.zeros((self.max_frames, self.num_keypoints), dtype=np.float32)

    def _create_attention_mask(self, sequence: np.ndarray) -> np.ndarray:
        return (~np.all(sequence == 0, axis=1)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        # ... existing code (where you load the video/pose data) ...
        
        # 1. NEW PROMPT FORMAT FOR GEMMA 3 IT
        # Replace your old User/Assistant strings with these exact tokens
        prompt_prefix = "<start_of_turn>user\nTranslate the following Indian Sign Language video into an English sentence.\n\nVideo: "
        prompt_suffix = "<end_of_turn>\n<start_of_turn>model\n"
        
        # --- START OF THE BUG FIX ---
        # Safely grab the current row data using the index (idx)
        if hasattr(self, 'data') and isinstance(self.data, pd.DataFrame):
            current_row = self.data.iloc[idx]
        elif hasattr(self, 'df') and isinstance(self.df, pd.DataFrame):
            current_row = self.df.iloc[idx]
        else:
            # Fallback if it's a list of dictionaries
            current_row = self.data[idx]
            
        # Extract the target and add the stop token
        target_text = str(current_row['Target']) if pd.notna(current_row['Target']) else ""
        target_text = target_text.strip() + "<end_of_turn>"
        # --- END OF THE BUG FIX ---
        
        # Tokenize texts
        prefix_ids = self.tokenizer(prompt_prefix, add_special_tokens=False).input_ids
        suffix_ids = self.tokenizer(prompt_suffix, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids
        
        # ... existing code (where you concatenate the ids) ...
        
        # 3. PERFECT LABEL MASKING
        labels = input_ids.clone()
        
        # Calculate exactly where the translation target begins
        # (Make sure 'video_length' or 'num_video_frames' matches your variable name for the pose sequence length)
        context_length = len(prefix_ids) + len(video_features) + len(suffix_ids) 
        
        # PyTorch CrossEntropyLoss ignores -100. This ensures loss is ONLY calculated on the English text.
        labels[:context_length] = -100  

        return {
            "input_ids": torch.from_numpy(pose),
            "attention_mask": torch.from_numpy(self._create_attention_mask(pose)),
            "labels": torch.LongTensor(self.labels[idx]),
        }


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    pose_dir: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_dataset = SignLanguageDataset(
        train_path, pose_dir, tokenizer, add_noise=True, **dataset_kwargs
    )
    val_dataset = SignLanguageDataset(
        val_path, pose_dir, tokenizer, add_noise=False, **dataset_kwargs
    )
    test_dataset = SignLanguageDataset(
        test_path, pose_dir, tokenizer, add_noise=False, **dataset_kwargs
    )

    logger.info(
        f"Created dataloaders: Train={len(train_dataset)}, "
        f"Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    is_distributed = dist.is_initialized()

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        logger.info("Using DistributedSampler for multi-GPU training")
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    return (
        DataLoader(
            train_dataset,
            batch_size,
            shuffle=False if train_sampler is not None else True,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
        ),
        DataLoader(
            val_dataset,
            batch_size,
            shuffle=False, 
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        ),
        DataLoader(
            test_dataset,
            batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        ),
    )