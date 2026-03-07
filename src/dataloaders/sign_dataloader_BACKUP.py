"""
Generic Sign Language DataLoader
Supports pose-based sign language translation
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


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
        num_keypoints: int = 266,   # ✅ FIXED (133 × 2)
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

        # Load CSV
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} samples from {self.data_path}")

        # Labels
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

    def _load_pose(self, uid: str) -> Optional[np.ndarray]:
        pose_path = self.pose_dir / f"{uid}.pkl"

        if not pose_path.exists():
            logger.warning(f"Pose file not found: {pose_path}")
            return None

        try:
            with open(pose_path, "rb") as f:
                pose_dict = pickle.load(f)  # ✅ dict

            keypoints = self._extract_keypoints(pose_dict)

            # Temporal downsampling
            keypoints = keypoints[:: self.step_frames]

            # Noise augmentation
            if self.add_noise:
                keypoints += np.random.normal(0, self.noise_std, keypoints.shape)

            # Pad / truncate
            keypoints = self._pad_or_truncate(keypoints, self.max_frames)

            return keypoints

        except Exception as e:
            logger.error(f"Error loading pose {uid}: {e}")
            return None

    def _extract_keypoints(self, pose_dict: dict) -> np.ndarray:
        """
        Expected pose_dict:
        - keypoints: (T, 1, 133, 2) OR list equivalent
        """

        if "keypoints" not in pose_dict:
            raise ValueError("Pose file missing 'keypoints' field")

        # ✅ Convert list → ndarray ALWAYS
        keypoints = np.asarray(pose_dict["keypoints"], dtype=np.float32)

        if keypoints.ndim != 4:
            raise ValueError(f"Unexpected keypoints shape: {keypoints.shape}")

        # Remove person dimension
        keypoints = keypoints[:, 0]  # (T, 133, 2)

        # Flatten (x, y)
        keypoints = keypoints.reshape(keypoints.shape[0], -1)  # (T, 266)

        return keypoints

    def _pad_or_truncate(self, sequence: np.ndarray, max_len: int) -> np.ndarray:
        T, D = sequence.shape

        if T > max_len:
            return sequence[:max_len]

        if T < max_len:
            pad = np.zeros((max_len - T, D), dtype=np.float32)
            return np.vstack([sequence, pad])

        return sequence

    def _create_attention_mask(self, sequence: np.ndarray) -> np.ndarray:
        return (~np.all(sequence == 0, axis=1)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        uid = row.get("uid", row.get("video_path", str(idx)))

        pose_sequence = self._load_pose(uid)

        if pose_sequence is None:
            pose_sequence = np.zeros(
                (self.max_frames, self.num_keypoints), dtype=np.float32
            )

        attention_mask = self._create_attention_mask(pose_sequence)
        labels = self.labels[idx]

        return {
            "input_ids": torch.from_numpy(pose_sequence),
            "attention_mask": torch.from_numpy(attention_mask),
            "labels": torch.LongTensor(labels),
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
        data_path=train_path,
        pose_dir=pose_dir,
        tokenizer=tokenizer,
        add_noise=True,
        **dataset_kwargs,
    )

    val_dataset = SignLanguageDataset(
        data_path=val_path,
        pose_dir=pose_dir,
        tokenizer=tokenizer,
        add_noise=False,
        **dataset_kwargs,
    )

    test_dataset = SignLanguageDataset(
        data_path=test_path,
        pose_dir=pose_dir,
        tokenizer=tokenizer,
        add_noise=False,
        **dataset_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Created dataloaders: "
        f"Train={len(train_dataset)}, "
        f"Val={len(val_dataset)}, "
        f"Test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
