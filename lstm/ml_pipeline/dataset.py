"""PyTorch Dataset for verification-based behavioral biometrics (triplet training)."""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from data_loader import (
    preprocess_keystrokes,
    preprocess_scrolls,
    extract_imu_stats,
)


class VerificationDataset(Dataset):
    """
    Dataset that returns (anchor, positive, negative) triplets.
    
    - Anchor: a sample from user X
    - Positive: a DIFFERENT sample from user X (same person)
    - Negative: a sample from user Y (different person)
    
    This trains the LSTM to produce embeddings where same-user
    samples are close and different-user samples are far apart.
    """
    
    def __init__(
        self,
        user_samples: Dict[str, List[Dict[str, list]]],
        scalers_dict: Dict | None = None,
        augment: bool = True
    ):
        """
        Args:
            user_samples: {user_id: [{keystrokes, scrolls, imu}, ...]}
                          Each user must have >= 2 samples for positive pairs.
            scalers_dict: {keystrokes: scaler, scrolls: scaler, imu: scaler}
            augment: Whether to apply timing noise augmentation
        """
        self.scalers_dict = scalers_dict or {}
        self.augment = augment
        
        # Only keep users with >= 2 samples (need positive pairs)
        self.user_samples = {
            uid: samples for uid, samples in user_samples.items()
            if len(samples) >= 2
        }
        self.user_ids = list(self.user_samples.keys())
        
        if len(self.user_ids) < 2:
            raise ValueError(f"Need at least 2 users with 2+ samples, got {len(self.user_ids)}")
        
        # Flat index: [(user_id, sample_idx), ...]
        self.index = []
        for uid in self.user_ids:
            for i in range(len(self.user_samples[uid])):
                self.index.append((uid, i))
        
        print(f"VerificationDataset: {len(self.user_ids)} users, {len(self.index)} samples")
    
    def __len__(self) -> int:
        return len(self.index)
    
    def _preprocess(self, sample: Dict[str, list]) -> Dict[str, torch.Tensor]:
        """Preprocess a raw sample into model-ready tensors."""
        ks_scaler = self.scalers_dict.get("keystrokes")
        sc_scaler = self.scalers_dict.get("scrolls")
        imu_scaler = self.scalers_dict.get("imu")
        
        keystrokes = preprocess_keystrokes(sample["keystrokes"], scaler=ks_scaler)
        scrolls = preprocess_scrolls(sample["scrolls"], scaler=sc_scaler)
        imu_stats = extract_imu_stats(sample["imu"], scaler=imu_scaler)
        
        # Add small noise for augmentation (all features are timing-based now)
        if self.augment:
            keystrokes += np.random.normal(0, 0.02, keystrokes.shape).astype(np.float32)
            if scrolls.any():
                scrolls[:, :, 2:] += np.random.normal(0, 0.02, scrolls[:, :, 2:].shape).astype(np.float32)
            if imu_stats.any():
                imu_stats += np.random.normal(0, 0.02, imu_stats.shape).astype(np.float32)
        
        return {
            "keystrokes": torch.from_numpy(keystrokes).float().squeeze(0),  # [8, 2]
            "scrolls": torch.from_numpy(scrolls).float().squeeze(0),        # [20, 6]
            "imu_stats": torch.from_numpy(imu_stats).float().squeeze(0),    # [4]
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a flat dict with anchor/pos/neg tensors.
        Keys: anchor_keystrokes, anchor_scrolls, anchor_imu_stats,
              pos_keystrokes, pos_scrolls, pos_imu_stats,
              neg_keystrokes, neg_scrolls, neg_imu_stats
        """
        anchor_uid, anchor_sidx = self.index[idx]
        
        # Anchor
        anchor = self._preprocess(self.user_samples[anchor_uid][anchor_sidx])
        
        # Positive: different sample from SAME user
        pos_candidates = [i for i in range(len(self.user_samples[anchor_uid])) if i != anchor_sidx]
        pos_sidx = random.choice(pos_candidates)
        positive = self._preprocess(self.user_samples[anchor_uid][pos_sidx])
        
        # Negative: sample from DIFFERENT user
        neg_uid = anchor_uid
        while neg_uid == anchor_uid:
            neg_uid = random.choice(self.user_ids)
        neg_sidx = random.choice(range(len(self.user_samples[neg_uid])))
        negative = self._preprocess(self.user_samples[neg_uid][neg_sidx])
        
        return {
            "anchor_keystrokes": anchor["keystrokes"],
            "anchor_scrolls": anchor["scrolls"],
            "anchor_imu_stats": anchor["imu_stats"],
            "pos_keystrokes": positive["keystrokes"],
            "pos_scrolls": positive["scrolls"],
            "pos_imu_stats": positive["imu_stats"],
            "neg_keystrokes": negative["keystrokes"],
            "neg_scrolls": negative["scrolls"],
            "neg_imu_stats": negative["imu_stats"],
        }


def split_users(
    user_samples: Dict[str, List[Dict[str, list]]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple[Dict[str, List], Dict[str, List]]:
    """
    Split users into train and validation sets.
    
    Splits by USER (not by sample), so validation tests
    on completely unseen users — the real verification task.
    """
    rng = random.Random(seed)
    user_ids = list(user_samples.keys())
    rng.shuffle(user_ids)
    
    split_idx = int(len(user_ids) * train_ratio)
    train_ids = user_ids[:split_idx]
    val_ids = user_ids[split_idx:]
    
    train_data = {uid: user_samples[uid] for uid in train_ids}
    val_data = {uid: user_samples[uid] for uid in val_ids}
    
    return train_data, val_data


if __name__ == "__main__":
    from data_loader import load_raw_enrollment, load_keyrecs_fixed, window_user_data, load_scalers
    from pathlib import Path
    
    root = Path(__file__).parent
    db_path = root / "sentinel_lab.db"
    
    # Load and window DB data
    unified_data = load_raw_enrollment(db_path)
    user_samples = {}
    for uid, data in unified_data.items():
        windows = window_user_data(data)
        if len(windows) >= 2:
            user_samples[uid] = windows
    
    # Load keyrecs
    keyrecs = load_keyrecs_fixed(root / "datasets" / "keyrecs" / "fixed-text.csv")
    for uid, samples in keyrecs.items():
        user_samples[f"kr_{uid}"] = samples
    
    scalers = load_scalers()
    
    train_data, val_data = split_users(user_samples)
    print(f"Train users: {len(train_data)}, Val users: {len(val_data)}")
    
    dataset = VerificationDataset(train_data, scalers, augment=True)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    for k, v in sample.items():
        print(f"  {k}: {v.shape}")
