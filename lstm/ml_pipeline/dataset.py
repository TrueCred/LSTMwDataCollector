from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from data_loader import SessionSample


@dataclass
class TripletRef:
    anchor_user: str
    anchor_idx: int
    positive_idx: int
    negative_user: str
    negative_idx: int


class BehavioralDataset(Dataset):
    def __init__(
        self,
        user_data: Dict[str, List[SessionSample]],
        user_to_idx: Dict[str, int],
        mode: str = "finetune_all",
        triplets_per_user: int = 50,
        is_training: bool = True,
        seed: int = 42,
    ):
        self.user_data = user_data
        self.user_to_idx = user_to_idx
        self.mode = mode
        self.triplets_per_user = triplets_per_user
        self.is_training = is_training
        self.rng = np.random.default_rng(seed)

        self.users = self._eligible_users(sorted(user_data.keys()))
        if len(self.users) < 2:
            raise ValueError(f"Need at least 2 users for mode={mode} triplet training.")

        self.fixed_triplets: List[TripletRef] = []
        if not is_training:
            self.fixed_triplets = self._build_fixed_triplets()

    def _eligible_users(self, users: List[str]) -> List[str]:
        out = []
        for u in users:
            sessions = self.user_data[u]
            if len(sessions) < 2:
                continue

            if self.mode == "pretrain_keys":
                if any(s.has_keystroke for s in sessions):
                    out.append(u)
            elif self.mode == "pretrain_scrolls":
                if any(s.has_scroll for s in sessions):
                    out.append(u)
            elif self.mode == "finetune_all":
                if sessions[0].source == "booth" and any(s.has_keystroke and s.has_scroll and s.has_imu for s in sessions):
                    out.append(u)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

        return out

    def _sample_triplet(self) -> TripletRef:
        anchor_user = self.rng.choice(self.users)
        sessions = self.user_data[anchor_user]

        idxs = self.rng.choice(len(sessions), size=2, replace=len(sessions) < 2)
        anchor_idx, positive_idx = int(idxs[0]), int(idxs[1])

        neg_candidates = [u for u in self.users if u != anchor_user]
        negative_user = self.rng.choice(neg_candidates)
        negative_idx = int(self.rng.integers(0, len(self.user_data[negative_user])))

        return TripletRef(
            anchor_user=anchor_user,
            anchor_idx=anchor_idx,
            positive_idx=positive_idx,
            negative_user=negative_user,
            negative_idx=negative_idx,
        )

    def _build_fixed_triplets(self) -> List[TripletRef]:
        refs: List[TripletRef] = []
        for u in self.users:
            for _ in range(self.triplets_per_user):
                refs.append(self._sample_triplet())
        return refs

    def __len__(self) -> int:
        if self.is_training:
            return len(self.users) * self.triplets_per_user
        return len(self.fixed_triplets)

    def _ref_at(self, idx: int) -> TripletRef:
        if self.is_training:
            return self._sample_triplet()
        return self.fixed_triplets[idx]

    def __getitem__(self, idx: int):
        ref = self._ref_at(idx)

        a = self.user_data[ref.anchor_user][ref.anchor_idx]
        p = self.user_data[ref.anchor_user][ref.positive_idx]
        n = self.user_data[ref.negative_user][ref.negative_idx]

        return {
            "anchor_keys": torch.tensor(a.keystrokes, dtype=torch.float32),
            "anchor_scrolls": torch.tensor(a.scrolls, dtype=torch.float32),
            "anchor_imu": torch.tensor(a.imu, dtype=torch.float32),
            "anchor_key_mask": torch.tensor(1.0 if a.has_keystroke else 0.0, dtype=torch.float32),
            "anchor_scroll_mask": torch.tensor(1.0 if a.has_scroll else 0.0, dtype=torch.float32),
            "anchor_imu_mask": torch.tensor(1.0 if a.has_imu else 0.0, dtype=torch.float32),
            "positive_keys": torch.tensor(p.keystrokes, dtype=torch.float32),
            "positive_scrolls": torch.tensor(p.scrolls, dtype=torch.float32),
            "positive_imu": torch.tensor(p.imu, dtype=torch.float32),
            "positive_key_mask": torch.tensor(1.0 if p.has_keystroke else 0.0, dtype=torch.float32),
            "positive_scroll_mask": torch.tensor(1.0 if p.has_scroll else 0.0, dtype=torch.float32),
            "positive_imu_mask": torch.tensor(1.0 if p.has_imu else 0.0, dtype=torch.float32),
            "negative_keys": torch.tensor(n.keystrokes, dtype=torch.float32),
            "negative_scrolls": torch.tensor(n.scrolls, dtype=torch.float32),
            "negative_imu": torch.tensor(n.imu, dtype=torch.float32),
            "negative_key_mask": torch.tensor(1.0 if n.has_keystroke else 0.0, dtype=torch.float32),
            "negative_scroll_mask": torch.tensor(1.0 if n.has_scroll else 0.0, dtype=torch.float32),
            "negative_imu_mask": torch.tensor(1.0 if n.has_imu else 0.0, dtype=torch.float32),
            "anchor_label": torch.tensor(self.user_to_idx[ref.anchor_user], dtype=torch.long),
        }
