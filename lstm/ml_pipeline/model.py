"""LSTM-based behavioral biometrics model (key-agnostic version).

This version does NOT use key identity as an input feature. Instead, it
relies purely on timing features (hold_time, flight_time) so that
verification works regardless of what text the user is typing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BehavioralLSTM(nn.Module):
    """Key-agnostic LSTM model for behavioral authentication."""

    def __init__(self, num_users: int):
        """
        Args:
            num_users: Number of users for classification head (used during training only)
        """
        super().__init__()

        # Keystroke branch — timing features only: [hold_time, flight_time]
        self.key_lstm = nn.LSTM(
            input_size=2,      # log_hold_time, log_flight_time
            hidden_size=64,    # wider to compensate for removed key embedding
            num_layers=2,      # deeper for better pattern extraction
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # Scroll branch
        self.scroll_lstm = nn.LSTM(
            input_size=6,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Fusion layers
        # key: 64*2 (bidirectional) = 128, scroll: 32*2 = 64, imu: 4
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 4, 96),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # DNA head (embedding)
        self.dna_head = nn.Linear(48, 32)

        # Classification head (for triplet training structure)
        self.classifier = nn.Linear(32, num_users)

        # Risk head
        self.risk_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        keystrokes: torch.Tensor,
        scrolls: torch.Tensor,
        imu_stats: torch.Tensor,
        return_dna: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            keystrokes: [batch, 8, 2] — timing features only
            scrolls:    [batch, 20, 6]
            imu_stats:  [batch, 4]
            return_dna: If True, return only DNA; else return (dna, logits, risk)
        """
        # Keystroke branch — pure timing
        _, (key_hidden, _) = self.key_lstm(keystrokes)
        # key_hidden: [num_layers*2, batch, 64] — take last layer
        key_out = torch.cat(
            [key_hidden[-2], key_hidden[-1]], dim=-1
        )  # [batch, 128]

        # Scroll branch
        _, (scroll_hidden, _) = self.scroll_lstm(scrolls)
        scroll_out = torch.cat(
            [scroll_hidden[0], scroll_hidden[1]], dim=-1
        )  # [batch, 64]

        # Fusion
        fused = torch.cat([key_out, scroll_out, imu_stats], dim=-1)  # [batch, 196]
        features = self.fusion(fused)  # [batch, 48]

        # DNA head with L2 normalization
        dna = self.dna_head(features)  # [batch, 32]
        dna = F.normalize(dna, p=2, dim=1)

        if return_dna:
            return dna

        logits = self.classifier(dna)
        risk = self.risk_head(dna)

        return dna, logits, risk


if __name__ == "__main__":
    model = BehavioralLSTM(num_users=40)

    batch_size = 4
    keystrokes = torch.randn(batch_size, 8, 2)   # timing only
    scrolls = torch.randn(batch_size, 20, 6)
    imu_stats = torch.randn(batch_size, 4)

    dna = model(keystrokes, scrolls, imu_stats, return_dna=True)
    print(f"DNA shape: {dna.shape}")
    assert dna.shape == (batch_size, 32)

    dna, logits, risk = model(keystrokes, scrolls, imu_stats, return_dna=False)
    print(f"Logits: {logits.shape}, Risk: {risk.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("Model test passed!")
