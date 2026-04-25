"""LSTM-based behavioral biometrics model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BehavioralLSTM(nn.Module):
    """LSTM model for behavioral authentication using keystroke, scroll, and IMU data."""
    
    def __init__(self, num_users: int, key_vocab_size: int = 17, embed_dim: int = 8):
        """
        Initialize BehavioralLSTM.
        
        Args:
            num_users: Number of users for classification head
            key_vocab_size: Size of keystroke vocabulary (including UNK)
            embed_dim: Embedding dimension for keystrokes
        """
        super().__init__()
        
        # Keystroke branch
        self.key_embed = nn.Embedding(key_vocab_size, embed_dim, padding_idx=0)
        self.key_lstm = nn.LSTM(
            input_size=embed_dim + 3,  # embedding + 3 features
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Scroll branch
        self.scroll_lstm = nn.LSTM(
            input_size=6,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 4, 64),  # key(64) + scroll(64) + imu(4)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # DNA head (embedding)
        self.dna_head = nn.Linear(32, 32)
        
        # Classification head (for training)
        self.classifier = nn.Linear(32, num_users)
        
        # Risk head (binary classification)
        self.risk_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        keystrokes: torch.Tensor,
        scrolls: torch.Tensor,
        imu_stats: torch.Tensor,
        return_dna: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            keystrokes: [batch, 8, 4]
            scrolls: [batch, 20, 6]
            imu_stats: [batch, 4]
            return_dna: If True, return only DNA; else return (dna, logits, risk)
            
        Returns:
            If return_dna: DNA tensor [batch, 32]
            Else: Tuple of (dna, logits, risk)
        """
        # Keystroke branch
        # keystrokes[:, :, 0] is the key index
        key_indices = keystrokes[:, :, 0].long()
        key_emb = self.key_embed(key_indices)  # [batch, 8, embed_dim]
        key_numeric = keystrokes[:, :, 1:]  # [batch, 8, 3]
        key_input = torch.cat([key_emb, key_numeric], dim=-1)  # [batch, 8, embed_dim+3]
        
        _, (key_hidden, _) = self.key_lstm(key_input)
        key_out = torch.cat([key_hidden[0], key_hidden[1]], dim=-1)  # [batch, 64]
        
        # Scroll branch
        _, (scroll_hidden, _) = self.scroll_lstm(scrolls)
        scroll_out = torch.cat([scroll_hidden[0], scroll_hidden[1]], dim=-1)  # [batch, 64]
        
        # Fusion
        fused = torch.cat([key_out, scroll_out, imu_stats], dim=-1)  # [batch, 132]
        features = self.fusion(fused)  # [batch, 32]
        
        # DNA head with normalization
        dna = self.dna_head(features)  # [batch, 32]
        dna = F.normalize(dna, p=2, dim=1)
        
        if return_dna:
            return dna
        
        # Classification and risk heads
        logits = self.classifier(dna)  # [batch, num_users]
        risk = self.risk_head(dna)  # [batch, 1]
        
        return dna, logits, risk


if __name__ == "__main__":
    # Test model
    model = BehavioralLSTM(num_users=40)
    
    batch_size = 4
    keystrokes = torch.randn(batch_size, 8, 4)
    scrolls = torch.randn(batch_size, 20, 6)
    imu_stats = torch.randn(batch_size, 4)
    
    # Test return_dna=True
    dna = model(keystrokes, scrolls, imu_stats, return_dna=True)
    print(f"DNA shape: {dna.shape}")
    assert dna.shape == (batch_size, 32)
    
    # Test return_dna=False
    dna, logits, risk = model(keystrokes, scrolls, imu_stats, return_dna=False)
    print(f"DNA shape: {dna.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Risk shape: {risk.shape}")
    assert dna.shape == (batch_size, 32)
    assert logits.shape == (batch_size, 40)
    assert risk.shape == (batch_size, 1)
    
    print("Model test passed!")
