from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriBranchLSTM(nn.Module):
    def __init__(self, key_vocab_size: int = 17, embed_dim: int = 8):
        super().__init__()

        self.key_embed = nn.Embedding(key_vocab_size, embed_dim, padding_idx=0)
        self.key_lstm = nn.LSTM(
            input_size=embed_dim + 3,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.scroll_lstm = nn.LSTM(
            input_size=6,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.imu_lstm = nn.LSTM(
            input_size=5,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.dna_head = nn.Linear(64, 32)

        self.risk_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        keystrokes,
        scrolls,
        imu,
        key_mask=None,
        scroll_mask=None,
        imu_mask=None,
        return_embedding: bool = False,
    ):
        keys = keystrokes[:, :, 0].long()
        k_emb = self.key_embed(keys)
        k_num = keystrokes[:, :, 1:]
        k_in = torch.cat([k_emb, k_num], dim=-1)
        _, (k_h, _) = self.key_lstm(k_in)
        k_out = torch.cat([k_h[0], k_h[1]], dim=-1)

        _, (s_h, _) = self.scroll_lstm(scrolls)
        s_out = torch.cat([s_h[0], s_h[1]], dim=-1)

        _, (i_h, _) = self.imu_lstm(imu)
        i_out = torch.cat([i_h[0], i_h[1]], dim=-1)

        if key_mask is not None:
            k_out = k_out * key_mask.unsqueeze(1)
        if scroll_mask is not None:
            s_out = s_out * scroll_mask.unsqueeze(1)
        if imu_mask is not None:
            i_out = i_out * imu_mask.unsqueeze(1)

        fused = torch.cat([k_out, s_out, i_out], dim=-1)
        features = self.fusion(fused)

        dna = self.dna_head(features)
        dna = F.normalize(dna, p=2, dim=1)

        if return_embedding:
            return dna

        risk = self.risk_head(dna)
        return dna, risk


class DNAOnlyWrapper(nn.Module):
    def __init__(self, base_model: TriBranchLSTM):
        super().__init__()
        self.base_model = base_model

    def forward(self, keystrokes, scrolls, imu):
        return self.base_model(keystrokes, scrolls, imu, return_embedding=True)
