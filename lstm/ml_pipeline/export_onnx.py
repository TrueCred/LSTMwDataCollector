from __future__ import annotations

from pathlib import Path

import torch

from data_loader import KEY_VOCAB
from model import DNAOnlyWrapper, TriBranchLSTM

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"


def export():
    model = TriBranchLSTM(key_vocab_size=len(KEY_VOCAB))
    state = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy_keys = torch.randn(1, 8, 4)
    dummy_scrolls = torch.randn(1, 20, 6)
    dummy_imu = torch.randn(1, 250, 5)

    torch.onnx.export(
        model,
        (dummy_keys, dummy_scrolls, dummy_imu),
        CHECKPOINT_DIR / "sentinel_lstm.onnx",
        input_names=["keystrokes", "scrolls", "imu"],
        output_names=["dna", "risk"],
        dynamic_axes={
            "keystrokes": {0: "batch_size"},
            "scrolls": {0: "batch_size"},
            "imu": {0: "batch_size"},
            "dna": {0: "batch_size"},
            "risk": {0: "batch_size"},
        },
        opset_version=13,
    )

    dna_model = DNAOnlyWrapper(model)
    dna_model.eval()

    torch.onnx.export(
        dna_model,
        (dummy_keys, dummy_scrolls, dummy_imu),
        CHECKPOINT_DIR / "sentinel_lstm_dna.onnx",
        input_names=["keystrokes", "scrolls", "imu"],
        output_names=["dna"],
        dynamic_axes={
            "keystrokes": {0: "batch_size"},
            "scrolls": {0: "batch_size"},
            "imu": {0: "batch_size"},
            "dna": {0: "batch_size"},
        },
        opset_version=13,
    )

    print("Exported ONNX models to", CHECKPOINT_DIR)


if __name__ == "__main__":
    export()
