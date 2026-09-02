"""Export BehavioralLSTM encoder to ONNX format (key-agnostic version)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.onnx

from model import BehavioralLSTM


def export_to_onnx(
    model_path: str | Path,
    output_path: str | Path,
    device: str = "cpu"
) -> None:
    """Export encoder part of BehavioralLSTM to ONNX."""
    model_path = Path(model_path)
    output_path = Path(output_path)

    # Load checkpoint and auto-detect num_users
    state_dict = torch.load(model_path, map_location=device)
    num_users = state_dict["classifier.bias"].shape[0]

    model = BehavioralLSTM(num_users=num_users).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy inputs — keystrokes are now [batch, 8, 2] (timing only)
    batch_size = 1
    dummy_keystrokes = torch.randn(batch_size, 8, 2, device=device)
    dummy_scrolls = torch.randn(batch_size, 20, 6, device=device)
    dummy_imu = torch.randn(batch_size, 4, device=device)

    torch.onnx.export(
        model,
        args=(dummy_keystrokes, dummy_scrolls, dummy_imu, True),
        f=str(output_path),
        input_names=["keystrokes", "scrolls", "imu_stats"],
        output_names=["dna"],
        opset_version=14,
        do_constant_folding=True,
        dynamic_axes={
            "keystrokes": {0: "batch_size"},
            "scrolls": {0: "batch_size"},
            "imu_stats": {0: "batch_size"},
            "dna": {0: "batch_size"}
        },
    )

    print(f"Exported model to {output_path}")
    print(f"  Input: keystrokes [B, 8, 2], scrolls [B, 20, 6], imu [B, 4]")
    print(f"  Output: dna [B, 32]")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    checkpoint_dir = root / "checkpoints"

    model_path = checkpoint_dir / "best_model.pt"
    output_path = root / "sentinel_encoder.onnx"

    if model_path.exists():
        export_to_onnx(model_path, output_path)
    else:
        print(f"Model not found at {model_path}")
        print("Run train.py first to generate the model")
