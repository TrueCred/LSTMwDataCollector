"""Inference engine for behavioral authentication."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from data_loader import (
    preprocess_keystrokes,
    preprocess_scrolls,
    extract_imu_stats,
    KEY_VOCAB
)


class BehavioralInference:
    """Inference engine for behavioral authentication."""
    
    def __init__(
        self,
        onnx_model_path: str | Path = "sentinel_encoder.onnx",
        scalers_path: str | Path = "scalers.pkl",
        key_vocab_path: str | Path = "key_vocab.json"
    ):
        """
        Initialize inference engine.
        
        Args:
            onnx_model_path: Path to sentinel_encoder.onnx
            scalers_path: Path to scalers.pkl
            key_vocab_path: Path to key_vocab.json
        """
        self.onnx_model_path = Path(onnx_model_path)
        self.scalers_path = Path(scalers_path)
        self.key_vocab_path = Path(key_vocab_path)
        
        # Load ONNX session
        if self.onnx_model_path.exists():
            self.session = ort.InferenceSession(str(self.onnx_model_path))
        else:
            self.session = None
            print(f"Warning: ONNX model not found at {self.onnx_model_path}")
        
        # Load scalers
        if self.scalers_path.exists():
            with open(self.scalers_path, "rb") as f:
                self.scalers = pickle.load(f)
        else:
            self.scalers = {}
            print(f"Warning: Scalers not found at {self.scalers_path}")
        
        # Load key vocab
        if self.key_vocab_path.exists():
            with open(self.key_vocab_path, "r") as f:
                self.key_vocab = json.load(f)
        else:
            self.key_vocab = KEY_VOCAB
    
    def extract_dna(
        self,
        keystrokes_list: list,
        scrolls_list: list,
        imu_list: list
    ) -> Optional[np.ndarray]:
        """
        Extract DNA embedding from behavioral data.
        
        Args:
            keystrokes_list: List of keystroke events
            scrolls_list: List of scroll events
            imu_list: List of IMU readings
            
        Returns:
            numpy array of shape [1, 32] or None if inference fails
        """
        if self.session is None:
            print("ONNX session not available")
            return None
        
        try:
            # Preprocess
            ks_scaler = self.scalers.get("keystrokes")
            sc_scaler = self.scalers.get("scrolls")
            imu_scaler = self.scalers.get("imu")
            
            keystrokes = preprocess_keystrokes(keystrokes_list, scaler=ks_scaler)
            scrolls = preprocess_scrolls(scrolls_list, scaler=sc_scaler)
            imu_stats = extract_imu_stats(imu_list, scaler=imu_scaler)
            
            # Run inference
            input_dict = {
                "keystrokes": keystrokes.astype(np.float32),
                "scrolls": scrolls.astype(np.float32),
                "imu_stats": imu_stats.astype(np.float32)
            }
            
            outputs = self.session.run(None, input_dict)
            dna = outputs[0]  # [1, 32]
            
            return dna
        except Exception as e:
            print(f"Error in extract_dna: {e}")
            return None
    
    def compute_risk(
        self,
        live_dna: np.ndarray,
        template_dna: np.ndarray
    ) -> float:
        """
        Compute risk score from live and template DNA.
        
        Args:
            live_dna: [1, 32] numpy array
            template_dna: [1, 32] numpy array
            
        Returns:
            Risk score between 0 and 1 (0 = authentic, 1 = imposter)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity = cosine_similarity(live_dna, template_dna)[0, 0]
        risk = 1.0 - float(similarity)
        
        return max(0.0, min(1.0, risk))


if __name__ == "__main__":
    # Test inference
    inference = BehavioralInference()
    
    # Create dummy data
    dummy_keystrokes = [
        {"key": "v", "hold_time_ms": 50, "flight_time_ms": 20},
        {"key": "k", "hold_time_ms": 45, "flight_time_ms": 25}
    ]
    dummy_scrolls = [
        {"direction_deg": 90, "distance_px": 100},
        {"direction_deg": 45, "distance_px": 150}
    ]
    dummy_imu = [
        {"gyro_x": 0.5, "gyro_y": 0.3, "tilt_pitch": 10, "tilt_roll": 5}
    ]
    
    dna = inference.extract_dna(dummy_keystrokes, dummy_scrolls, dummy_imu)
    
    if dna is not None:
        print(f"DNA shape: {dna.shape}")
        print(f"DNA norm: {np.linalg.norm(dna):.4f}")
        
        # Test risk computation
        risk = inference.compute_risk(dna, dna)
        print(f"Self-risk (should be ~0): {risk:.4f}")
