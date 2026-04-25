from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort

from data_loader import KEY_VOCAB


class BehavioralInference:
    def __init__(
        self,
        model_path: str | Path,
        scalers_path: str | Path,
        key_vocab_path: str | Path | None = None,
    ):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        with open(scalers_path, "rb") as f:
            self.scalers: Dict = pickle.load(f)

        if key_vocab_path is not None and Path(key_vocab_path).exists():
            with open(key_vocab_path, "r", encoding="utf-8") as f:
                self.key_to_idx = json.load(f)
        else:
            self.key_to_idx = KEY_VOCAB

    def _encode_key(self, key: str) -> int:
        if not key:
            return 0
        return int(self.key_to_idx.get(str(key), 0))

    def preprocess_keystrokes(self, keystrokes_list: List[dict]) -> np.ndarray:
        arr = np.zeros((8, 4), dtype=np.float32)
        for i, ev in enumerate(keystrokes_list[:8]):
            arr[i, 0] = float(self._encode_key(str(ev.get("key", ""))))
            arr[i, 1] = np.log1p(max(0.0, float(ev.get("hold_time_ms", 0.0))))
            arr[i, 2] = np.log1p(max(0.0, float(ev.get("flight_time_ms", 0.0))))
            arr[i, 3] = float(ev.get("pressure", 0.5))

        arr[:, 1:4] = self.scalers["keys_num"].transform(arr[:, 1:4])
        return arr[np.newaxis, ...].astype(np.float32)

    def preprocess_scrolls(self, scrolls_list: List[dict]) -> np.ndarray:
        max_len = 20
        arr = np.zeros((max_len, 6), dtype=np.float32)
        for i, ev in enumerate(scrolls_list[:max_len]):
            direction = float(ev.get("direction_deg", 0.0))
            rad = np.deg2rad(direction)
            arr[i, 0] = float(ev.get("velocity_px_per_sec", 0.0))
            arr[i, 1] = np.sin(rad)
            arr[i, 2] = np.cos(rad)
            arr[i, 3] = float(ev.get("distance_px", 0.0))
            arr[i, 4] = float(ev.get("avg_pressure", 0.5))
            arr[i, 5] = float(ev.get("pixel_density_dpi", 0.0))

        arr = self.scalers["scrolls"].transform(arr)
        return arr[np.newaxis, ...].astype(np.float32)

    def preprocess_imu(self, imu_list: List[dict]) -> np.ndarray:
        target_len = 250
        raw = np.zeros((len(imu_list), 5), dtype=np.float32)

        for i, ev in enumerate(imu_list):
            raw[i, 0] = float(ev.get("gyro_x", 0.0))
            raw[i, 1] = float(ev.get("gyro_y", 0.0))
            raw[i, 2] = float(ev.get("gyro_z", 0.0))
            raw[i, 3] = float(ev.get("tilt_pitch", 0.0))
            raw[i, 4] = float(ev.get("tilt_roll", 0.0))

        if len(raw) == 0:
            arr = np.zeros((target_len, 5), dtype=np.float32)
        elif len(raw) >= target_len:
            start = max(0, (len(raw) - target_len) // 2)
            arr = raw[start:start + target_len]
        else:
            reps = int(np.ceil(target_len / len(raw)))
            arr = np.tile(raw, (reps, 1))[:target_len]

        arr = self.scalers["imu"].transform(arr)
        return arr[np.newaxis, ...].astype(np.float32)

    def extract(self, keystrokes: List[dict], scrolls: List[dict], imu: List[dict]):
        k = self.preprocess_keystrokes(keystrokes)
        s = self.preprocess_scrolls(scrolls)
        i = self.preprocess_imu(imu)

        outputs = self.session.run(None, {"keystrokes": k, "scrolls": s, "imu": i})
        if len(outputs) == 1:
            dna = outputs[0]
            risk = None
        else:
            dna, risk = outputs[0], outputs[1]

        return dna, risk

    def extract_dna(self, keystrokes: List[dict], scrolls: List[dict], imu: List[dict]) -> np.ndarray:
        dna, _ = self.extract(keystrokes, scrolls, imu)
        return dna

    def extract_template(self, keystrokes: List[dict], scrolls: List[dict], imu: List[dict]) -> np.ndarray:
        return self.extract_dna(keystrokes, scrolls, imu)

    def compute_risk(self, live_dna: np.ndarray, template_dna: np.ndarray) -> float:
        live = live_dna.reshape(1, -1)
        tmpl = template_dna.reshape(1, -1)

        live_norm = np.linalg.norm(live, axis=1, keepdims=True)
        tmpl_norm = np.linalg.norm(tmpl, axis=1, keepdims=True)
        denom = np.clip(live_norm * tmpl_norm, 1e-8, None)
        sim = float(np.sum(live * tmpl, axis=1, keepdims=True) / denom)
        sim = max(0.0, min(1.0, sim))
        return float(1.0 - sim)
