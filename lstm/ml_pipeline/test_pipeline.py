from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_loader import merge_all_sources
from export_onnx import export
from inference import BehavioralInference
from train import run_training

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DB_PATH = ROOT.parent.parent / "fastapiCollector" / "sentinel_lab.db"
DATASETS_DIR = ROOT / "datasets"


def quick_test():
    print("Testing full pipeline...")

    run_training(db_path=DB_PATH, datasets_dir=DATASETS_DIR, phase="all", batch_size=8)
    export()

    infer = BehavioralInference(
        model_path=CHECKPOINT_DIR / "sentinel_lstm.onnx",
        scalers_path=CHECKPOINT_DIR / "scalers.pkl",
        key_vocab_path=CHECKPOINT_DIR / "key_vocab.json",
    )

    data = merge_all_sources(DB_PATH, DATASETS_DIR)
    users = sorted(data.keys())

    if len(users) < 2:
        raise RuntimeError("Need >=2 users with complete sessions for test inference.")

    u0 = users[0]
    u1 = users[1]

    s0 = data[u0][0]
    s1 = data[u0][1]
    s2 = data[u1][0]

    # This conversion is only for test convenience and preserves feature semantics.
    def to_payload_like(sample):
        return {
            "keys": [
                {
                    "key": "v" if float(x[0]) > 0 else " ",
                    "hold_time_ms": max(0.0, float(np.expm1(x[1]))),
                    "flight_time_ms": max(0.0, float(np.expm1(x[2]))),
                    "pressure": float(x[3]),
                }
                for x in sample.keystrokes
            ],
            "scrolls": [
                {
                    "velocity_px_per_sec": float(x[0]),
                    "direction_deg": float(np.degrees(np.arctan2(x[1], x[2]))),
                    "distance_px": float(x[3]),
                    "avg_pressure": float(x[4]),
                    "pixel_density_dpi": float(x[5]),
                }
                for x in sample.scrolls
            ],
            "imu": [
                {
                    "gyro_x": float(x[0]),
                    "gyro_y": float(x[1]),
                    "gyro_z": float(x[2]),
                    "tilt_pitch": float(x[3]),
                    "tilt_roll": float(x[4]),
                }
                for x in sample.imu
            ],
        }

    p0 = to_payload_like(s0)
    p1 = to_payload_like(s1)
    p2 = to_payload_like(s2)

    d0 = infer.extract_dna(p0["keys"], p0["scrolls"], p0["imu"])
    d1 = infer.extract_dna(p1["keys"], p1["scrolls"], p1["imu"])
    d2 = infer.extract_dna(p2["keys"], p2["scrolls"], p2["imu"])

    risk_same = infer.compute_risk(d0, d1)
    risk_diff = infer.compute_risk(d0, d2)

    with open(CHECKPOINT_DIR / "training_log.json", "r", encoding="utf-8") as f:
        log = json.load(f)

    print("DNA shape:", d0.shape)
    print("risk(same_user):", round(risk_same, 4))
    print("risk(diff_user):", round(risk_diff, 4))
    print("Logged phases:", len(log.get("phases", [])))


def test_keyrecs_embedding():
    data = merge_all_sources(DB_PATH, DATASETS_DIR)
    infer = BehavioralInference(
        model_path=CHECKPOINT_DIR / "sentinel_lstm.onnx",
        scalers_path=CHECKPOINT_DIR / "scalers.pkl",
        key_vocab_path=CHECKPOINT_DIR / "key_vocab.json",
    )
    for uid, sessions in data.items():
        if sessions and sessions[0].source == "keyrecs":
            s = sessions[0]
            keys = [
                {
                    "key": "v" if float(x[0]) > 0 else " ",
                    "hold_time_ms": float(np.expm1(x[1])),
                    "flight_time_ms": float(np.expm1(x[2])),
                    "pressure": float(x[3]),
                }
                for x in s.keystrokes
            ]
            dna = infer.extract_dna(keys, [], [])
            print("KeyRecs DNA shape:", dna.shape)
            return
    print("KeyRecs test skipped: dataset not found")


def test_touchalytics_embedding():
    data = merge_all_sources(DB_PATH, DATASETS_DIR)
    infer = BehavioralInference(
        model_path=CHECKPOINT_DIR / "sentinel_lstm.onnx",
        scalers_path=CHECKPOINT_DIR / "scalers.pkl",
        key_vocab_path=CHECKPOINT_DIR / "key_vocab.json",
    )
    for uid, sessions in data.items():
        if sessions and sessions[0].source == "touchalytics":
            s = sessions[0]
            scrolls = [
                {
                    "velocity_px_per_sec": float(x[0]),
                    "direction_deg": float(np.degrees(np.arctan2(x[1], x[2]))),
                    "distance_px": float(x[3]),
                    "avg_pressure": float(x[4]),
                    "pixel_density_dpi": float(x[5]),
                }
                for x in s.scrolls
            ]
            dna = infer.extract_dna([], scrolls, [])
            print("Touchalytics DNA shape:", dna.shape)
            return
    print("Touchalytics test skipped: dataset not found")


def test_booth_user_similarity():
    data = merge_all_sources(DB_PATH, DATASETS_DIR)
    infer = BehavioralInference(
        model_path=CHECKPOINT_DIR / "sentinel_lstm.onnx",
        scalers_path=CHECKPOINT_DIR / "scalers.pkl",
        key_vocab_path=CHECKPOINT_DIR / "key_vocab.json",
    )
    booth_users = [u for u, s in data.items() if s and s[0].source == "booth"]
    if len(booth_users) < 2:
        print("Booth similarity test skipped: insufficient booth users")
        return

    u0, u1 = booth_users[0], booth_users[1]
    s0, s1, s2 = data[u0][0], data[u0][1], data[u1][0]

    def payload(sample):
        return [
            {"key": "v" if float(x[0]) > 0 else " ", "hold_time_ms": float(np.expm1(x[1])), "flight_time_ms": float(np.expm1(x[2])), "pressure": float(x[3])}
            for x in sample.keystrokes
        ], [
            {"velocity_px_per_sec": float(x[0]), "direction_deg": float(np.degrees(np.arctan2(x[1], x[2]))), "distance_px": float(x[3]), "avg_pressure": float(x[4]), "pixel_density_dpi": float(x[5])}
            for x in sample.scrolls
        ], [
            {"gyro_x": float(x[0]), "gyro_y": float(x[1]), "gyro_z": float(x[2]), "tilt_pitch": float(x[3]), "tilt_roll": float(x[4])}
            for x in sample.imu
        ]

    k0, sc0, i0 = payload(s0)
    k1, sc1, i1 = payload(s1)
    k2, sc2, i2 = payload(s2)

    d0 = infer.extract_dna(k0, sc0, i0)
    d1 = infer.extract_dna(k1, sc1, i1)
    d2 = infer.extract_dna(k2, sc2, i2)

    same_sim = 1.0 - infer.compute_risk(d0, d1)
    diff_sim = 1.0 - infer.compute_risk(d0, d2)
    print("booth_same_similarity:", round(float(same_sim), 4))
    print("booth_diff_similarity:", round(float(diff_sim), 4))


if __name__ == "__main__":
    quick_test()
    test_keyrecs_embedding()
    test_touchalytics_embedding()
    test_booth_user_similarity()
