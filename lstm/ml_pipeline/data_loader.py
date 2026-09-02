"""Data loading and preprocessing for behavioral biometrics."""
from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Key vocabulary mapping
KEY_VOCAB = {
    "v": 1, "k": 2, "e": 3, "r": 4, "j": 5, "p": 6, "w": 7, "u": 8,
    "1": 9, "3": 10, "7": 11, "9": 12, "2": 13, "8": 14, "4": 15, "6": 16
}

# Keyrecs fixed-text key sequence: vpwjkeurkb
KEYRECS_KEYS = ["v", "p", "w", "j", "k", "e", "u", "r", "k", "b"]
KEYRECS_HOLD_COLS = [
    "DU.v.v", "DU.p.p", "DU.w.w", "DU.j.j", "DU.k.k",
    "DU.e.e", "DU.u.u", "DU.r.r", "DU.k.k.1", "DU.b.b"
]
KEYRECS_FLIGHT_COLS = [
    None, "UD.v.p", "UD.p.w", "UD.w.j", "UD.j.k",
    "UD.k.e", "UD.e.u", "UD.u.r", "UD.r.k", "UD.k.b"
]


def save_key_vocab(output_path: str | Path = "key_vocab.json") -> None:
    """Save key vocabulary to JSON file."""
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(KEY_VOCAB, f)
    print(f"Saved key_vocab to {output_path}")


def load_raw_enrollment(db_path: str | Path) -> Dict[str, Dict[str, list]]:
    """
    Load raw_enrollment table from database.
    
    Args:
        db_path: Path to sentinel_lab.db
        
    Returns:
        Dictionary {user_id: {keystrokes: [...], scrolls: [...], imu: [...]}}
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT user_id, keystrokes_json, scrolls_json, imu_json FROM raw_enrollment")
        rows = cursor.fetchall()
    finally:
        conn.close()
    
    unified_data = {}
    for user_id, ks_json, sc_json, imu_json in rows:
        keystrokes = json.loads(ks_json) if ks_json else []
        scrolls = json.loads(sc_json) if sc_json else []
        imu = json.loads(imu_json) if imu_json else []
        
        if user_id not in unified_data:
            unified_data[user_id] = {"keystrokes": [], "scrolls": [], "imu": []}
        
        unified_data[user_id]["keystrokes"].extend(keystrokes)
        unified_data[user_id]["scrolls"].extend(scrolls)
        unified_data[user_id]["imu"].extend(imu)
    
    return unified_data


def load_keyrecs_fixed(csv_path: str | Path) -> Dict[str, List[Dict[str, list]]]:
    """
    Load keyrecs fixed-text.csv and convert to windowed samples per user.
    
    Each row is one typing session of 'vpwjkeurkb'. We extract per-key
    hold times (DU.x.x) and inter-key flight times (UD.x.y).
    
    Returns:
        {participant_id: [{keystrokes: [...], scrolls: [], imu: []}, ...]}
        Each inner dict is one session (= one sample).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    user_samples: Dict[str, List[Dict[str, list]]] = {}
    
    for _, row in df.iterrows():
        participant = str(row["participant"])
        
        # Build keystroke events from digraph timings
        events = []
        for i, key in enumerate(KEYRECS_KEYS):
            hold_time = max(0.0, float(row[KEYRECS_HOLD_COLS[i]])) * 1000  # s→ms
            
            if KEYRECS_FLIGHT_COLS[i] is not None:
                flight_time = max(0.0, float(row[KEYRECS_FLIGHT_COLS[i]])) * 1000
            else:
                flight_time = 0.0
            
            events.append({
                "key": key,
                "hold_time_ms": hold_time,
                "flight_time_ms": flight_time
            })
        
        sample = {"keystrokes": events, "scrolls": [], "imu": []}
        
        if participant not in user_samples:
            user_samples[participant] = []
        user_samples[participant].append(sample)
    
    return user_samples


def window_user_data(
    user_data: Dict[str, list],
    ks_window: int = 8,
    ks_stride: int = 4
) -> List[Dict[str, list]]:
    """
    Window a single user's raw data into multiple samples.
    
    Keystrokes are windowed with overlap. Scrolls and IMU are
    shared across windows (the preprocessor handles truncation/averaging).
    
    Args:
        user_data: {keystrokes: [...], scrolls: [...], imu: [...]}
        ks_window: Keystroke window size
        ks_stride: Keystroke window stride
        
    Returns:
        List of sample dicts, each with {keystrokes, scrolls, imu}
    """
    keystrokes = user_data["keystrokes"]
    scrolls = user_data["scrolls"]
    imu = user_data["imu"]
    
    samples = []
    
    if len(keystrokes) < ks_window:
        # Not enough for even one window, use what we have
        samples.append({"keystrokes": keystrokes, "scrolls": scrolls, "imu": imu})
    else:
        for start in range(0, len(keystrokes) - ks_window + 1, ks_stride):
            ks_win = keystrokes[start:start + ks_window]
            samples.append({"keystrokes": ks_win, "scrolls": scrolls, "imu": imu})
    
    return samples


def preprocess_keystrokes(
    keystrokes_list: list,
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Preprocess keystrokes to [1, 8, 2] tensor (key-agnostic, timing only).
    
    Takes last 8 keystrokes, applies log1p to hold/flight times, pads to 8.
    Features: [log_hold_time, log_flight_time]
    
    Args:
        keystrokes_list: List of keystroke events with keys: 'hold_time_ms', 'flight_time_ms'
        scaler: Optional StandardScaler for features
        
    Returns:
        numpy array of shape [1, 8, 2]
    """
    result = np.zeros((1, 8, 2), dtype=np.float32)
    
    if not keystrokes_list:
        return result
    
    # Take last 8
    events = keystrokes_list[-8:]
    
    for i, event in enumerate(events):
        hold_time = max(0.0, float(event.get("hold_time_ms", 0.0)))
        flight_time = max(0.0, float(event.get("flight_time_ms", 0.0)))
        
        result[0, i, 0] = np.log1p(hold_time)
        result[0, i, 1] = np.log1p(flight_time)
    
    # Apply scaler if provided
    if scaler is not None:
        features_to_scale = result[0, :, :].reshape(-1, 2)
        scaled = scaler.transform(features_to_scale)
        result[0, :, :] = scaled.reshape(8, 2)
    
    return result


def preprocess_scrolls(
    scrolls_list: list,
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Preprocess scrolls to [1, 20, 6] tensor.
    
    Takes last 20 scrolls, converts direction_deg to sin/cos, pads to 20.
    
    Args:
        scrolls_list: List of scroll events with keys: 'direction_deg', 'distance_px'
        scaler: Optional StandardScaler for numerical features
        
    Returns:
        numpy array of shape [1, 20, 6]
    """
    result = np.zeros((1, 20, 6), dtype=np.float32)
    
    if not scrolls_list:
        return result
    
    # Take last 20
    events = scrolls_list[-20:]
    
    for i, event in enumerate(events):
        direction_deg = float(event.get("direction_deg", 0.0))
        distance_px = max(0.0, float(event.get("distance_px", 0.0)))
        
        # Convert direction to sin/cos
        angle_rad = np.radians(direction_deg)
        
        result[0, i, 0] = np.sin(angle_rad)
        result[0, i, 1] = np.cos(angle_rad)
        result[0, i, 2] = distance_px
        result[0, i, 3] = 1.0  # velocity placeholder
        result[0, i, 4] = 1.0  # acceleration placeholder
        result[0, i, 5] = 1.0  # timestamp placeholder
    
    # Apply scaler if provided
    if scaler is not None:
        features_to_scale = result[0, :, 2:6].reshape(-1, 4)
        scaled = scaler.transform(features_to_scale)
        result[0, :, 2:6] = scaled.reshape(20, 4)
    
    return result


def extract_imu_stats(
    imu_list: list,
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Extract IMU statistics as [1, 4] tensor.
    
    Returns mean of gyro_x, gyro_y, tilt_pitch, tilt_roll.
    
    Args:
        imu_list: List of IMU readings with keys: 'gyro_x', 'gyro_y', 'tilt_pitch', 'tilt_roll'
        scaler: Optional StandardScaler for features
        
    Returns:
        numpy array of shape [1, 4]
    """
    result = np.zeros((1, 4), dtype=np.float32)
    
    if not imu_list:
        return result
    
    gyro_x_vals = []
    gyro_y_vals = []
    pitch_vals = []
    roll_vals = []
    
    for reading in imu_list:
        gyro_x_vals.append(float(reading.get("gyro_x", 0.0)))
        gyro_y_vals.append(float(reading.get("gyro_y", 0.0)))
        pitch_vals.append(float(reading.get("tilt_pitch", 0.0)))
        roll_vals.append(float(reading.get("tilt_roll", 0.0)))
    
    result[0, 0] = np.mean(gyro_x_vals) if gyro_x_vals else 0.0
    result[0, 1] = np.mean(gyro_y_vals) if gyro_y_vals else 0.0
    result[0, 2] = np.mean(pitch_vals) if pitch_vals else 0.0
    result[0, 3] = np.mean(roll_vals) if roll_vals else 0.0
    
    # Apply scaler if provided
    if scaler is not None:
        scaled = scaler.transform(result)
        result[:] = scaled
    
    return result


def fit_scalers_from_samples(
    all_samples: List[Dict[str, list]]
) -> tuple[StandardScaler, StandardScaler, StandardScaler]:
    """
    Fit StandardScalers on a list of raw samples.
    
    Args:
        all_samples: List of {keystrokes, scrolls, imu} dicts
        
    Returns:
        Tuple of (keystrokes_scaler, scrolls_scaler, imu_scaler)
    """
    ks_features = []
    sc_features = []
    imu_features = []
    
    for sample in all_samples:
        ks_array = preprocess_keystrokes(sample["keystrokes"])
        ks_features.append(ks_array[0, :, :].reshape(-1, 2))
        
        sc_array = preprocess_scrolls(sample["scrolls"])
        sc_features.append(sc_array[0, :, 2:6].reshape(-1, 4))
        
        imu_array = extract_imu_stats(sample["imu"])
        imu_features.append(imu_array[0, :])
    
    ks_all = np.vstack(ks_features)
    sc_all = np.vstack(sc_features)
    imu_all = np.vstack(imu_features)
    
    ks_scaler = StandardScaler()
    ks_scaler.fit(ks_all)
    
    sc_scaler = StandardScaler()
    sc_scaler.fit(sc_all)
    
    imu_scaler = StandardScaler()
    imu_scaler.fit(imu_all)
    
    return ks_scaler, sc_scaler, imu_scaler


# Keep legacy function for backward compatibility
def fit_scalers(unified_data: Dict[str, Dict[str, list]]) -> tuple[StandardScaler, StandardScaler, StandardScaler]:
    """Fit StandardScalers on unified data (legacy)."""
    samples = [data for data in unified_data.values()]
    return fit_scalers_from_samples(samples)


def save_scalers(
    ks_scaler: StandardScaler,
    sc_scaler: StandardScaler,
    imu_scaler: StandardScaler,
    output_path: str | Path = "scalers.pkl"
) -> None:
    """Save scalers to pickle file."""
    output_path = Path(output_path)
    scalers = {
        "keystrokes": ks_scaler,
        "scrolls": sc_scaler,
        "imu": imu_scaler
    }
    with open(output_path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"Saved scalers to {output_path}")


def load_scalers(input_path: str | Path = "scalers.pkl") -> Dict[str, StandardScaler]:
    """Load scalers from pickle file."""
    input_path = Path(input_path)
    with open(input_path, "rb") as f:
        scalers = pickle.load(f)
    return scalers


if __name__ == "__main__":
    root = Path(__file__).parent
    db_path = root / "sentinel_lab.db"
    keyrecs_path = root / "datasets" / "keyrecs" / "fixed-text.csv"
    
    print(f"Loading raw enrollment from {db_path}")
    unified_data = load_raw_enrollment(db_path)
    print(f"  Loaded {len(unified_data)} users from DB")
    
    # Window DB data
    db_samples = {}
    for uid, data in unified_data.items():
        windows = window_user_data(data)
        if len(windows) >= 2:
            db_samples[uid] = windows
    print(f"  Windowed into {sum(len(v) for v in db_samples.values())} samples from {len(db_samples)} users")
    
    # Load keyrecs
    print(f"\nLoading keyrecs from {keyrecs_path}")
    keyrecs_samples = load_keyrecs_fixed(keyrecs_path)
    print(f"  Loaded {sum(len(v) for v in keyrecs_samples.values())} samples from {len(keyrecs_samples)} participants")
    
    # Save key vocab
    save_key_vocab()
    
    # Fit scalers on all samples combined
    all_samples = []
    for samples in db_samples.values():
        all_samples.extend(samples)
    for samples in keyrecs_samples.values():
        all_samples.extend(samples)
    
    ks_scaler, sc_scaler, imu_scaler = fit_scalers_from_samples(all_samples)
    save_scalers(ks_scaler, sc_scaler, imu_scaler)
    
    print(f"\nTotal: {len(all_samples)} samples across {len(db_samples) + len(keyrecs_samples)} users")
    print("Data preprocessing complete")
