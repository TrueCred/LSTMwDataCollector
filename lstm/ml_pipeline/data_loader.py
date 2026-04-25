from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

KEY_VOCAB = {
    "v": 0,
    "k": 1,
    "e": 2,
    "r": 3,
    "j": 4,
    "p": 5,
    "w": 6,
    "u": 7,
    "1": 8,
    "3": 9,
    "7": 10,
    "9": 11,
    "2": 12,
    "8": 13,
    "4": 14,
    "6": 15,
    " ": 16,
}

UNKNOWN_KEY_INDEX = 0


@dataclass
class SessionSample:
    user_id: str
    keystrokes: np.ndarray  # [8,4]
    scrolls: np.ndarray  # [20,6]
    imu: np.ndarray  # [250,5]
    has_keystroke: bool
    has_scroll: bool
    has_imu: bool
    source: str


def _db_connect(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def _safe_json_load(text: str | None) -> list:
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _encode_key(key: str) -> int:
    if not key:
        return UNKNOWN_KEY_INDEX
    return KEY_VOCAB.get(str(key), UNKNOWN_KEY_INDEX)


def _zero_keys() -> np.ndarray:
    return np.zeros((8, 4), dtype=np.float32)


def _zero_scrolls() -> np.ndarray:
    return np.zeros((20, 6), dtype=np.float32)


def _zero_imu() -> np.ndarray:
    return np.zeros((250, 5), dtype=np.float32)


def _keystrokes_to_fixed(events: List[dict]) -> np.ndarray:
    out = np.zeros((8, 4), dtype=np.float32)
    for i, ev in enumerate(events[:8]):
        out[i, 0] = float(_encode_key(str(ev.get("key", ""))))
        out[i, 1] = np.log1p(max(0.0, float(ev.get("hold_time_ms", 0.0))))
        out[i, 2] = np.log1p(max(0.0, float(ev.get("flight_time_ms", 0.0))))
        out[i, 3] = float(ev.get("pressure", 0.5))
    return out


def _keystrokes_from_fixed_text_row(row: pd.Series) -> List[dict]:
    """Parse KeyRecs fixed-text engineered-feature row into pseudo-keystroke events."""
    key_order: List[str] = []
    holds: Dict[str, float] = {}
    flights: Dict[Tuple[str, str], float] = {}

    for col_name in row.index:
        col = str(col_name)
        m_hold = re.match(r"^DU\.([^.]+)\.\1$", col)
        if m_hold:
            k = m_hold.group(1)
            if k not in key_order:
                key_order.append(k)
            try:
                holds[k] = float(row[col_name])
            except Exception:
                holds[k] = 0.0
            continue

        m_flight = re.match(r"^DD\.([^.]+)\.([^.]+)$", col)
        if m_flight:
            a, b = m_flight.group(1), m_flight.group(2)
            try:
                flights[(a, b)] = float(row[col_name])
            except Exception:
                flights[(a, b)] = 0.0

    events: List[dict] = []
    for i, k in enumerate(key_order):
        prev = key_order[i - 1] if i > 0 else None
        flight = 0.0 if prev is None else flights.get((prev, k), 0.0)
        events.append(
            {
                "key": k,
                "hold_time_ms": max(0.0, holds.get(k, 0.0) * 1000.0),
                "flight_time_ms": max(0.0, flight * 1000.0),
                "pressure": 0.5,
            }
        )

    return events


def _keystrokes_from_free_text_group(group_df: pd.DataFrame) -> List[dict]:
    events: List[dict] = []

    def as_float(v, default=0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    for _, row in group_df.iterrows():
        k1 = str(row.get("key1", ""))
        k2 = str(row.get("key2", ""))
        hold = as_float(row.get("DU.key1.key1", 0.0), 0.0)
        flight = as_float(row.get("DD.key1.key2", 0.0), 0.0)

        events.append(
            {
                "key": k1,
                "hold_time_ms": max(0.0, hold * 1000.0),
                "flight_time_ms": max(0.0, flight * 1000.0),
                "pressure": 0.5,
            }
        )

        # Ensure transition destination key appears in stream.
        if k2 and (not events or events[-1]["key"] != k2):
            events.append(
                {
                    "key": k2,
                    "hold_time_ms": max(0.0, hold * 1000.0),
                    "flight_time_ms": 0.0,
                    "pressure": 0.5,
                }
            )

    return events


def _events_from_keystroke_frame(df: pd.DataFrame) -> List[dict]:
    cols = {c.lower(): c for c in df.columns}

    def cands(*names: str) -> Optional[str]:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    key_col = cands("key", "character", "char", "key_char")
    hold_col = cands("hold_time", "hold_time_ms", "hold", "dwell", "dwell_time", "duration")
    flight_col = cands("flight_time", "flight_time_ms", "flight", "latency", "inter_key")
    pressure_col = cands("pressure", "force")
    ts_col = cands("timestamp", "time", "ts")

    if key_col is None:
        return []

    frame = df.copy()
    if ts_col is not None:
        frame = frame.sort_values(ts_col)

    events: List[dict] = []
    prev_ts = None
    for _, row in frame.iterrows():
        key = str(row.get(key_col, ""))
        hold = float(row.get(hold_col, 0.0) if hold_col else 0.0)
        pressure = float(row.get(pressure_col, 0.5) if pressure_col else 0.5)

        if flight_col is not None:
            flight = float(row.get(flight_col, 0.0))
        elif ts_col is not None:
            ts_val = float(row.get(ts_col, 0.0))
            flight = 0.0 if prev_ts is None else max(0.0, ts_val - prev_ts)
            prev_ts = ts_val
        else:
            flight = 0.0

        events.append(
            {
                "key": key,
                "hold_time_ms": hold,
                "flight_time_ms": flight,
                "pressure": pressure,
            }
        )

    return events


def _scrolls_to_fixed(events: List[dict], max_len: int = 20) -> np.ndarray:
    out = np.zeros((max_len, 6), dtype=np.float32)
    for i, ev in enumerate(events[:max_len]):
        velocity = float(ev.get("velocity_px_per_sec", 0.0))
        direction = float(ev.get("direction_deg", 0.0))
        distance = float(ev.get("distance_px", 0.0))
        pressure = float(ev.get("avg_pressure", 0.5))
        density = float(ev.get("pixel_density_dpi", 0.0))

        rad = np.deg2rad(direction)
        out[i, 0] = velocity
        out[i, 1] = np.sin(rad)
        out[i, 2] = np.cos(rad)
        out[i, 3] = distance
        out[i, 4] = pressure
        out[i, 5] = density
    return out


def _touch_points_to_scroll_events(points: pd.DataFrame) -> List[dict]:
    cols = {c.lower(): c for c in points.columns}

    def col(*names: str) -> Optional[str]:
        for name in names:
            if name in cols:
                return cols[name]
        return None

    time_col = col("time", "timestamp", "ts")
    action_col = col("action", "event", "event_type")
    x_col = col("x", "pos_x")
    y_col = col("y", "pos_y")
    pressure_col = col("pressure", "force")

    if time_col is None or x_col is None or y_col is None:
        return []

    df = points.sort_values(time_col)
    actions = df[action_col] if action_col else pd.Series(np.ones(len(df)), index=df.index)

    events: List[dict] = []
    down_row = None
    pressure_track: List[float] = []

    for idx, row in df.iterrows():
        action = int(row.get(action_col, 1)) if action_col else 1

        if action == 0:
            down_row = row
            pressure_track = [float(row.get(pressure_col, 0.5)) if pressure_col else 0.5]
            continue

        if down_row is None:
            continue

        pressure_track.append(float(row.get(pressure_col, 0.5)) if pressure_col else 0.5)

        if action == 1 or idx == df.index[-1]:
            dt = float(row[time_col]) - float(down_row[time_col])
            dx = float(row[x_col]) - float(down_row[x_col])
            dy = float(row[y_col]) - float(down_row[y_col])
            dist = float(np.sqrt(dx * dx + dy * dy))

            if dt <= 0.0 or dist <= 0.0:
                down_row = None
                pressure_track = []
                continue

            deg = float(np.degrees(np.arctan2(dy, dx)))
            vel = dist / dt * 1000.0

            events.append(
                {
                    "velocity_px_per_sec": vel,
                    "direction_deg": deg,
                    "distance_px": dist,
                    "avg_pressure": float(np.mean(pressure_track) if pressure_track else 0.5),
                    "pixel_density_dpi": 420.0,
                }
            )

            down_row = None
            pressure_track = []

    return events


def _resample_or_pad_imu(events: List[dict], target_len: int = 250) -> np.ndarray:
    arr = np.zeros((len(events), 5), dtype=np.float32)
    for i, ev in enumerate(events):
        arr[i, 0] = float(ev.get("gyro_x", 0.0))
        arr[i, 1] = float(ev.get("gyro_y", 0.0))
        arr[i, 2] = float(ev.get("gyro_z", 0.0))
        arr[i, 3] = float(ev.get("tilt_pitch", 0.0))
        arr[i, 4] = float(ev.get("tilt_roll", 0.0))

    if len(arr) == 0:
        return np.zeros((target_len, 5), dtype=np.float32)

    if len(arr) >= target_len:
        start = max(0, (len(arr) - target_len) // 2)
        return arr[start:start + target_len].astype(np.float32)

    reps = int(np.ceil(target_len / len(arr)))
    tiled = np.tile(arr, (reps, 1))
    return tiled[:target_len].astype(np.float32)


def parse_keyrecs(data_dir: str | Path = Path("datasets") / "keyrecs") -> Dict[str, List[SessionSample]]:
    root = Path(data_dir)
    if not root.exists():
        print(f"[WARN] KeyRecs folder missing at {root}. Skipping.")
        return {}

    user_sessions: Dict[str, List[SessionSample]] = {}
    csv_files = sorted(root.rglob("*.csv"))

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue

        if df.empty:
            continue

        lowered = {str(c).lower().strip(): c for c in df.columns}

        # KeyRecs fixed-text file
        if "participant" in lowered and "repetition" in lowered and any(str(c).startswith("DU.") for c in df.columns):
            for _, row in df.iterrows():
                participant = str(row[lowered["participant"]])
                uid = f"keyrecs::{participant}"
                events = _keystrokes_from_fixed_text_row(row)
                if len(events) < 4:
                    continue
                user_sessions.setdefault(uid, []).append(
                    SessionSample(
                        user_id=uid,
                        keystrokes=_keystrokes_to_fixed(events),
                        scrolls=_zero_scrolls(),
                        imu=_zero_imu(),
                        has_keystroke=True,
                        has_scroll=False,
                        has_imu=False,
                        source="keyrecs",
                    )
                )
            continue

        # KeyRecs free-text file
        if "participant" in lowered and "session" in lowered and "key1" in lowered and "key2" in lowered:
            pcol = lowered["participant"]
            scol = lowered["session"]
            renamed = df.rename(columns={pcol: "participant", scol: "session"})
            for (participant, session), grp in renamed.groupby(["participant", "session"]):
                uid = f"keyrecs::{participant}"
                events = _keystrokes_from_free_text_group(grp)
                if len(events) < 4:
                    continue
                for i in range(0, len(events), 8):
                    chunk = events[i:i + 8]
                    if not chunk:
                        continue
                    user_sessions.setdefault(uid, []).append(
                        SessionSample(
                            user_id=uid,
                            keystrokes=_keystrokes_to_fixed(chunk),
                            scrolls=_zero_scrolls(),
                            imu=_zero_imu(),
                            has_keystroke=True,
                            has_scroll=False,
                            has_imu=False,
                            source="keyrecs",
                        )
                    )
            continue

        # Fallback generic parser
        uid_col = lowered.get("user_id") or lowered.get("userid") or lowered.get("user")
        sess_col = lowered.get("session_id") or lowered.get("session") or lowered.get("phrase_id")
        if uid_col is None:
            continue

        grouped = df.groupby([uid_col, sess_col]) if sess_col else [(None, df)]
        for grp_key, group_df in grouped:
            uid_val = str(grp_key[0]) if sess_col else str(group_df[uid_col].iloc[0])
            uid = f"keyrecs::{uid_val}"
            events = _events_from_keystroke_frame(group_df)
            if len(events) < 4:
                continue
            user_sessions.setdefault(uid, []).append(
                SessionSample(
                    user_id=uid,
                    keystrokes=_keystrokes_to_fixed(events),
                    scrolls=_zero_scrolls(),
                    imu=_zero_imu(),
                    has_keystroke=True,
                    has_scroll=False,
                    has_imu=False,
                    source="keyrecs",
                )
            )

    return user_sessions


def parse_touchalytics(data_dir: str | Path = Path("datasets") / "touchalytics") -> Dict[str, List[SessionSample]]:
    root = Path(data_dir)
    if not root.exists():
        print(f"[WARN] Touchalytics folder missing at {root}. Skipping.")
        return {}

    csv_files = sorted(root.rglob("*.csv"))
    if not csv_files:
        print(f"[WARN] No Touchalytics CSV files found in {root}. Skipping.")
        return {}

    user_sessions: Dict[str, List[SessionSample]] = {}

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue

        if df.empty:
            continue

        lowered = {str(c).lower().strip(): c for c in df.columns}

        # Touchalytics official dump is often headerless 11-column CSV.
        expected = [
            "phone_id", "user_id", "doc_id", "time", "action", "orientation",
            "x", "y", "pressure", "area", "finger_orientation",
        ]
        if all(str(c).isdigit() for c in df.columns[: min(5, len(df.columns))]) and len(df.columns) >= 11:
            df.columns = expected[: len(df.columns)]
            lowered = {str(c).lower().strip(): c for c in df.columns}

        if "user_id" not in lowered and len(df.columns) >= 11:
            # Retry explicit headerless read for robustness.
            try:
                df2 = pd.read_csv(csv_path, header=None, names=expected, low_memory=False)
                if not df2.empty:
                    df = df2
                    lowered = {str(c).lower().strip(): c for c in df.columns}
            except Exception:
                pass

        uid_col = lowered.get("user_id") or lowered.get("userid") or lowered.get("user") or lowered.get("phone_id")
        doc_col = lowered.get("doc_id") or lowered.get("document_id") or lowered.get("session_id")

        if uid_col is None:
            continue

        if doc_col is None:
            df["_doc_id"] = "single"
            doc_col = "_doc_id"

        for (uid_raw, doc_raw), grp in df.groupby([uid_col, doc_col]):
            uid = f"touchalytics::{uid_raw}"
            events = _touch_points_to_scroll_events(grp)
            if len(events) < 1:
                continue

            user_sessions.setdefault(uid, []).append(
                SessionSample(
                    user_id=uid,
                    keystrokes=_zero_keys(),
                    scrolls=_scrolls_to_fixed(events),
                    imu=_zero_imu(),
                    has_keystroke=False,
                    has_scroll=True,
                    has_imu=False,
                    source="touchalytics",
                )
            )

    return user_sessions


def load_sessions_from_sqlite(
    db_path: str | Path,
    min_keys: int = 40,
    min_scrolls: int = 5,
    min_imu: int = 250,
) -> Dict[str, List[SessionSample]]:
    conn = _db_connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT user_id, keystrokes_json, scrolls_json, imu_json
        FROM raw_enrollment
        """
    )

    grouped: Dict[str, List[SessionSample]] = {}

    rows = cursor.fetchall()
    conn.close()

    for user_id, keys_json, scrolls_json, imu_json in rows:
        key_events = _safe_json_load(keys_json)
        scroll_events = _safe_json_load(scrolls_json)
        imu_events = _safe_json_load(imu_json)

        if len(key_events) < min_keys or len(scroll_events) < min_scrolls or len(imu_events) < min_imu:
            continue

        sample = SessionSample(
            user_id=str(user_id),
            keystrokes=_keystrokes_to_fixed(key_events),
            scrolls=_scrolls_to_fixed(scroll_events),
            imu=_resample_or_pad_imu(imu_events),
            has_keystroke=True,
            has_scroll=True,
            has_imu=True,
            source="booth",
        )
        grouped.setdefault(f"booth::{str(user_id)}", []).append(sample)

    return grouped


def merge_all_sources(
    sqlite_db_path: str | Path,
    datasets_root: str | Path = Path("datasets"),
) -> Dict[str, List[SessionSample]]:
    datasets_root = Path(datasets_root)

    booth = load_sessions_from_sqlite(sqlite_db_path)
    keyrecs = parse_keyrecs(datasets_root / "keyrecs")
    touchalytics = parse_touchalytics(datasets_root / "touchalytics")

    merged: Dict[str, List[SessionSample]] = {}
    for source_dict in (booth, keyrecs, touchalytics):
        for uid, sessions in source_dict.items():
            merged.setdefault(uid, []).extend(sessions)

    return merged


def fit_feature_scalers(user_sessions: Dict[str, List[SessionSample]]) -> Dict[str, StandardScaler]:
    key_num = []
    scroll = []
    imu = []

    for sessions in user_sessions.values():
        for s in sessions:
            if s.has_keystroke:
                key_num.append(s.keystrokes[:, 1:4])
            if s.has_scroll:
                scroll.append(s.scrolls)
            if s.has_imu:
                imu.append(s.imu)

    if not key_num:
        raise ValueError("No data available to fit scalers.")

    key_num_arr = np.concatenate(key_num, axis=0)
    scroll_arr = np.concatenate(scroll, axis=0) if scroll else np.zeros((1, 6), dtype=np.float32)
    imu_arr = np.concatenate(imu, axis=0) if imu else np.zeros((1, 5), dtype=np.float32)

    return {
        "keys_num": StandardScaler().fit(key_num_arr),
        "scrolls": StandardScaler().fit(scroll_arr),
        "imu": StandardScaler().fit(imu_arr),
    }


def apply_scalers(
    user_sessions: Dict[str, List[SessionSample]],
    scalers: Dict[str, StandardScaler],
) -> Dict[str, List[SessionSample]]:
    out: Dict[str, List[SessionSample]] = {}
    for uid, sessions in user_sessions.items():
        out[uid] = []
        for s in sessions:
            k = s.keystrokes.copy()
            sc = s.scrolls.copy()
            im = s.imu.copy()

            if s.has_keystroke:
                k[:, 1:4] = scalers["keys_num"].transform(k[:, 1:4])
            if s.has_scroll:
                sc = scalers["scrolls"].transform(sc)
            if s.has_imu:
                im = scalers["imu"].transform(im)

            out[uid].append(
                SessionSample(
                    user_id=uid,
                    keystrokes=k,
                    scrolls=sc,
                    imu=im,
                    has_keystroke=s.has_keystroke,
                    has_scroll=s.has_scroll,
                    has_imu=s.has_imu,
                    source=s.source,
                )
            )
    return out


def build_user_index(user_sessions: Dict[str, List[SessionSample]]) -> Dict[str, int]:
    users = sorted(user_sessions.keys())
    return {u: i for i, u in enumerate(users)}


def split_users(
    user_sessions: Dict[str, List[SessionSample]],
    train_ratio: float = 0.75,
    val_ratio: float = 0.125,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    rng = np.random.default_rng(seed)
    users = np.array(sorted(user_sessions.keys()))
    rng.shuffle(users)

    n = len(users)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio)) if n >= 3 else 0

    train_users = users[:n_train].tolist()
    val_users = users[n_train:n_train + n_val].tolist()
    test_users = users[n_train + n_val:].tolist()

    if not test_users and len(train_users) > 1:
        test_users = [train_users.pop()]

    return train_users, val_users, test_users


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    db_path = root / "datasets" / "sentinel_lab.db"
    if not db_path.exists():
        db_path = root.parent.parent / "fastapiCollector" / "sentinel_lab.db"

    merged = merge_all_sources(db_path, root / "datasets")
    by_source = {"booth": set(), "keyrecs": set(), "touchalytics": set()}

    for uid, sessions in merged.items():
        if not sessions:
            continue
        by_source[sessions[0].source].add(uid)

    print(f"Booth users: {len(by_source['booth'])}")
    print(f"KeyRecs users: {len(by_source['keyrecs'])}")
    print(f"Touchalytics users: {len(by_source['touchalytics'])}")
    print(f"Total users merged: {len(merged)}")
