#!/usr/bin/env python3
"""
test_api.py — End-to-end smoke test for the Sentinel backend.

Usage:
    python test_api.py               # assumes localhost:8000
    python test_api.py 192.168.1.42  # custom IP
"""

import sys
import json
import time
import random
import requests

BASE = f"http://{sys.argv[1] if len(sys.argv) > 1 else 'localhost'}:8000"


def rnd(lo, hi):
    return round(random.uniform(lo, hi), 3)


def make_keystrokes(n=10):
    phrases = ["balanced_random", "repetition", "numeric"]
    return [
        {
            "phrase_id": random.choice(phrases),
            "repetition_index": i % 5,
            "key": random.choice("abcdefghijklmnopqrstuvwxyz"),
            "hold_time_ms": rnd(80, 200),
            "flight_time_ms": rnd(50, 150) if i > 0 else 0,
            "pressure": rnd(0.3, 0.9),
            "pixel_density": 420.0,
        }
        for i in range(n)
    ]


def make_scrolls(n=12):
    return [
        {
            "velocity_px_per_sec": rnd(200, 800),
            "direction_deg": random.choice([0, 180]),
            "distance_px": rnd(200, 1200),
            "avg_pressure": rnd(0.4, 0.8),
            "pixel_density_dpi": 420.0,
            "timestamp": int(time.time() * 1000) + i * 500,
        }
        for i in range(n)
    ]


def make_imu(n=150):
    return [
        {
            "gyro_x": rnd(-0.2, 0.2),
            "gyro_y": rnd(-0.2, 0.2),
            "gyro_z": rnd(-0.05, 0.05),
            "tilt_pitch": rnd(10, 25),
            "tilt_roll": rnd(-15, 15),
            "timestamp": int(time.time() * 1000) + i * 20,
        }
        for i in range(n)
    ]


def check(label, resp):
    ok = "✅" if resp.status_code < 300 else "❌"
    print(f"{ok} [{resp.status_code}] {label}")
    try:
        data = resp.json()
        print(f"   {json.dumps(data, indent=2)[:300]}")
    except Exception:
        print(f"(non-JSON body)")
    return resp


print(f"\n=== Sentinel API Test  →  {BASE} ===\n")

# 1. Health check
r = check("GET /health", requests.get(f"{BASE}/health"))

# 2. Create user
r = check(
    "POST /users/create",
    requests.post(f"{BASE}/users/create", json={"name": "TestUser_Alice"}),
)
user_id = r.json()["user_id"]

# 3. Enroll
payload = {
    "user_id": user_id,
    "user_name": "TestUser_Alice",
    "keystrokes": make_keystrokes(40),
    "scrolls": make_scrolls(12),
    "imu": make_imu(150),
}
r = check("POST /enroll", requests.post(f"{BASE}/enroll", json=payload))

# 4. Verify same user (should be low risk)
verify_payload = {
    "user_id": user_id,
    "keystrokes": make_keystrokes(20),
    "scrolls": make_scrolls(10),
    "imu": make_imu(50),
}
r = check(
    "POST /verify (same user — expect low risk)",
    requests.post(f"{BASE}/verify", json=verify_payload),
)

# 5. Verify unknown user (should be critical)
r = check(
    "POST /verify (unknown user — expect locked)",
    requests.post(
        f"{BASE}/verify",
        json={**verify_payload, "user_id": "nonexistent-user-000"},
    ),
)

# 6. Get user info
r = check(f"GET /user/{user_id}", requests.get(f"{BASE}/user/{user_id}"))

# 7. Get user raw data
r = check(f"GET /user/{user_id}/data", requests.get(f"{BASE}/user/{user_id}/data"))

print("\n=== Done ===\n")