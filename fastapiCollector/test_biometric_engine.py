# test_biometric_engine.py — Unit tests for the Gaussian biometric engine

import json
import sys
sys.path.insert(0, ".")

from biometric_engine import (
    extract_keystroke_features,
    extract_scroll_features,
    extract_imu_features,
    build_gaussian_profile,
    GaussianProfile,
    compute_fused_score,
    update_profile_drift,
    mahalanobis_distance,
    distance_to_similarity,
    check_timestamp_jitter,
)
import numpy as np


def test_keystroke_features():
    keystrokes = [
        {"key": "v", "hold_time_ms": 80, "flight_time_ms": 0, "pressure": 0.5, "phrase_id": "balanced_random"},
        {"key": "k", "hold_time_ms": 60, "flight_time_ms": 120, "pressure": 0.6, "phrase_id": "balanced_random"},
        {"key": "e", "hold_time_ms": 70, "flight_time_ms": 100, "pressure": 0.5, "phrase_id": "balanced_random"},
    ]
    features = extract_keystroke_features(keystrokes)
    assert len(features) == 6, f"Expected 6 features, got {len(features)}"
    assert features[0] > 0, "Mean hold should be > 0"
    print(f"✓ Keystroke features: {features}")


def test_scroll_features():
    scrolls = [
        {"velocity_px_per_sec": 500, "distance_px": 200, "direction_deg": 180, "timestamp": 1000},
        {"velocity_px_per_sec": 600, "distance_px": 250, "direction_deg": 180, "timestamp": 1200},
        {"velocity_px_per_sec": 400, "distance_px": 150, "direction_deg": 0, "timestamp": 1500},
    ]
    features = extract_scroll_features(scrolls)
    assert len(features) == 7, f"Expected 7 features, got {len(features)}"
    assert features[4] > 0, "Direction ratio should be > 0"  # 2/3 downward
    print(f"✓ Scroll features: {features}")


def test_imu_features():
    imu = [
        {"gyro_x": 0.1, "gyro_y": -0.2, "gyro_z": 0.05, "tilt_pitch": 15, "tilt_roll": 5}
        for _ in range(50)
    ]
    features = extract_imu_features(imu)
    assert len(features) == 14, f"Expected 14 features, got {len(features)}"
    print(f"✓ IMU features: {features}")


def test_gaussian_profile():
    # Create 3 enrollment sessions with slightly different data
    sessions = []
    for i in range(3):
        sessions.append({
            "keystrokes": [
                {"key": "v", "hold_time_ms": 80 + i * 5, "flight_time_ms": 0, "pressure": 0.5},
                {"key": "k", "hold_time_ms": 60 + i * 3, "flight_time_ms": 120 + i * 10, "pressure": 0.6},
            ],
            "scrolls": [
                {"velocity_px_per_sec": 500 + i * 20, "distance_px": 200, "direction_deg": 180, "timestamp": 1000},
                {"velocity_px_per_sec": 600 + i * 15, "distance_px": 250, "direction_deg": 0, "timestamp": 1200},
            ],
            "imu": [
                {"gyro_x": 0.1 + i * 0.01, "gyro_y": -0.2, "gyro_z": 0.05, "tilt_pitch": 15, "tilt_roll": 5}
                for _ in range(30)
            ],
        })

    profile = build_gaussian_profile(sessions)

    assert profile.keystroke is not None, "Keystroke profile should exist"
    assert profile.scroll is not None, "Scroll profile should exist"
    assert profile.imu is not None, "IMU profile should exist"
    assert profile.enrollment_sessions == 3

    # Test serialization roundtrip
    json_str = profile.to_json()
    loaded = GaussianProfile.from_json(json_str)
    assert loaded.enrollment_sessions == 3
    np.testing.assert_array_almost_equal(loaded.keystroke.mean, profile.keystroke.mean)
    print(f"✓ Gaussian profile built: ks_dims={len(profile.keystroke.mean)}, "
          f"sc_dims={len(profile.scroll.mean)}, imu_dims={len(profile.imu.mean)}")

    return profile, sessions


def test_verification():
    profile, sessions = test_gaussian_profile()

    # Verify with genuine data (from enrollment)
    genuine_result = compute_fused_score(
        profile=profile,
        keystrokes=sessions[0]["keystrokes"],
        scrolls=sessions[0]["scrolls"],
        imu_samples=sessions[0]["imu"],
    )
    print(f"\n✓ Genuine verification:")
    print(f"  Trust score: {genuine_result.trust_score:.4f}")
    print(f"  Trust level: {genuine_result.trust_level}")
    print(f"  Fused similarity: {genuine_result.fused_similarity:.4f}")
    assert genuine_result.trust_score > 0.5, "Genuine user should have high trust"

    # Verify with impostor data (very different)
    impostor_result = compute_fused_score(
        profile=profile,
        keystrokes=[
            {"key": "a", "hold_time_ms": 200, "flight_time_ms": 500, "pressure": 0.3},
            {"key": "b", "hold_time_ms": 300, "flight_time_ms": 400, "pressure": 0.2},
        ],
        scrolls=[
            {"velocity_px_per_sec": 2000, "distance_px": 800, "direction_deg": 0, "timestamp": 5000},
            {"velocity_px_per_sec": 1500, "distance_px": 600, "direction_deg": 0, "timestamp": 5500},
        ],
        imu_samples=[
            {"gyro_x": 2.0, "gyro_y": -3.0, "gyro_z": 1.5, "tilt_pitch": 60, "tilt_roll": 45}
            for _ in range(30)
        ],
    )
    print(f"\n✓ Impostor verification:")
    print(f"  Trust score: {impostor_result.trust_score:.4f}")
    print(f"  Trust level: {impostor_result.trust_level}")
    print(f"  Fused similarity: {impostor_result.fused_similarity:.4f}")
    assert impostor_result.trust_score < genuine_result.trust_score, \
        "Impostor should have lower trust than genuine"


def test_exponential_smoothing():
    profile, sessions = test_gaussian_profile()

    # Simulate continuous auth with smoothing
    trust = None
    failures = 0
    print(f"\n✓ Exponential smoothing (genuine user):")
    for i in range(5):
        result = compute_fused_score(
            profile=profile,
            keystrokes=sessions[i % len(sessions)]["keystrokes"],
            scrolls=sessions[i % len(sessions)]["scrolls"],
            imu_samples=sessions[i % len(sessions)]["imu"],
            previous_trust=trust,
            previous_consecutive_failures=failures,
        )
        trust = result.trust_score
        failures = result.consecutive_failures
        print(f"  Check {i+1}: trust={trust:.4f} level={result.trust_level} fails={failures}")
    assert trust > 0.65, "Genuine user should stay authenticated after 5 checks"


def test_impostor_convergence():
    """Verify that an impostor reaches session_terminate within 7 checks."""
    profile, sessions = test_gaussian_profile()

    # Impostor data — significantly different from enrollment
    impostor_ks = [
        {"key": "x", "hold_time_ms": 200, "flight_time_ms": 500, "pressure": 0.3},
        {"key": "y", "hold_time_ms": 250, "flight_time_ms": 400, "pressure": 0.2},
    ]
    impostor_sc = [
        {"velocity_px_per_sec": 1500, "distance_px": 600, "direction_deg": 0, "timestamp": 5000},
        {"velocity_px_per_sec": 1200, "distance_px": 500, "direction_deg": 0, "timestamp": 5500},
    ]
    impostor_imu = [
        {"gyro_x": 1.5, "gyro_y": -2.0, "gyro_z": 1.0, "tilt_pitch": 50, "tilt_roll": 35}
        for _ in range(30)
    ]

    # Start from a high-trust state (as if genuine user just left)
    trust = 0.95
    failures = 0
    locked_at = None

    print(f"\n✓ Impostor convergence simulation (starting trust=0.95):")
    for i in range(10):
        result = compute_fused_score(
            profile=profile,
            keystrokes=impostor_ks,
            scrolls=impostor_sc,
            imu_samples=impostor_imu,
            previous_trust=trust,
            previous_consecutive_failures=failures,
        )
        trust = result.trust_score
        failures = result.consecutive_failures
        print(f"  Check {i+1}: trust={trust:.4f} sim={result.fused_similarity:.4f} "
              f"level={result.trust_level} fails={failures}")

        if result.trust_level == "session_terminate" and locked_at is None:
            locked_at = i + 1

    assert locked_at is not None, "Impostor should reach session_terminate"
    assert locked_at <= 7, f"Should lock within 7 checks, locked at check {locked_at}"
    print(f"  >> LOCKED OUT at check {locked_at} (~{locked_at * 15}s)")


def test_profile_drift():
    profile, sessions = test_gaussian_profile()
    original_ks_mean = profile.keystroke.mean.copy()

    # Update with high trust
    updated = update_profile_drift(
        profile, sessions[0]["keystrokes"], sessions[0]["scrolls"],
        sessions[0]["imu"], current_trust=0.9,
    )
    assert not np.array_equal(updated.keystroke.mean, original_ks_mean), \
        "Profile should drift when trust is high"

    # Try update with low trust
    low_trust_mean = updated.keystroke.mean.copy()
    updated2 = update_profile_drift(
        updated, sessions[0]["keystrokes"], sessions[0]["scrolls"],
        sessions[0]["imu"], current_trust=0.3,
    )
    np.testing.assert_array_equal(updated2.keystroke.mean, low_trust_mean)
    print(f"✓ Profile drift: updated with high trust, blocked with low trust")


def test_liveness():
    # Regular timestamps (bot)
    regular = [1000, 1100, 1200, 1300, 1400]
    assert not check_timestamp_jitter(regular, min_jitter_ms=20), \
        "Regular timestamps should fail liveness"

    # Irregular timestamps (human)
    irregular = [1000, 1087, 1193, 1342, 1401]
    assert check_timestamp_jitter(irregular, min_jitter_ms=20), \
        "Irregular timestamps should pass liveness"
    print(f"✓ Liveness check: bot=FAIL, human=PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Biometric Engine")
    print("=" * 60)

    test_keystroke_features()
    test_scroll_features()
    test_imu_features()
    test_gaussian_profile()
    test_verification()
    test_exponential_smoothing()
    test_impostor_convergence()
    test_profile_drift()
    test_liveness()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
