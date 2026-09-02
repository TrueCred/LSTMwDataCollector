# biometric_engine.py — Gaussian Profile Continuous Behavioral Authentication Engine
#
# This implements a Mahalanobis-distance-based biometric verification system that:
# 1. Extracts feature vectors per modality (keystroke, scroll, IMU)
# 2. Builds Gaussian profiles (mean + variance) during enrollment
# 3. Uses Mahalanobis distance for verification scoring
# 4. Fuses multi-modal scores with configurable weights
# 5. Maintains a continuous trust score with exponential smoothing
# 6. Supports safe profile drift updates (only when trust is high)

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ── Configuration ──────────────────────────────────────────────────────────────

# Modality fusion weights
W_KEYSTROKE = 0.50
W_SCROLL    = 0.25
W_IMU       = 0.25

# Mahalanobis distance → similarity conversion scale
# D is normalized by sqrt(dim), so D≈1 means 1σ away on average
SIMILARITY_SCALE = 3.0   # S = exp(-D / scale)  — stricter: 2σ → 51% similarity

# Trust score exponential smoothing factor
TRUST_ALPHA = 0.55        # new_trust = α * live_sim + (1-α) * old_trust
                          # Higher α = faster reaction to behavioral changes

# Trust thresholds
TRUST_AUTHENTICATED = 0.65   # green — genuine user
TRUST_SOFT_CHALLENGE = 0.40  # yellow — re-verify soon
TRUST_HARD_CHALLENGE = 0.28  # red — step-up auth required
# Below 0.28 → session terminate

# Consecutive failure penalty — breaks the exponential smoothing convergence plateau.
# When fused_similarity is below FAIL_SIM_THRESHOLD for N consecutive checks,
# apply a multiplicative penalty to the trust score so it keeps dropping.
FAIL_SIM_THRESHOLD = 0.45     # Similarity below this counts as a "failure"
FAIL_COUNT_BEFORE_PENALTY = 3  # Start penalizing after this many consecutive failures
FAIL_PENALTY_MULTIPLIER = 0.80 # trust *= 0.80 per failure beyond the threshold

# Profile drift update rate (only applied when trust > 0.80)
DRIFT_BETA = 0.95
DRIFT_TRUST_MIN = 0.80

# Minimum regularization for variance
# MUST be small so low-magnitude features (gyro 0-2 rad/s) remain discriminative.
MIN_VARIANCE_FLOOR = 0.05  # Absolute minimum std

# Minimum enrollment sessions for robust σ
MIN_ENROLLMENT_SESSIONS = 1  # Relaxed for initial testing; ideally 3+

# Per-phrase complexity weights (more unique n-grams → higher weight)
PHRASE_WEIGHTS = {
    "balanced_random": 1.2,   # 'vkerjpwu' — diverse n-grams
    "repetition":      0.8,   # 'kkjjkkjj' — repetitive, less discriminative
    "numeric":         1.0,   # '13792846' — moderate
}


# ── Feature Extraction ─────────────────────────────────────────────────────────

def extract_keystroke_features(keystrokes: list[dict]) -> np.ndarray:
    """
    Extract statistical keystroke features from a list of keystroke events.

    Always produces a fixed 6-D global feature vector:
      [μ_hold, σ_hold, μ_flight, σ_flight, μ_pressure, σ_pressure]

    We intentionally ignore phrase_id grouping to ensure enrollment features
    (with phrase_id) and verification features (without) have the same shape.

    Returns:
        1-D numpy array of 6 features.
    """
    if not keystrokes:
        return np.zeros(6, dtype=np.float64)

    holds     = [max(0.0, float(e.get("hold_time_ms") or 0))   for e in keystrokes]
    flights   = [max(0.0, float(e.get("flight_time_ms") or 0))  for e in keystrokes]
    pressures = [max(0.0, float(e.get("pressure") or 0.5))      for e in keystrokes]

    return np.array([
        np.mean(holds)      if holds     else 0.0,
        np.std(holds)       if holds     else 0.0,
        np.mean(flights)    if flights   else 0.0,
        np.std(flights)     if flights   else 0.0,
        np.mean(pressures)  if pressures else 0.0,
        np.std(pressures)   if pressures else 0.0,
    ], dtype=np.float64)


def extract_scroll_features(scrolls: list[dict]) -> np.ndarray:
    """
    Extract scroll behavior features.

    Returns 7-D vector:
      [μ_vel, σ_vel, μ_dist, σ_dist, direction_ratio, μ_interval, σ_interval]
    """
    if not scrolls or len(scrolls) < 2:
        return np.zeros(7, dtype=np.float64)

    velocities = []
    distances  = []
    downward   = 0
    timestamps = []

    for s in scrolls:
        vel = float(s.get("velocity_px_per_sec") or 0)
        dist = float(s.get("distance_px") or 0)
        direction = float(s.get("direction_deg") or 0)
        ts = float(s.get("timestamp") or 0)

        velocities.append(vel)
        distances.append(dist)
        timestamps.append(ts)
        if 90 <= direction <= 270:  # downward scroll
            downward += 1

    direction_ratio = downward / len(scrolls) if scrolls else 0.5

    # Inter-event intervals
    intervals = []
    sorted_ts = sorted(timestamps)
    for i in range(1, len(sorted_ts)):
        dt = sorted_ts[i] - sorted_ts[i - 1]
        if 0 < dt < 10000:  # ignore unreasonable gaps
            intervals.append(dt)

    return np.array([
        np.mean(velocities)  if velocities else 0.0,
        np.std(velocities)   if velocities else 0.0,
        np.mean(distances)   if distances  else 0.0,
        np.std(distances)    if distances  else 0.0,
        direction_ratio,
        np.mean(intervals)   if intervals  else 0.0,
        np.std(intervals)    if intervals  else 0.0,
    ], dtype=np.float64)


def _fft_dominant_freq(signal: list[float], sample_rate: float = 10.0) -> float:
    """Compute dominant frequency from FFT of a time-series signal."""
    if len(signal) < 4:
        return 0.0
    arr = np.array(signal, dtype=np.float64)
    arr = arr - np.mean(arr)  # remove DC
    fft_vals = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(len(arr), d=1.0 / sample_rate)
    # Ignore DC component (index 0)
    if len(fft_vals) > 1:
        idx = np.argmax(fft_vals[1:]) + 1
        return float(freqs[idx])
    return 0.0


def extract_imu_features(imu_samples: list[dict]) -> np.ndarray:
    """
    Extract IMU features from gyroscope and tilt data.

    Returns 14-D vector:
      [μ_gx, σ_gx, μ_gy, σ_gy, μ_gz, σ_gz,
       μ_pitch, σ_pitch, μ_roll, σ_roll,
       fft_dom_gx, fft_dom_gy, fft_dom_gz,
       rms_gyro_magnitude]
    """
    if not imu_samples:
        return np.zeros(14, dtype=np.float64)

    gx = [float(s.get("gyro_x") or 0) for s in imu_samples]
    gy = [float(s.get("gyro_y") or 0) for s in imu_samples]
    gz = [float(s.get("gyro_z") or 0) for s in imu_samples]
    pitches = [float(s.get("tilt_pitch") or 0) for s in imu_samples]
    rolls   = [float(s.get("tilt_roll") or 0)  for s in imu_samples]

    # Per-axis stats
    stats = [
        np.mean(gx), np.std(gx),
        np.mean(gy), np.std(gy),
        np.mean(gz), np.std(gz),
        np.mean(pitches), np.std(pitches),
        np.mean(rolls), np.std(rolls),
    ]

    # FFT dominant frequencies (estimate 10 Hz sample rate)
    stats.append(_fft_dominant_freq(gx))
    stats.append(_fft_dominant_freq(gy))
    stats.append(_fft_dominant_freq(gz))

    # RMS of gyro magnitude
    magnitudes = [math.sqrt(x ** 2 + y ** 2 + z ** 2) for x, y, z in zip(gx, gy, gz)]
    rms = math.sqrt(np.mean(np.array(magnitudes) ** 2)) if magnitudes else 0.0
    stats.append(rms)

    return np.array(stats, dtype=np.float64)


# ── Gaussian Profile ───────────────────────────────────────────────────────────

@dataclass
class ModalityProfile:
    """Gaussian profile for one modality: mean + standard deviation per feature."""
    mean: np.ndarray     # μ — shape (D,)
    std: np.ndarray      # σ — shape (D,)
    n_sessions: int = 0  # How many enrollment sessions contributed

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_sessions": self.n_sessions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModalityProfile":
        return cls(
            mean=np.array(d["mean"], dtype=np.float64),
            std=np.array(d["std"], dtype=np.float64),
            n_sessions=d.get("n_sessions", 1),
        )


@dataclass
class GaussianProfile:
    """Complete user biometric profile across all modalities."""
    keystroke: Optional[ModalityProfile] = None
    scroll: Optional[ModalityProfile] = None
    imu: Optional[ModalityProfile] = None
    enrollment_sessions: int = 0

    def to_json(self) -> str:
        d = {
            "version": 2,
            "enrollment_sessions": self.enrollment_sessions,
            "keystroke": self.keystroke.to_dict() if self.keystroke else None,
            "scroll": self.scroll.to_dict() if self.scroll else None,
            "imu": self.imu.to_dict() if self.imu else None,
        }
        return json.dumps(d)

    @classmethod
    def from_json(cls, raw: str) -> "GaussianProfile":
        d = json.loads(raw)
        return cls(
            enrollment_sessions=d.get("enrollment_sessions", 1),
            keystroke=ModalityProfile.from_dict(d["keystroke"]) if d.get("keystroke") else None,
            scroll=ModalityProfile.from_dict(d["scroll"]) if d.get("scroll") else None,
            imu=ModalityProfile.from_dict(d["imu"]) if d.get("imu") else None,
        )


def build_gaussian_profile(
    sessions: list[dict],
) -> GaussianProfile:
    """
    Build a Gaussian profile from multiple enrollment sessions.

    Each session should have: {keystrokes: [...], scrolls: [...], imu: [...]}

    Uses all sessions to compute per-feature mean and std across sessions.
    """
    ks_features = []
    sc_features = []
    imu_features = []

    for session in sessions:
        ks_f = extract_keystroke_features(session.get("keystrokes", []))
        sc_f = extract_scroll_features(session.get("scrolls", []))
        imu_f = extract_imu_features(session.get("imu", []))

        ks_features.append(ks_f)
        sc_features.append(sc_f)
        imu_features.append(imu_f)

    n = len(sessions)
    profile = GaussianProfile(enrollment_sessions=n)

    if ks_features:
        # Ensure all have the same length (pad shorter ones)
        max_len = max(len(f) for f in ks_features)
        padded = [np.pad(f, (0, max_len - len(f))) for f in ks_features]
        stacked = np.stack(padded)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        # Adaptive floor: at least 30% of mean magnitude or MIN_VARIANCE_FLOOR
        # Wider floor for single-session profiles — natural typing variation is large
        adaptive_floor = np.maximum(np.abs(mean) * 0.20, MIN_VARIANCE_FLOOR)
        std = np.maximum(std, adaptive_floor)
        profile.keystroke = ModalityProfile(mean=mean, std=std, n_sessions=n)

    if sc_features:
        stacked = np.stack(sc_features)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        adaptive_floor = np.maximum(np.abs(mean) * 0.20, MIN_VARIANCE_FLOOR)
        std = np.maximum(std, adaptive_floor)
        profile.scroll = ModalityProfile(mean=mean, std=std, n_sessions=n)

    if imu_features:
        stacked = np.stack(imu_features)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        adaptive_floor = np.maximum(np.abs(mean) * 0.20, MIN_VARIANCE_FLOOR)
        std = np.maximum(std, adaptive_floor)
        profile.imu = ModalityProfile(mean=mean, std=std, n_sessions=n)

    return profile


# ── Mahalanobis Distance ───────────────────────────────────────────────────────

def mahalanobis_distance(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    """
    Compute normalized diagonal Mahalanobis distance.

    D_raw = sqrt( Σᵢ [ (xᵢ - μᵢ)² / σᵢ² ] )
    D     = D_raw / sqrt(dim)   ← normalize so D≈1 means "1σ away on average"

    This is the simplified form using diagonal covariance
    (each feature independent), which is robust with small N.
    Normalized by sqrt(dim) to make scores comparable across modalities.
    """
    # Ensure same dimensionality
    dim = min(len(x), len(mean), len(std))
    if dim == 0:
        return 0.0
    x = x[:dim]
    mean = mean[:dim]
    std = std[:dim]

    # Clamp std to avoid division by zero
    safe_std = np.maximum(std, MIN_VARIANCE_FLOOR)

    z_scores = (x - mean) / safe_std
    d_raw = float(np.sqrt(np.sum(z_scores ** 2)))

    # Normalize by dimensionality so D≈1 means ~1σ per feature
    d = d_raw / math.sqrt(dim)

    return d


def distance_to_similarity(d: float, scale: float = SIMILARITY_SCALE) -> float:
    """
    Convert Mahalanobis distance to [0, 1] similarity score.

    S = exp(-D / scale_factor)
    """
    return float(np.exp(-d / scale))


# ── Multi-Modal Score Fusion ───────────────────────────────────────────────────

@dataclass
class ModalityScore:
    """Score breakdown for one modality."""
    distance: float = 0.0
    similarity: float = 1.0
    available: bool = False


@dataclass
class FusedScore:
    """Complete verification result."""
    keystroke: ModalityScore = field(default_factory=ModalityScore)
    scroll: ModalityScore = field(default_factory=ModalityScore)
    imu: ModalityScore = field(default_factory=ModalityScore)
    fused_similarity: float = 0.0
    trust_score: float = 0.0
    trust_level: str = "unknown"
    consecutive_failures: int = 0   # Tracks how many checks failed in a row

    def to_dict(self) -> dict:
        return {
            "keystroke": asdict(self.keystroke),
            "scroll": asdict(self.scroll),
            "imu": asdict(self.imu),
            "fused_similarity": round(self.fused_similarity, 4),
            "trust_score": round(self.trust_score, 4),
            "trust_level": self.trust_level,
            "consecutive_failures": self.consecutive_failures,
        }


def compute_fused_score(
    profile: GaussianProfile,
    keystrokes: list[dict],
    scrolls: list[dict],
    imu_samples: list[dict],
    previous_trust: Optional[float] = None,
    previous_consecutive_failures: int = 0,
) -> FusedScore:
    """
    Compute fused verification score by comparing live data against profile.

    Steps:
    1. Extract live features per modality
    2. Compute Mahalanobis distance per modality
    3. Convert to similarity scores
    4. Weighted fusion
    5. Exponential smoothing with previous trust
    6. Consecutive failure penalty (breaks convergence plateau)
    """
    result = FusedScore()
    total_weight = 0.0
    weighted_sim = 0.0

    # Keystroke scoring
    if profile.keystroke and keystrokes:
        live_ks = extract_keystroke_features(keystrokes)
        dim = min(len(live_ks), len(profile.keystroke.mean))
        if dim > 0:
            d = mahalanobis_distance(live_ks[:dim], profile.keystroke.mean[:dim], profile.keystroke.std[:dim])
            s = distance_to_similarity(d)
            result.keystroke = ModalityScore(distance=round(d, 4), similarity=round(s, 4), available=True)
            weighted_sim += W_KEYSTROKE * s
            total_weight += W_KEYSTROKE

    # Scroll scoring
    if profile.scroll and scrolls and len(scrolls) >= 2:
        live_sc = extract_scroll_features(scrolls)
        d = mahalanobis_distance(live_sc, profile.scroll.mean, profile.scroll.std)
        s = distance_to_similarity(d)
        result.scroll = ModalityScore(distance=round(d, 4), similarity=round(s, 4), available=True)
        weighted_sim += W_SCROLL * s
        total_weight += W_SCROLL

    # IMU scoring
    if profile.imu and imu_samples:
        live_imu = extract_imu_features(imu_samples)
        d = mahalanobis_distance(live_imu, profile.imu.mean, profile.imu.std)
        s = distance_to_similarity(d)
        result.imu = ModalityScore(distance=round(d, 4), similarity=round(s, 4), available=True)
        weighted_sim += W_IMU * s
        total_weight += W_IMU

    # Normalize weights
    if total_weight > 0:
        result.fused_similarity = weighted_sim / total_weight
    else:
        result.fused_similarity = 0.0

    # ── Exponential smoothing ──────────────────────────────────────────────
    if previous_trust is not None:
        result.trust_score = TRUST_ALPHA * result.fused_similarity + (1 - TRUST_ALPHA) * previous_trust
    else:
        result.trust_score = result.fused_similarity

    # ── Consecutive failure penalty ────────────────────────────────────────
    # Tracks how many checks in a row had low similarity.
    # After FAIL_COUNT_BEFORE_PENALTY consecutive failures, applies a
    # multiplicative penalty that pushes trust below the convergence floor.
    if result.fused_similarity < FAIL_SIM_THRESHOLD:
        result.consecutive_failures = previous_consecutive_failures + 1
    else:
        result.consecutive_failures = 0  # Reset on any passing check

    if result.consecutive_failures > FAIL_COUNT_BEFORE_PENALTY:
        penalties = result.consecutive_failures - FAIL_COUNT_BEFORE_PENALTY
        result.trust_score *= FAIL_PENALTY_MULTIPLIER ** penalties

    # ── Classify trust level ───────────────────────────────────────────────
    if result.trust_score >= TRUST_AUTHENTICATED:
        result.trust_level = "authenticated"
    elif result.trust_score >= TRUST_SOFT_CHALLENGE:
        result.trust_level = "soft_challenge"
    elif result.trust_score >= TRUST_HARD_CHALLENGE:
        result.trust_level = "hard_challenge"
    else:
        result.trust_level = "session_terminate"

    return result


# ── Profile Drift Update ──────────────────────────────────────────────────────

def update_profile_drift(
    profile: GaussianProfile,
    keystrokes: list[dict],
    scrolls: list[dict],
    imu_samples: list[dict],
    current_trust: float,
) -> GaussianProfile:
    """
    Slowly update the user's profile with new data — ONLY when trust is high.

    μ_user ← β × μ_user + (1-β) × F_live  (only when trust > DRIFT_TRUST_MIN)

    NEVER update when trust is low — that's how an attacker poisons the model.
    """
    if current_trust < DRIFT_TRUST_MIN:
        return profile  # Don't update — not confident enough

    beta = DRIFT_BETA

    # Update keystroke profile
    if profile.keystroke:
        live_ks = extract_keystroke_features(keystrokes)
        dim = min(len(live_ks), len(profile.keystroke.mean))
        if dim > 0:
            profile.keystroke.mean[:dim] = beta * profile.keystroke.mean[:dim] + (1 - beta) * live_ks[:dim]

    # Update scroll profile
    if profile.scroll and scrolls and len(scrolls) >= 2:
        live_sc = extract_scroll_features(scrolls)
        profile.scroll.mean = beta * profile.scroll.mean + (1 - beta) * live_sc

    # Update IMU profile
    if profile.imu and imu_samples:
        live_imu = extract_imu_features(imu_samples)
        profile.imu.mean = beta * profile.imu.mean + (1 - beta) * live_imu

    return profile


# ── Timestamp Jitter Check (Liveness) ──────────────────────────────────────────

def check_timestamp_jitter(timestamps: list[float], min_jitter_ms: float = 2.0) -> bool:
    """
    Liveness check: real human input has irregular timing intervals.
    Bot-generated events have too-regular (near-zero jitter) intervals.

    Returns True if the timestamps look human, False if suspiciously regular.
    """
    if len(timestamps) < 3:
        return True  # Not enough data to judge

    intervals = []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt > 0:
            intervals.append(dt)

    if not intervals:
        return True

    jitter = float(np.std(intervals))
    return jitter >= min_jitter_ms
