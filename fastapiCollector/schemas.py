# schemas.py — Pydantic v2 request / response models

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Sensor Event Models ────────────────────────────────────────────────────────
# Fields marked Optional with defaults are enrollment-specific.
# The Dashboard's continuous-verification flow only sends timing data.

class KeystrokeEvent(BaseModel):
    key: str = ""
    hold_time_ms: float = 0.0
    flight_time_ms: float = 0.0       # 0 for the very first key in a phrase
    phrase_id: Optional[str] = None   # Set during enrollment; absent in verify
    repetition_index: Optional[int] = None
    pressure: float = 0.5             # touch force, fallback 0.5
    pixel_density: float = 0.0        # PixelRatio * 160


class ScrollEvent(BaseModel):
    direction_deg: float = 0.0        # 0 = up, 180 = down
    distance_px: float = 0.0
    velocity_px_per_sec: float = 0.0
    avg_pressure: float = 0.5         # touch force during scroll, fallback 0.5
    pixel_density_dpi: float = 0.0
    timestamp: Optional[int] = None   # Date.now() ms


class IMUSample(BaseModel):
    gyro_x: float = 0.0               # rad/s
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    tilt_pitch: float = 0.0           # degrees
    tilt_roll: float = 0.0            # degrees
    timestamp: Optional[int] = None   # Date.now() ms


# ── API Request Bodies ─────────────────────────────────────────────────────────

class CreateUserRequest(BaseModel):
    name: str
    password: Optional[str] = None    # Password for unlock/login
    user_id: Optional[str] = None     # If omitted the backend generates a UUID


class LoginRequest(BaseModel):
    name: str
    password: str


class EnrollmentPayload(BaseModel):
    user_id: Optional[str] = None     # Omit → backend generates
    user_name: str
    keystrokes: List[KeystrokeEvent]
    scrolls: List[ScrollEvent]
    imu: List[IMUSample]


class VerifyPayload(BaseModel):
    user_id: str
    session_token: Optional[str] = None
    keystrokes: List[KeystrokeEvent]
    scrolls: List[ScrollEvent]
    imu: List[IMUSample]


# ── API Response Bodies ────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    user_id: str
    name: str
    source: str
    has_template: bool = False


class EnrollResponse(BaseModel):
    status: str
    user_id: str
    session_id: str
    stats: dict


class VerifyResponse(BaseModel):
    risk_score: float
    similarity: float
    alert_level: str          # "normal" | "warning" | "critical"
    locked: bool
    is_authenticated: bool
    # Gaussian engine fields (optional — present when Gaussian profile exists)
    trust_score: Optional[float] = None
    trust_level: Optional[str] = None       # "authenticated" | "soft_challenge" | "hard_challenge" | "session_terminate"
    engine: Optional[str] = None            # "gaussian" | "lstm" | "stats"
    modality_scores: Optional[dict] = None  # Per-modality breakdown


class HealthResponse(BaseModel):
    status: str
    users_count: int
    templates_count: int