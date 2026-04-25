# schemas.py — Pydantic v2 request / response models

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Sensor Event Models ────────────────────────────────────────────────────────

class KeystrokeEvent(BaseModel):
    phrase_id: str                    # "balanced_random" | "repetition" | "numeric"
    repetition_index: int
    key: str
    hold_time_ms: float
    flight_time_ms: float             # 0 for the very first key in a phrase
    pressure: float                   # touch force, fallback 0.5
    pixel_density: float              # PixelRatio * 160


class ScrollEvent(BaseModel):
    velocity_px_per_sec: float
    direction_deg: float              # 0 = up, 180 = down
    distance_px: float
    avg_pressure: float               # touch force during scroll, fallback 0.5
    pixel_density_dpi: float
    timestamp: int                    # Date.now() ms


class IMUSample(BaseModel):
    gyro_x: float                     # rad/s
    gyro_y: float
    gyro_z: float
    tilt_pitch: float                 # degrees
    tilt_roll: float                  # degrees
    timestamp: int                    # Date.now() ms


# ── API Request Bodies ─────────────────────────────────────────────────────────

class CreateUserRequest(BaseModel):
    name: str
    user_id: Optional[str] = None     # If omitted the backend generates a UUID


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


class HealthResponse(BaseModel):
    status: str
    users_count: int
    templates_count: int