# main.py — Sentinel Biometrics FastAPI server
# Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
# Architecture:
#   Primary:  Gaussian Profile engine (Mahalanobis distance + trust score)
#   Fallback: LSTM ONNX inference → cosine-similarity stats engine

from __future__ import annotations
import json
import uuid
import sys
import importlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

import models
import schemas
from database import Base, engine, get_db
from config import (
    APP_TITLE, APP_VERSION,
    RISK_NORMAL_MAX, RISK_WARNING_MAX, RISK_LOCK_THRESHOLD, RISK_AUTH_THRESHOLD,
    USE_GAUSSIAN_ENGINE, TRUST_LOCK_THRESHOLD,
)
from biometric_engine import (
    build_gaussian_profile,
    GaussianProfile,
    compute_fused_score,
    update_profile_drift,
    check_timestamp_jitter,
    TRUST_AUTHENTICATED, TRUST_SOFT_CHALLENGE, TRUST_HARD_CHALLENGE,
)

# Keep runtime state in-memory for temporal risk smoothing.
sessions_db: dict[str, dict] = {}
inference_engine = None


def _pipeline_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    ml_root = root / "lstm" / "ml_pipeline"
    ckpt = ml_root / "checkpoints"
    return ml_root, ckpt / "sentinel_lstm.onnx", ckpt / "scalers.pkl"


def load_behavioral_model() -> None:
    global inference_engine

    ml_root, model_path, scalers_path = _pipeline_paths()
    vocab_path = ml_root / "checkpoints" / "key_vocab.json"

    if not ml_root.exists():
        inference_engine = None
        print("[LSTM] Pipeline folder not found. Using statistical fallback.")
        return

    if str(ml_root) not in sys.path:
        sys.path.append(str(ml_root))

    try:
        BehavioralInference = importlib.import_module("inference").BehavioralInference

        if model_path.exists() and scalers_path.exists():
            inference_engine = BehavioralInference(
                model_path=model_path,
                scalers_path=scalers_path,
                key_vocab_path=vocab_path,
            )
            print("[LSTM] ONNX inference loaded.")
        else:
            inference_engine = None
            print("[LSTM] ONNX artifacts missing. Using statistical fallback.")
    except Exception as exc:
        inference_engine = None
        print(f"[LSTM] Failed to load inference engine: {exc}")

# ── Bootstrap ──────────────────────────────────────────────────────────────────

# Auto-create all tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# Allow all origins so the React Native / Expo app can reach the server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_event_handler("startup", load_behavioral_model)


# ── Feature Extraction Helper (Legacy) ─────────────────────────────────────────

def _safe_mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _safe_std(vals: list[float]) -> float:
    return float(np.std(vals)) if vals else 0.0


def compute_stats(
    keystrokes: list[schemas.KeystrokeEvent],
    scrolls: list[schemas.ScrollEvent],
    imu: list[schemas.IMUSample],
) -> dict:
    """Compute 13-feature stats vector from raw sensor events."""
    holds = [k.hold_time_ms for k in keystrokes]
    flights = [k.flight_time_ms for k in keystrokes]
    kpressures = [k.pressure for k in keystrokes]

    velocities = [s.velocity_px_per_sec for s in scrolls]
    spressures = [s.avg_pressure for s in scrolls]

    gx = [i.gyro_x for i in imu]
    gy = [i.gyro_y for i in imu]
    gz = [i.gyro_z for i in imu]
    pitches = [i.tilt_pitch for i in imu]
    rolls = [i.tilt_roll for i in imu]

    return {
        "mean_hold":          _safe_mean(holds),
        "std_hold":           _safe_std(holds),
        "mean_flight":        _safe_mean(flights),
        "std_flight":         _safe_std(flights),
        "mean_pressure":      _safe_mean(kpressures),
        "mean_velocity":      _safe_mean(velocities),
        "std_velocity":       _safe_std(velocities),
        "mean_scroll_pressure": _safe_mean(spressures),
        "mean_gyro_x":        _safe_mean(gx),
        "mean_gyro_y":        _safe_mean(gy),
        "mean_gyro_z":        _safe_mean(gz),
        "mean_tilt_pitch":    _safe_mean(pitches),
        "mean_tilt_roll":     _safe_mean(rolls),
    }


def stats_to_vector(stats: dict) -> list[float]:
    """Return feature values in deterministic order."""
    keys = [
        "mean_hold", "std_hold", "mean_flight", "std_flight", "mean_pressure",
        "mean_velocity", "std_velocity", "mean_scroll_pressure",
        "mean_gyro_x", "mean_gyro_y", "mean_gyro_z",
        "mean_tilt_pitch", "mean_tilt_roll",
    ]
    return [stats[k] for k in keys]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors. Returns 0 if zero-norm."""
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _decode_template_payload(stats_vector_text: str) -> tuple[Optional[list[float]], Optional[list[float]]]:
    """Returns (stats_vector, fused_dna) from legacy or upgraded template payload."""
    try:
        parsed = json.loads(stats_vector_text)
    except Exception:
        return None, None

    if isinstance(parsed, list):
        return parsed, None

    if isinstance(parsed, dict):
        stats = parsed.get("stats_vector")
        dna = parsed.get("fused_dna")
        return (stats if isinstance(stats, list) else None, dna if isinstance(dna, list) else None)

    return None, None


# ── Helper: Convert Pydantic models to plain dicts for biometric engine ────────

def _events_to_dicts(events) -> list[dict]:
    """Convert list of pydantic models to list of plain dicts."""
    return [e.model_dump() for e in events]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=schemas.HealthResponse, tags=["meta"])
def health(db: Session = Depends(get_db)):
    """Liveness check — also returns user and template counts."""
    users_count = db.query(models.User).count()
    templates_count = db.query(models.Template).count()
    return {"status": "ok", "users_count": users_count, "templates_count": templates_count}


@app.post("/users/create", response_model=schemas.UserResponse, tags=["users"])
def create_user(body: schemas.CreateUserRequest, db: Session = Depends(get_db)):
    """
    Create a new user.  If user_id is provided, use it; otherwise generate a UUID.
    Idempotent: calling again with the same user_id just returns the existing record.
    """
    uid = body.user_id or str(uuid.uuid4())

    existing = db.query(models.User).filter(models.User.id == uid).first()
    if existing:
        has_template = db.query(models.Template).filter(
            models.Template.user_id == uid
        ).first() is not None
        return schemas.UserResponse(
            user_id=existing.id,
            name=existing.name,
            source=existing.source,
            has_template=has_template,
        )

    user = models.User(id=uid, name=body.name, password_hash=body.password, source="booth")
    db.add(user)
    db.commit()
    db.refresh(user)

    return schemas.UserResponse(user_id=user.id, name=user.name, source=user.source)


@app.post("/users/login", tags=["users"])
def login_user(body: schemas.LoginRequest, db: Session = Depends(get_db)):
    """
    Login / unlock: verify name + password.
    Used by LoginScreen and LockScreen.
    """
    user = db.query(models.User).filter(models.User.name == body.name).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    if user.password_hash != body.password:
        raise HTTPException(status_code=401, detail="Wrong password")

    has_template = db.query(models.Template).filter(
        models.Template.user_id == user.id
    ).first() is not None

    return {
        "user_id": user.id,
        "name": user.name,
        "is_enrolled": has_template,
    }


@app.post("/enroll", response_model=schemas.EnrollResponse, tags=["biometrics"])
def enroll(payload: schemas.EnrollmentPayload, db: Session = Depends(get_db)):
    """
    Full enrollment endpoint.
    1. Create/update user record.
    2. Persist raw sensor data as JSON.
    3. Compute 13-feature stats vector and upsert Template (legacy).
    4. Build Gaussian biometric profile and store alongside.
    """
    uid = payload.user_id or str(uuid.uuid4())

    # 1. Upsert user
    user = db.query(models.User).filter(models.User.id == uid).first()
    if user is None:
        user = models.User(id=uid, name=payload.user_name, source="booth")
        db.add(user)
        db.flush()
    else:
        user.name = payload.user_name  # allow name update

    # 2. Save raw data
    session_id = str(uuid.uuid4())
    raw = models.RawEnrollment(
        user_id=uid,
        session_id=session_id,
        keystrokes_json=json.dumps([k.model_dump() for k in payload.keystrokes]),
        scrolls_json=json.dumps([s.model_dump() for s in payload.scrolls]),
        imu_json=json.dumps([i.model_dump() for i in payload.imu]),
    )
    db.add(raw)

    # 3. Compute legacy features
    stats = compute_stats(payload.keystrokes, payload.scrolls, payload.imu)
    vector = stats_to_vector(stats)

    fused_dna = None
    if inference_engine is not None:
        try:
            fused_dna = inference_engine.extract_template(
                [k.model_dump() for k in payload.keystrokes],
                [s.model_dump() for s in payload.scrolls],
                [i.model_dump() for i in payload.imu],
            )
            fused_dna = fused_dna.reshape(-1).tolist()
        except Exception as exc:
            print(f"[LSTM] enroll embedding failed: {exc}")
            fused_dna = None

    template_payload = {
        "stats_vector": vector,
        "fused_dna": fused_dna,
    }

    # 4. Build Gaussian profile from ALL enrollment sessions for this user
    all_raw = db.query(models.RawEnrollment).filter(
        models.RawEnrollment.user_id == uid
    ).all()

    enrollment_sessions = []
    for r in all_raw:
        session_data = {
            "keystrokes": json.loads(r.keystrokes_json or "[]"),
            "scrolls": json.loads(r.scrolls_json or "[]"),
            "imu": json.loads(r.imu_json or "[]"),
        }
        enrollment_sessions.append(session_data)

    # Also include current (just flushed, not committed yet) data
    if not any(r.session_id == session_id for r in all_raw):
        enrollment_sessions.append({
            "keystrokes": [k.model_dump() for k in payload.keystrokes],
            "scrolls": [s.model_dump() for s in payload.scrolls],
            "imu": [i.model_dump() for i in payload.imu],
        })

    gaussian_profile = build_gaussian_profile(enrollment_sessions)
    gaussian_json = gaussian_profile.to_json()
    print(f"[Gaussian] Built profile from {len(enrollment_sessions)} sessions "
          f"(ks={gaussian_profile.keystroke is not None}, "
          f"sc={gaussian_profile.scroll is not None}, "
          f"imu={gaussian_profile.imu is not None})")

    # Upsert template (one per user — replace if re-enrolling)
    tmpl = db.query(models.Template).filter(models.Template.user_id == uid).first()
    if tmpl is None:
        tmpl = models.Template(
            user_id=uid,
            stats_vector=json.dumps(template_payload),
            gaussian_profile=gaussian_json,
        )
        db.add(tmpl)
    else:
        tmpl.stats_vector = json.dumps(template_payload)
        tmpl.gaussian_profile = gaussian_json

    db.commit()

    return schemas.EnrollResponse(
        status="enrolled",
        user_id=uid,
        session_id=session_id,
        stats=stats,
    )


@app.post("/verify", response_model=schemas.VerifyResponse, tags=["biometrics"])
def verify(payload: schemas.VerifyPayload, db: Session = Depends(get_db)):
    """
    Continuous-auth verification.

    Engine priority:
    1. Gaussian Profile (Mahalanobis distance + trust score) — if profile exists
    2. LSTM ONNX inference (cosine similarity) — if ONNX model loaded
    3. Statistical fallback (cosine similarity on 13-feature vector)
    """
    # Load template
    tmpl = db.query(models.Template).filter(
        models.Template.user_id == payload.user_id
    ).first()

    if tmpl is None:
        # No template → treat as fully untrusted
        return schemas.VerifyResponse(
            risk_score=1.0,
            similarity=0.0,
            alert_level="critical",
            locked=True,
            is_authenticated=False,
            trust_score=0.0,
            trust_level="session_terminate",
            engine="none",
        )

    # ── ENGINE 1: Gaussian Profile ──────────────────────────────────────────
    if USE_GAUSSIAN_ENGINE and tmpl.gaussian_profile:
        try:
            profile = GaussianProfile.from_json(tmpl.gaussian_profile)

            # Retrieve previous trust + failure count for smoothing
            token = payload.session_token or payload.user_id
            previous = sessions_db.get(token)
            prev_trust = float(previous["trust"]) if previous and "trust" in previous else None
            prev_failures = int(previous.get("consecutive_failures", 0)) if previous else 0

            # Convert pydantic models to dicts
            ks_dicts = _events_to_dicts(payload.keystrokes)
            sc_dicts = _events_to_dicts(payload.scrolls)
            imu_dicts = _events_to_dicts(payload.imu)

            # Liveness check on keystroke timestamps
            ks_timestamps = [float(k.get("timestamp", 0)) for k in ks_dicts if k.get("timestamp")]
            if ks_timestamps and not check_timestamp_jitter(ks_timestamps):
                print(f"[Gaussian] Liveness check failed for user {payload.user_id}")

            # Compute fused score
            fused = compute_fused_score(
                profile=profile,
                keystrokes=ks_dicts,
                scrolls=sc_dicts,
                imu_samples=imu_dicts,
                previous_trust=prev_trust,
                previous_consecutive_failures=prev_failures,
            )

            # Profile drift update (only when trust is high)
            if fused.trust_score >= 0.80:
                updated_profile = update_profile_drift(
                    profile, ks_dicts, sc_dicts, imu_dicts, fused.trust_score
                )
                tmpl.gaussian_profile = updated_profile.to_json()
                db.commit()

            # Store trust + failure count in session
            sessions_db[token] = {
                "trust": fused.trust_score,
                "risk": 1.0 - fused.trust_score,
                "consecutive_failures": fused.consecutive_failures,
                "last_seen": datetime.utcnow().isoformat(),
            }

            # Map trust score to legacy risk fields for backward compatibility
            risk_score = round(1.0 - fused.trust_score, 4)

            if fused.trust_level == "authenticated":
                alert_level = "normal"
            elif fused.trust_level == "soft_challenge":
                alert_level = "warning"
            else:
                alert_level = "critical"

            return schemas.VerifyResponse(
                risk_score=risk_score,
                similarity=round(fused.fused_similarity, 4),
                alert_level=alert_level,
                locked=fused.trust_level == "session_terminate",
                is_authenticated=fused.trust_level == "authenticated",
                trust_score=round(fused.trust_score, 4),
                trust_level=fused.trust_level,
                engine="gaussian",
                modality_scores=fused.to_dict(),
            )

        except Exception as exc:
            print(f"[Gaussian] Verification failed, falling back: {exc}")
            import traceback
            traceback.print_exc()

    # ── ENGINE 2: LSTM ONNX Inference ───────────────────────────────────────
    stats_vector, template_dna = _decode_template_payload(tmpl.stats_vector or "{}")

    if inference_engine is not None and template_dna:
        try:
            live_dna = inference_engine.extract_dna(
                [k.model_dump() for k in payload.keystrokes],
                [s.model_dump() for s in payload.scrolls],
                [i.model_dump() for i in payload.imu],
            )
            template_dna_arr = np.array(template_dna, dtype=np.float32).reshape(1, -1)
            risk_score = float(inference_engine.compute_risk(live_dna, template_dna_arr))

            token = payload.session_token or payload.user_id
            previous = sessions_db.get(token)
            if previous and "risk" in previous:
                risk_score = 0.7 * float(previous["risk"]) + 0.3 * risk_score

            sessions_db[token] = {
                "risk": risk_score,
                "last_seen": datetime.utcnow().isoformat(),
            }

            similarity = round(1.0 - risk_score, 4)

            if risk_score < RISK_NORMAL_MAX:
                alert_level = "normal"
            elif risk_score < RISK_WARNING_MAX:
                alert_level = "warning"
            else:
                alert_level = "critical"

            return schemas.VerifyResponse(
                risk_score=round(risk_score, 4),
                similarity=similarity,
                alert_level=alert_level,
                locked=risk_score > RISK_LOCK_THRESHOLD,
                is_authenticated=risk_score < RISK_AUTH_THRESHOLD,
                engine="lstm",
            )
        except Exception as exc:
            print(f"[LSTM] verify inference failed, fallback to stats: {exc}")

    # ── ENGINE 3: Statistical Fallback ──────────────────────────────────────
    live_stats = compute_stats(payload.keystrokes, payload.scrolls, payload.imu)
    live_vector = stats_to_vector(live_stats)
    template_vector = stats_vector if stats_vector else []

    if not template_vector:
        return schemas.VerifyResponse(
            risk_score=1.0,
            similarity=0.0,
            alert_level="critical",
            locked=True,
            is_authenticated=False,
            engine="stats",
        )

    similarity = cosine_similarity(live_vector, template_vector)
    risk_score = round(1.0 - similarity, 4)

    if risk_score < RISK_NORMAL_MAX:
        alert_level = "normal"
    elif risk_score < RISK_WARNING_MAX:
        alert_level = "warning"
    else:
        alert_level = "critical"

    return schemas.VerifyResponse(
        risk_score=risk_score,
        similarity=round(similarity, 4),
        alert_level=alert_level,
        locked=risk_score > RISK_LOCK_THRESHOLD,
        is_authenticated=risk_score < RISK_AUTH_THRESHOLD,
        engine="stats",
    )


@app.get("/user/{user_id}", response_model=schemas.UserResponse, tags=["users"])
def get_user(user_id: str, db: Session = Depends(get_db)):
    """Return user info + whether they have an enrolled template."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    has_template = db.query(models.Template).filter(
        models.Template.user_id == user_id
    ).first() is not None

    return schemas.UserResponse(
        user_id=user.id,
        name=user.name,
        source=user.source,
        has_template=has_template,
    )


@app.get("/user/{user_id}/data", tags=["users"])
def get_user_data(user_id: str, db: Session = Depends(get_db)):
    """Return all raw enrollment rows for a user (for debugging / export)."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    rows = db.query(models.RawEnrollment).filter(
        models.RawEnrollment.user_id == user_id
    ).all()

    return {
        "user_id": user_id,
        "name": user.name,
        "enrollments": [
            {
                "id": r.id,
                "session_id": r.session_id,
                "collected_at": r.collected_at,
                "keystroke_count": len(json.loads(r.keystrokes_json or "[]")),
                "scroll_count": len(json.loads(r.scrolls_json or "[]")),
                "imu_count": len(json.loads(r.imu_json or "[]")),
            }
            for r in rows
        ],
    }


@app.get("/user/{user_id}/profile", tags=["biometrics"])
def get_user_profile(user_id: str, db: Session = Depends(get_db)):
    """Return the user's Gaussian biometric profile (for debugging)."""
    tmpl = db.query(models.Template).filter(
        models.Template.user_id == user_id
    ).first()

    if tmpl is None:
        raise HTTPException(status_code=404, detail="No template found")

    result = {"user_id": user_id, "has_gaussian": False, "has_lstm": False}

    if tmpl.gaussian_profile:
        try:
            profile = GaussianProfile.from_json(tmpl.gaussian_profile)
            result["has_gaussian"] = True
            result["enrollment_sessions"] = profile.enrollment_sessions
            result["modalities"] = {
                "keystroke": profile.keystroke is not None,
                "scroll": profile.scroll is not None,
                "imu": profile.imu is not None,
            }
            if profile.keystroke:
                result["keystroke_dims"] = len(profile.keystroke.mean)
            if profile.scroll:
                result["scroll_dims"] = len(profile.scroll.mean)
            if profile.imu:
                result["imu_dims"] = len(profile.imu.mean)
        except Exception:
            pass

    if tmpl.stats_vector:
        _, dna = _decode_template_payload(tmpl.stats_vector)
        result["has_lstm"] = dna is not None

    return result