"""FastAPI backend for TrueCred behavioral authentication."""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import BehavioralInference

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="TrueCred API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Inference engine ──────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent
inference = BehavioralInference(
    onnx_model_path=root / "sentinel_encoder.onnx",
    scalers_path=root / "scalers.pkl",
    key_vocab_path=root / "key_vocab.json",
)

# ── Database ──────────────────────────────────────────────────────────────────
db_path = root / "truecred.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            is_enrolled INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            dna_vector TEXT NOT NULL,
            enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()


init_db()


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


# ── Request / Response models ─────────────────────────────────────────────────
class CreateUserReq(BaseModel):
    name: str
    password: str


class LoginReq(BaseModel):
    name: str
    password: str


class EnrollRequest(BaseModel):
    user_id: str
    keystrokes: list = []
    scrolls: list = []
    imu: list = []


class VerifyRequest(BaseModel):
    user_id: str
    keystrokes: list = []
    scrolls: list = []
    imu: list = []


# ── Temporal smoothing state (in-memory) ──────────────────────────────────────
temporal_state: dict[str, float] = {}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/users/create")
async def create_user(req: CreateUserReq):
    uid = str(uuid.uuid4())
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (id, name, password_hash) VALUES (?, ?, ?)",
            (uid, req.name, hash_password(req.password)),
        )
        conn.commit()
        return {"user_id": uid, "name": req.name}
    except sqlite3.IntegrityError:
        raise HTTPException(400, "User already exists")
    finally:
        conn.close()


@app.post("/users/login")
async def login(req: LoginReq):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id, name, is_enrolled FROM users WHERE name = ? AND password_hash = ?",
            (req.name, hash_password(req.password)),
        ).fetchone()
        if not row:
            raise HTTPException(401, "Invalid credentials")
        return {"user_id": row["id"], "name": row["name"], "is_enrolled": bool(row["is_enrolled"])}
    finally:
        conn.close()


@app.post("/enroll")
async def enroll(req: EnrollRequest):
    """Enroll user — windows keystrokes and averages DNA across all windows."""
    keystrokes = req.keystrokes or []
    scrolls = req.scrolls or []
    imu = req.imu or []

    # Window keystrokes into 8-key chunks (matching model input size)
    windows = []
    for start in range(0, max(1, len(keystrokes) - 7), 4):
        chunk = keystrokes[start:start + 8]
        if len(chunk) >= 4:  # need at least 4 keystrokes
            windows.append(chunk)

    if not windows:
        windows = [keystrokes]  # fallback: use whatever we have

    # Extract DNA for each window and average
    dna_vectors = []
    for window in windows:
        dna = inference.extract_dna(window, scrolls, imu)
        if dna is not None:
            dna_vectors.append(dna[0, :])

    if not dna_vectors:
        raise HTTPException(400, "Failed to extract behavioral DNA")

    # Average and L2-normalize the template
    avg_dna = np.mean(dna_vectors, axis=0)
    avg_dna = avg_dna / (np.linalg.norm(avg_dna) + 1e-8)

    print(f"Enrollment: {len(dna_vectors)} windows averaged for user {req.user_id[:8]}")

    dna_json = json.dumps(avg_dna.tolist())
    conn = get_db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO templates (user_id, dna_vector) VALUES (?, ?)",
            (req.user_id, dna_json),
        )
        conn.execute("UPDATE users SET is_enrolled = 1 WHERE id = ?", (req.user_id,))
        conn.commit()

        row = conn.execute(
            "SELECT enrolled_at FROM templates WHERE user_id = ?", (req.user_id,)
        ).fetchone()
        return {
            "status": "enrolled",
            "user_id": req.user_id,
            "enrolled_at": row["enrolled_at"] if row else "",
            "windows_used": len(dna_vectors),
        }
    finally:
        conn.close()


@app.post("/verify")
async def verify(req: VerifyRequest):
    live_dna = inference.extract_dna(req.keystrokes, req.scrolls, req.imu)
    if live_dna is None:
        raise HTTPException(400, "Failed to extract behavioral DNA")

    conn = get_db()
    try:
        row = conn.execute(
            "SELECT dna_vector FROM templates WHERE user_id = ?", (req.user_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "User not enrolled")

        template_dna = np.array(json.loads(row["dna_vector"])).reshape(1, -1)
        raw_risk = inference.compute_risk(live_dna, template_dna)

        # Temporal smoothing (responsive: weights current reading more heavily)
        prev = temporal_state.get(req.user_id, raw_risk)
        smoothed = 0.3 * prev + 0.7 * raw_risk
        temporal_state[req.user_id] = smoothed

        if smoothed < 0.35:
            level = "normal"
        elif smoothed < 0.60:
            level = "caution"
        elif smoothed < 0.80:
            level = "warning"
        else:
            level = "critical"

        return {
            "status": "verified",
            "risk": round(float(smoothed), 4),
            "raw_risk": round(float(raw_risk), 4),
            "risk_level": level,
            "user_id": req.user_id,
        }
    finally:
        conn.close()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": inference.session is not None,
        "database": "ok" if db_path.exists() else "missing",
    }


if __name__ == "__main__":
    import uvicorn
    print("TrueCred API starting on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
