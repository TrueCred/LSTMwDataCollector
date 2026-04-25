"""FastAPI integration for behavioral authentication."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import BehavioralInference


# Models
class EnrollRequest(BaseModel):
    user_id: str
    keystrokes: list
    scrolls: list
    imu: list


class EnrollResponse(BaseModel):
    status: str
    user_id: str
    enrolled_at: str


class VerifyRequest(BaseModel):
    user_id: str
    keystrokes: list
    scrolls: list
    imu: list


class VerifyResponse(BaseModel):
    status: str
    risk: float
    risk_level: str
    user_id: str


# FastAPI app
app = FastAPI()

# Initialize inference engine
root = Path(__file__).resolve().parent
inference = BehavioralInference(
    onnx_model_path=root / "sentinel_encoder.onnx",
    scalers_path=root / "scalers.pkl",
    key_vocab_path=root / "key_vocab.json"
)

# Database path (relative to ml_pipeline folder)
db_path = Path(__file__).parent.parent.parent / "backend" / "sentinel_lab.db"


def get_db_connection() -> sqlite3.Connection:
    """Get database connection."""
    return sqlite3.connect(str(db_path))


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(request: EnrollRequest) -> EnrollResponse:
    """
    Enroll a new user.
    
    Extracts DNA embedding and saves to templates table.
    """
    try:
        # Extract DNA
        dna = inference.extract_dna(request.keystrokes, request.scrolls, request.imu)
        
        if dna is None:
            raise HTTPException(status_code=400, detail="Failed to extract DNA")
        
        # Store in templates table
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Convert DNA to JSON strings (save individual components)
            key_dna = json.dumps(dna[0, :].tolist())
            scroll_dna = json.dumps(dna[0, :].tolist())  # Same embedding
            imu_stats = json.dumps(dna[0, :].tolist())  # Same embedding
            
            cursor.execute(
                """
                INSERT OR REPLACE INTO templates (user_id, key_dna, scroll_dna, imu_stats)
                VALUES (?, ?, ?, ?)
                """,
                (request.user_id, key_dna, scroll_dna, imu_stats)
            )
            conn.commit()
            
            # Get enrollment timestamp
            cursor.execute("SELECT enrolled_at FROM templates WHERE user_id = ?", (request.user_id,))
            row = cursor.fetchone()
            enrolled_at = row[0] if row else ""
            
            return EnrollResponse(
                status="success",
                user_id=request.user_id,
                enrolled_at=enrolled_at
            )
        finally:
            conn.close()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Temporal smoothing state (in-memory for demo)
temporal_state = {}


@app.post("/verify", response_model=VerifyResponse)
async def verify(request: VerifyRequest) -> VerifyResponse:
    """
    Verify user identity.
    
    Extracts DNA, loads template, computes risk with temporal smoothing.
    """
    try:
        import numpy as np
        
        # Extract live DNA
        live_dna = inference.extract_dna(request.keystrokes, request.scrolls, request.imu)
        
        if live_dna is None:
            raise HTTPException(status_code=400, detail="Failed to extract DNA")
        
        # Load template
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT key_dna FROM templates WHERE user_id = ?", (request.user_id,))
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="User not enrolled")
            
            template_dna_json = row[0]
            template_dna = np.array(json.loads(template_dna_json)).reshape(1, -1)
            
            # Compute risk
            raw_risk = inference.compute_risk(live_dna, template_dna)
            
            # Apply temporal smoothing
            prev_risk = temporal_state.get(request.user_id, raw_risk)
            smoothed_risk = 0.7 * prev_risk + 0.3 * raw_risk
            temporal_state[request.user_id] = smoothed_risk
            
            # Determine risk level
            if smoothed_risk < 0.35:
                risk_level = "normal"
            elif smoothed_risk < 0.60:
                risk_level = "caution"
            elif smoothed_risk < 0.80:
                risk_level = "warning"
            else:
                risk_level = "critical_locked"
            
            return VerifyResponse(
                status="success",
                risk=float(smoothed_risk),
                risk_level=risk_level,
                user_id=request.user_id
            )
        finally:
            conn.close()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "inference_available": inference.session is not None,
        "database": "available" if db_path.exists() else "not_found"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
