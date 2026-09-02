# TrueCred Backend

FastAPI server that stores user accounts, accepts behavioral enrollment data, and scores live sessions against enrolled profiles.

This is where Phase 3 (trust engine) lives. See `../truecred_architecture.png` or `../truecred_system_architecture.svg` for the full flow.

## Engines

Verification runs through two layers:

1. **Gaussian profile engine** (primary, enabled by default). Extracts statistical features from keystroke, scroll, and IMU events. Builds per-user mean/variance profiles at enrollment. At verify time, computes Mahalanobis distance per modality, fuses scores with fixed weights, and maintains a smoothed trust score. See `biometric_engine.py`.

2. **LSTM ONNX fallback.** If `lstm/ml_pipeline/checkpoints/sentinel_lstm.onnx` and `scalers.pkl` exist, the server can also run neural inference. When the Gaussian profile is missing, it falls back to cosine-similarity scoring from the LSTM DNA vectors.

Toggle the Gaussian engine in `config.py`:

```python
USE_GAUSSIAN_ENGINE = True
```

## Scoring pipeline

At a high level, each `/verify` call does this:

1. Extract feature vectors from the incoming keystroke, scroll, and IMU batches.
2. Compute Mahalanobis distance against the stored Gaussian profile (`z = (x - mu) / sigma`).
3. Convert distance to similarity (`S = exp(-D / scale)`).
4. Fuse modalities: keystroke 50%, scroll 25%, IMU 25%.
5. Apply exponential smoothing: `T = alpha * S + (1 - alpha) * T_old`.
6. Apply consecutive failure penalty if similarity stays low.
7. Return verdict: authenticated, warn, or lock.

Modality diagnostics (which signal caused the mismatch) are included in the response for the UI.

## Database

SQLite via SQLAlchemy. Default path: `./sentinel_lab.db`.

Main tables: users, raw enrollment events, Gaussian profiles, and optional LSTM template vectors. Migration scripts (`migrate_add_password.py`, `migrate_add_gaussian.py`) add columns when upgrading older databases.

## Running

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: `http://localhost:8000/docs`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/users/create` | Create user with name + password |
| POST | `/users/login` | Login, returns enrollment status |
| POST | `/enroll` | Store behavioral enrollment + build profile |
| POST | `/verify` | Score live behavioral data, return trust/risk |
| GET | `/user/{user_id}` | User metadata |
| GET | `/user/{user_id}/data` | Raw stored enrollment events |
| GET | `/user/{user_id}/profile` | Gaussian profile summary |

Request and response models live in `schemas.py`.

## Trust score behavior

The Gaussian engine applies exponential smoothing so one noisy sample does not instantly lock the user. Consecutive low-similarity checks apply a penalty multiplier to break out of a plateau. Profile drift updates (slow adaptation to natural behavior change) only happen when trust is already high.

Frontend thresholds (defined in `expo/config.js`):

- 0.65 and above: authenticated
- 0.40 to 0.64: soft challenge / monitoring
- 0.28 to 0.39: hard challenge
- Below 0.28: session terminate / lock

## Testing

```bash
python -m pytest test_api.py test_biometric_engine.py
```

## Utilities

- `wipe_db.py`: Clear database for fresh testing
- `migrate_add_password.py`: Add password column to existing DB
- `migrate_add_gaussian.py`: Add Gaussian profile storage
