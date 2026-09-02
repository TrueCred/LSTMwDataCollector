# TrueCred

Behavioral biometrics for continuous session protection. TrueCred watches how you interact with your phone—touch patterns, scrolling, device posture—and compares that against a profile built during enrollment. If someone else takes over an unlocked session, the trust score drops and the app locks before sensitive actions go through.

Built by Team Brothers: Utkrist Mani Neupane, Saksham Gyawali, Sammit Poudyal, Nishant Pandit.

## The problem

Most apps stop caring about identity once you log in. That leaves a gap:

- Passwords and OTPs do not help if the device is snatched while unlocked.
- Remote access trojans and device takeover attacks can operate inside an already-authenticated session.
- Financial apps lose public trust quickly after unauthorized transfers.

TrueCred addresses the session itself, not just the login screen.

## How it works

1. **Calibration** — During enrollment, the user scrolls, types fixed phrases, and holds the device normally. The system records keystroke timing, scroll dynamics, and IMU readings to build a baseline.
2. **Silent monitoring** — While the app is in use, touch coordinates, velocity, flight time, and gyroscope data are collected in the background.
3. **Trust scoring** — Live behavior is compared against the enrolled profile. The backend computes a trust score using a Gaussian profile engine (Mahalanobis distance across keystroke, scroll, and IMU features). An LSTM model is available as a secondary path when ONNX artifacts are present.
4. **Intervention** — When behavior diverges enough, the trust score falls. The app moves through monitoring states and eventually locks the screen, requiring password re-entry.

Critical actions like sending money or reviewing a transaction are where strict checks matter most—the same idea shown in the project presentation under `final/`.

## Project structure

```
TrueCred/
├── expo/                 React Native mobile app (Expo)
├── fastapiCollector/     Backend API and biometric scoring engine
├── lstm/ml_pipeline/     LSTM model training and ONNX export
├── final/                Presentation site (index.html) and demo videos
└── verify_db.py          Quick SQLite sanity check script
```

| Component | Role |
|-----------|------|
| `expo/` | Collects sensor and touch data, runs enrollment flow, polls `/verify`, locks on low trust |
| `fastapiCollector/` | User accounts, enrollment storage, Gaussian + LSTM verification, SQLite persistence |
| `lstm/ml_pipeline/` | Train the behavioral LSTM, export ONNX weights, optional standalone inference API |
| `final/` | Slide deck and demo recordings for presentations |

## Getting started

You need Python 3.10+, Node.js, and the Expo CLI. Run the backend first, then point the mobile app at it.

### 1. Backend

```bash
cd fastapiCollector
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API listens on port 8000. SQLite database file `sentinel_lab.db` is created in `fastapiCollector/`.

### 2. Mobile app

```bash
cd expo
npm install
```

Set the backend URL in `expo/config.js` or via environment variable:

```bash
export EXPO_PUBLIC_API_BASE_URL=http://YOUR_MACHINE_IP:8000
npx expo start
```

Use Expo Go on a physical device for IMU and touch sensors. Update `API_BASE_URL` to your machine's LAN address—not `localhost`—when testing on a phone.

### 3. LSTM pipeline (optional)

The backend works out of the box with the Gaussian engine. To train and deploy the LSTM fallback:

```bash
cd lstm/ml_pipeline
pip install -r requirements.txt
python reset_db.py
python data_loader.py
python train.py
python export_onnx.py
```

Export writes `sentinel_encoder.onnx` in `lstm/ml_pipeline/`. Copy it to `checkpoints/sentinel_lstm.onnx`, then restart the backend.

See [lstm/ml_pipeline/README.md](lstm/ml_pipeline/README.md) for details.

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service status |
| POST | `/users/create` | Register account |
| POST | `/users/login` | Authenticate |
| POST | `/enroll` | Submit behavioral enrollment data |
| POST | `/verify` | Compare live behavior against profile |
| GET | `/user/{user_id}/profile` | Inspect stored biometric profile |

Full request/response shapes are in `fastapiCollector/schemas.py`.

## Trust levels

The Gaussian engine returns a continuous trust score. The mobile app maps it to states:

| Score | State | App behavior |
|-------|-------|--------------|
| ≥ 0.65 | Verified | Normal use |
| 0.40 – 0.64 | Monitoring | Elevated watch |
| 0.28 – 0.39 | Suspicious | Step-up auth likely |
| < 0.28 | Locked | Screen lock, password required |

Thresholds are configurable in `fastapiCollector/config.py` and `expo/config.js`.

## Presentation

Open `final/index.html` in a browser for the full slide deck, demo video player, and references. It walks through the problem, solution, pipeline, and live demo scenarios (normal user vs simulated attack).

## References

- [Touchalytics](https://arxiv.org/pdf/1207.6231) — touch dynamics on smartphones
- [ProKYC](https://www.catonetworks.com/blog/prokyc-selling-deepfake-tool-for-account-fraud-attacks/) — deepfake KYC fraud tooling
- [BingoMod](https://www.cleafy.com/cleafy-labs/bingomod-the-new-android-rat-that-steals-money-and-wipes-data) — Android RAT targeting financial apps
- [FBI IC3 Annual Report 2024](https://www.ic3.gov/AnnualReport/Reports/2024_IC3Report.pdf)
