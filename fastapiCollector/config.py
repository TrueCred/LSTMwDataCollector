# config.py — Sentinel Biometrics Backend Configuration

APP_TITLE = "Sentinel Biometrics API"
APP_VERSION = "1.0.0"
 
DB_URL = "sqlite:///./sentinel_lab.db"
 
# Risk thresholds
RISK_NORMAL_MAX = 0.35
RISK_WARNING_MAX = 0.65
RISK_LOCK_THRESHOLD = 0.75
RISK_AUTH_THRESHOLD = 0.40
 