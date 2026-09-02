// config.js — TrueCred app configuration
export const API_BASE_URL =
  process.env.EXPO_PUBLIC_API_BASE_URL || 'http://172.18.240.244:8000';

export const APP_NAME = 'TrueCred';

export const PHRASES = [
  { id: 'balanced_random', text: 'vkerjpwu', reps: 5 },
  { id: 'repetition',      text: 'kkjjkkjj', reps: 5 },
  { id: 'numeric',         text: '13792846', reps: 5 },
];

export const MIN_SCROLLS    = 10;
export const IMU_DURATION_S = 30;
export const MAX_QUEUE_SIZE = 50;

// How often (ms) to send behavioral data for verification
export const VERIFY_INTERVAL_MS = 15_000;

// Risk thresholds for lockout (legacy)
export const RISK_LOCK_THRESHOLD = 0.55;

// Trust score thresholds (Gaussian engine)
export const TRUST_THRESHOLDS = {
  AUTHENTICATED:    0.65,  // green — genuine user
  SOFT_CHALLENGE:   0.40,  // yellow — re-verify soon
  HARD_CHALLENGE:   0.28,  // red — step-up auth required
  SESSION_TERMINATE: 0.28, // below this → lock
};

// Trust level labels
export const TRUST_LEVELS = {
  authenticated:     { label: 'VERIFIED',        color: '#00D68F' },
  soft_challenge:    { label: 'MONITORING',      color: '#FFAA00' },
  hard_challenge:    { label: 'SUSPICIOUS',      color: '#FF6B35' },
  session_terminate: { label: 'LOCKED',          color: '#FF3366' },
  unknown:           { label: 'INITIALIZING...', color: '#555' },
};

export const STORAGE_KEYS = {
  USER_ID:          'truecred_user_id',
  USER_NAME:        'truecred_user_name',
  IS_ENROLLED:      'truecred_is_enrolled',
  PENDING_UPLOADS:  'truecred_pending_uploads',
};