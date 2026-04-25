// ─────────────────────────────────────────────────────────────────────────────
// config.js
// Update API_BASE_URL to match the laptop running the FastAPI backend.
// Both devices must be on the same WiFi network.
// ─────────────────────────────────────────────────────────────────────────────

export const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

export const PHRASES = [
  { id: 'balanced_random', text: 'vkerjpwu', reps: 5 },
  { id: 'repetition',      text: 'kkjjkkjj', reps: 5 },
  { id: 'numeric',         text: '13792846', reps: 5 },
];

export const MIN_SCROLLS    = 10;
export const IMU_DURATION_S = 30;  // seconds for IMU baseline
export const MAX_QUEUE_SIZE = 50;  // max pending uploads in AsyncStorage

export const STORAGE_KEYS = {
  USER_ID:          'sentinel_user_id',
  USER_NAME:        'sentinel_user_name',
  PENDING_UPLOADS:  'sentinel_pending_uploads',
};