// ─────────────────────────────────────────────────────────────────────────────
// utils/api.js
// Thin axios wrapper + offline queue helpers.
// ─────────────────────────────────────────────────────────────────────────────

import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL, STORAGE_KEYS, MAX_QUEUE_SIZE } from '../config';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Health check ──────────────────────────────────────────────────────────────
export async function checkHealth() {
  const res = await api.get('/health');
  return res.data;
}

// ── Create user ───────────────────────────────────────────────────────────────
export async function createUser(name, user_id = null) {
  const body = user_id ? { name, user_id } : { name };
  const res  = await api.post('/users/create', body);
  return res.data; // { user_id, name }
}

// ── Enroll ────────────────────────────────────────────────────────────────────
export async function enrollUser(payload) {
  const res = await api.post('/enroll', payload);
  return res.data;
}

// ── Verify ────────────────────────────────────────────────────────────────────
export async function verifyUser(payload) {
  const res = await api.post('/verify', payload);
  return res.data;
}

// ─────────────────────────────────────────────────────────────────────────────
// Offline queue helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Load the pending-upload queue from AsyncStorage */
export async function loadQueue() {
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEYS.PENDING_UPLOADS);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

/** Push a payload onto the queue (evicts oldest if at max capacity) */
export async function pushToQueue(payload) {
  try {
    const queue = await loadQueue();
    if (queue.length >= MAX_QUEUE_SIZE) queue.shift();
    queue.push({ payload, queued_at: Date.now() });
    await AsyncStorage.setItem(STORAGE_KEYS.PENDING_UPLOADS, JSON.stringify(queue));
  } catch (err) {
    console.warn('pushToQueue error:', err);
  }
}

/** Remove the first item from the queue after a successful upload */
async function shiftQueue() {
  const queue = await loadQueue();
  queue.shift();
  await AsyncStorage.setItem(STORAGE_KEYS.PENDING_UPLOADS, JSON.stringify(queue));
}

/** Attempt to flush the entire queue, oldest-first. Returns number of successes. */
export async function flushQueue() {
  const queue = await loadQueue();
  let uploaded = 0;
  for (let i = 0; i < queue.length; i++) {
    try {
      await enrollUser(queue[i].payload);
      await shiftQueue();
      uploaded++;
    } catch {
      // Stop on first failure — retry next time
      break;
    }
  }
  return uploaded;
}