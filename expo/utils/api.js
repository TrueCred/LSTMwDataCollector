// utils/api.js — TrueCred API client
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL, STORAGE_KEYS, MAX_QUEUE_SIZE } from '../config';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Auth ─────────────────────────────────────────────────────────────────────
export async function createUser(name, password) {
  const res = await api.post('/users/create', { name, password });
  return res.data; // { user_id, name }
}

export async function loginUser(name, password) {
  const res = await api.post('/users/login', { name, password });
  return res.data; // { user_id, name, is_enrolled }
}

// ── Health ───────────────────────────────────────────────────────────────────
export async function checkHealth() {
  const res = await api.get('/health');
  return res.data;
}

// ── Enroll ───────────────────────────────────────────────────────────────────
export async function enrollUser(payload) {
  const res = await api.post('/enroll', payload);
  return res.data;
}

// ── Verify ───────────────────────────────────────────────────────────────────
export async function verifyUser(payload) {
  const res = await api.post('/verify', payload);
  return res.data; // { status, risk, risk_level, user_id }
}

// ── Offline queue ────────────────────────────────────────────────────────────
export async function loadQueue() {
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEYS.PENDING_UPLOADS);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

export async function pushToQueue(payload) {
  try {
    const queue = await loadQueue();
    if (queue.length >= MAX_QUEUE_SIZE) queue.shift();
    queue.push({ payload, queued_at: Date.now() });
    await AsyncStorage.setItem(STORAGE_KEYS.PENDING_UPLOADS, JSON.stringify(queue));
  } catch (err) { console.warn('pushToQueue error:', err); }
}

export async function flushQueue() {
  const queue = await loadQueue();
  let uploaded = 0;
  for (const item of queue) {
    try {
      await enrollUser(item.payload);
      queue.shift();
      await AsyncStorage.setItem(STORAGE_KEYS.PENDING_UPLOADS, JSON.stringify(queue));
      uploaded++;
    } catch { break; }
  }
  return uploaded;
}