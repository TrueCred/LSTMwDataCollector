// screens/DashboardScreen.js — Main app with passive continuous monitoring
// Key-agnostic model: captures timing from ANY text input, not specific phrases
import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View, Text, TextInput, ScrollView,
  StyleSheet, TouchableOpacity, BackHandler,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { verifyUser } from '../utils/api';
import { startIMUCollection } from '../utils/sensors';
import { STORAGE_KEYS, APP_NAME, VERIFY_INTERVAL_MS, RISK_LOCK_THRESHOLD } from '../config';

const RISK_COLORS = {
  normal:   '#00D68F',
  caution:  '#FFAA00',
  warning:  '#FF6B35',
  critical: '#FF3366',
  unknown:  '#555',
};

export default function DashboardScreen({ navigation }) {
  const [userName,    setUserName]    = useState('');
  const [noteText,    setNoteText]    = useState('');
  const [riskLevel,   setRiskLevel]   = useState('unknown');
  const [riskScore,   setRiskScore]   = useState(null);
  const [rawRisk,     setRawRisk]     = useState(null);
  const [verifyCount, setVerifyCount] = useState(0);
  const [ksCount,     setKsCount]     = useState(0);

  // Behavioral data buffers
  const keystrokeBuf = useRef([]);
  const scrollBuf    = useRef([]);
  const imuBuf       = useRef([]);
  const imuCollector = useRef(null);
  const verifyTimer  = useRef(null);

  // Keystroke timing refs
  const lastKeyPressTime = useRef(null);
  const lastTextChangeTime = useRef(null);

  // ── Load user info & start monitoring ──────────────────────────────────────
  useEffect(() => {
    (async () => {
      const uname = await AsyncStorage.getItem(STORAGE_KEYS.USER_NAME);
      setUserName(uname || 'User');
    })();

    // Start IMU collection
    imuCollector.current = startIMUCollection((sample) => {
      imuBuf.current.push(sample);
      if (imuBuf.current.length > 500) imuBuf.current = imuBuf.current.slice(-300);
    });

    // Periodic verification
    verifyTimer.current = setInterval(() => { runVerification(); }, VERIFY_INTERVAL_MS);

    const backHandler = BackHandler.addEventListener('hardwareBackPress', () => true);

    return () => {
      if (imuCollector.current) imuCollector.current.stop();
      if (verifyTimer.current) clearInterval(verifyTimer.current);
      backHandler.remove();
    };
  }, []);

  // ── Run verification ───────────────────────────────────────────────────────
  const runVerification = useCallback(async () => {
    const uid = await AsyncStorage.getItem(STORAGE_KEYS.USER_ID);
    if (!uid) return;

    const ks  = keystrokeBuf.current.splice(0);
    const sc  = scrollBuf.current.splice(0);
    const imu = imuBuf.current.splice(0, 300);

    // Need enough keystroke data for meaningful verification
    if (ks.length < 5) return;

    try {
      const result = await verifyUser({
        user_id: uid,
        keystrokes: ks,
        scrolls: sc,
        imu: imu,
      });

      setRiskLevel(result.risk_level);
      setRiskScore(result.risk);
      setRawRisk(result.raw_risk);
      setVerifyCount((c) => c + 1);

      if (result.risk >= RISK_LOCK_THRESHOLD) {
        if (imuCollector.current) imuCollector.current.stop();
        if (verifyTimer.current) clearInterval(verifyTimer.current);
        navigation.replace('Lock', {
          risk: result.risk,
          risk_level: result.risk_level,
        });
      }
    } catch (err) {
      console.warn('Verify failed:', err.message);
    }
  }, [navigation]);

  // ── Keystroke timing capture ───────────────────────────────────────────────
  // Key-agnostic: we only capture TIMING (hold_time, flight_time), not which key
  function handleKeyPress(e) {
    const now = Date.now();

    // Flight time = gap since last key was released (textChange)
    const flightTime = lastTextChangeTime.current
      ? now - lastTextChangeTime.current
      : 0;

    lastKeyPressTime.current = now;

    // Store with a placeholder hold_time; will be patched in handleTextChange
    keystrokeBuf.current.push({
      key: '',  // not used by key-agnostic model
      hold_time_ms: 0,
      flight_time_ms: Math.max(0, flightTime),
      _pressTime: now,
    });

    setKsCount(keystrokeBuf.current.length);
  }

  function handleTextChange(text) {
    const now = Date.now();
    lastTextChangeTime.current = now;

    // Patch most recent keystroke's hold_time (keyPress → textChange ≈ hold duration)
    const buf = keystrokeBuf.current;
    if (buf.length > 0) {
      const last = buf[buf.length - 1];
      if (last._pressTime) {
        last.hold_time_ms = Math.max(1, now - last._pressTime);
        delete last._pressTime;
      }
    }

    setNoteText(text);
  }

  // ── Scroll capture ────────────────────────────────────────────────────────
  function handleScroll(e) {
    const { contentOffset } = e.nativeEvent;
    scrollBuf.current.push({
      direction_deg: contentOffset.y > 0 ? 90 : 270,
      distance_px: Math.abs(contentOffset.y),
    });
    if (scrollBuf.current.length > 100) scrollBuf.current = scrollBuf.current.slice(-50);
  }

  // ── Logout ─────────────────────────────────────────────────────────────────
  async function handleLogout() {
    await AsyncStorage.multiRemove([
      STORAGE_KEYS.USER_ID,
      STORAGE_KEYS.USER_NAME,
      STORAGE_KEYS.IS_ENROLLED,
    ]);
    navigation.replace('Welcome');
  }

  const riskColor = RISK_COLORS[riskLevel] || RISK_COLORS.unknown;

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.appName}>{APP_NAME}</Text>
          <Text style={styles.greeting}>Hello, {userName}</Text>
        </View>
        <TouchableOpacity onPress={handleLogout} style={styles.logoutBtn}>
          <Text style={styles.logoutText}>Logout</Text>
        </TouchableOpacity>
      </View>

      {/* Risk indicator */}
      <View style={[styles.riskCard, { borderColor: riskColor }]}>
        <View style={styles.riskRow}>
          <View style={[styles.riskDot, { backgroundColor: riskColor }]} />
          <Text style={[styles.riskLabel, { color: riskColor }]}>
            {riskLevel === 'unknown' ? 'Monitoring...' : riskLevel.toUpperCase()}
          </Text>
        </View>
        <Text style={styles.riskDetail}>
          {riskScore !== null
            ? `Risk: ${(riskScore * 100).toFixed(1)}%  (raw: ${((rawRisk ?? 0) * 100).toFixed(1)}%)`
            : 'Type below to start behavioral monitoring...'}
        </Text>
        <Text style={styles.riskSub}>
          Checks: {verifyCount} | Keystrokes buffered: {ksCount}
        </Text>
      </View>

      {/* Manual verify */}
      <TouchableOpacity style={styles.verifyBtn} onPress={runVerification} activeOpacity={0.8}>
        <Text style={styles.verifyBtnText}>Verify Now</Text>
      </TouchableOpacity>

      {/* Typing area — captures keystroke timing passively */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Notes</Text>
        <TextInput
          style={styles.noteInput}
          placeholder="Type anything here... your typing rhythm is being monitored"
          placeholderTextColor="#444"
          value={noteText}
          onChangeText={handleTextChange}
          onKeyPress={handleKeyPress}
          multiline
          textAlignVertical="top"
        />
      </View>

      {/* Scrollable content */}
      <ScrollView
        style={styles.feed}
        onScroll={handleScroll}
        scrollEventThrottle={200}
      >
        <Text style={styles.feedTitle}>How TrueCred Works</Text>
        {[
          'Your typing rhythm is unique — the time you hold each key and the gaps between keys form your behavioral fingerprint.',
          'TrueCred uses an LSTM neural network to encode your timing patterns into a 32-dimensional behavioral DNA vector.',
          'The model is key-agnostic — it only analyzes HOW you type (timing), not WHAT you type. This means it works with any text.',
          'Every 15 seconds, your buffered typing data is compared against your enrolled template. If the patterns don\'t match, the app locks.',
          'If someone else picks up your phone, their typing speed, rhythm, and device handling will be different — triggering a lockout.',
          'Scroll through this content and type in the notes area above. The more you interact, the more accurate the system becomes.',
        ].map((text, i) => (
          <View key={i} style={styles.feedCard}>
            <Text style={styles.feedText}>{text}</Text>
          </View>
        ))}
        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0A0A0F', paddingTop: 50 },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    marginBottom: 12,
  },
  appName:  { color: '#6C5CE7', fontSize: 13, fontWeight: '700', letterSpacing: 1.5, textTransform: 'uppercase' },
  greeting: { color: '#FFF', fontSize: 22, fontWeight: '700', marginTop: 2 },
  logoutBtn: { backgroundColor: '#1A1A25', borderRadius: 8, paddingHorizontal: 14, paddingVertical: 8 },
  logoutText: { color: '#888', fontSize: 13, fontWeight: '600' },

  riskCard: {
    marginHorizontal: 20,
    backgroundColor: '#111118',
    borderRadius: 14,
    borderWidth: 1.5,
    padding: 16,
    marginBottom: 10,
  },
  riskRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  riskDot: { width: 12, height: 12, borderRadius: 6 },
  riskLabel: { fontSize: 16, fontWeight: '800', letterSpacing: 1 },
  riskDetail: { color: '#AAA', fontSize: 14, marginTop: 6 },
  riskSub: { color: '#555', fontSize: 12, marginTop: 4 },

  verifyBtn: {
    marginHorizontal: 20,
    backgroundColor: '#1E1E2E',
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#2A2A3A',
  },
  verifyBtnText: { color: '#6C5CE7', fontSize: 14, fontWeight: '700' },

  section: { paddingHorizontal: 20, marginBottom: 10 },
  sectionTitle: { color: '#888', fontSize: 12, fontWeight: '700', marginBottom: 6, letterSpacing: 1, textTransform: 'uppercase' },
  noteInput: {
    backgroundColor: '#111118',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#1E1E2A',
    color: '#FFF',
    fontSize: 15,
    padding: 14,
    minHeight: 80,
    maxHeight: 120,
  },

  feed: { flex: 1, paddingHorizontal: 20 },
  feedTitle: { color: '#888', fontSize: 12, fontWeight: '700', marginBottom: 10, letterSpacing: 1, textTransform: 'uppercase' },
  feedCard: {
    backgroundColor: '#111118',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#1E1E2A',
    padding: 14,
    marginBottom: 10,
  },
  feedText: { color: '#CCC', fontSize: 14, lineHeight: 20 },
});
