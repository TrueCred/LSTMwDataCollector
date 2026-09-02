// screens/DashboardScreen.js — Main app with continuous behavioral monitoring
// Supports both Gaussian engine (trust score) and legacy (risk score)
import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View, Text, TextInput, ScrollView,
  StyleSheet, TouchableOpacity, BackHandler, Animated,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { verifyUser } from '../utils/api';
import { startIMUCollection } from '../utils/sensors';
import {
  STORAGE_KEYS, APP_NAME, VERIFY_INTERVAL_MS,
  RISK_LOCK_THRESHOLD, TRUST_LEVELS, TRUST_THRESHOLDS,
} from '../config';
import GboardKeyboard from '../components/Gboardkeyboard';

export default function DashboardScreen({ navigation }) {
  const [userName,    setUserName]    = useState('');
  const [noteText,    setNoteText]    = useState('');
  const [trustLevel,  setTrustLevel]  = useState('unknown');
  const [trustScore,  setTrustScore]  = useState(null);
  const [riskScore,   setRiskScore]   = useState(null);
  const [engine,      setEngine]      = useState(null);
  const [verifyCount, setVerifyCount] = useState(0);
  const [ksCount,     setKsCount]     = useState(0);
  const [modalScores, setModalScores] = useState(null);

  const [showKeyboard, setShowKeyboard] = useState(false);

  // Animations
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const trustBarAnim = useRef(new Animated.Value(0)).current;

  // Behavioral data buffers
  const keystrokeBuf = useRef([]);
  const scrollBuf    = useRef([]);
  const imuBuf       = useRef([]);
  const imuCollector = useRef(null);
  const verifyTimer  = useRef(null);

  // Keystroke timing refs
  const lastKeyPressTime = useRef(null);
  const lastTextChangeTime = useRef(null);

  // ── Pulse animation for trust indicator ────────────────────────────────
  useEffect(() => {
    const pulse = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.15, duration: 1200, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 1200, useNativeDriver: true }),
      ])
    );
    pulse.start();
    return () => pulse.stop();
  }, []);

  // Animate trust bar
  useEffect(() => {
    if (trustScore !== null) {
      Animated.spring(trustBarAnim, {
        toValue: trustScore,
        useNativeDriver: false,
        friction: 8,
      }).start();
    }
  }, [trustScore]);

  // ── Load user info & start monitoring ──────────────────────────────────
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

    return () => {
      if (imuCollector.current) imuCollector.current.stop();
      if (verifyTimer.current) clearInterval(verifyTimer.current);
    };
  }, []);

  // ── Handle Back Press to hide keyboard ─────────────────────────────────
  useEffect(() => {
    const backHandler = BackHandler.addEventListener('hardwareBackPress', () => {
      if (showKeyboard) {
        setShowKeyboard(false);
        return true; // prevent default behavior (closing app/navigating back)
      }
      return true; // still prevent going back from Dashboard
    });
    return () => backHandler.remove();
  }, [showKeyboard]);

  // ── Run verification ───────────────────────────────────────────────────
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

      // Handle Gaussian engine response
      if (result.trust_score !== undefined && result.trust_score !== null) {
        setTrustScore(result.trust_score);
        setTrustLevel(result.trust_level || 'unknown');
        setEngine(result.engine || 'gaussian');
        setModalScores(result.modality_scores || null);
      } else {
        // Legacy response — convert risk to trust
        const legacyTrust = 1.0 - (result.risk ?? 0);
        setTrustScore(legacyTrust);
        setEngine(result.engine || 'stats');
        if (legacyTrust >= TRUST_THRESHOLDS.AUTHENTICATED) {
          setTrustLevel('authenticated');
        } else if (legacyTrust >= TRUST_THRESHOLDS.SOFT_CHALLENGE) {
          setTrustLevel('soft_challenge');
        } else if (legacyTrust >= TRUST_THRESHOLDS.HARD_CHALLENGE) {
          setTrustLevel('hard_challenge');
        } else {
          setTrustLevel('session_terminate');
        }
      }

      setRiskScore(result.risk_score ?? result.risk ?? null);
      setVerifyCount((c) => c + 1);

      // Lock check — prefer trust_level from Gaussian engine
      const shouldLock = result.trust_level
        ? result.trust_level === 'session_terminate'
        : (result.risk_score >= RISK_LOCK_THRESHOLD);

      if (shouldLock) {
        if (imuCollector.current) imuCollector.current.stop();
        if (verifyTimer.current) clearInterval(verifyTimer.current);
        navigation.replace('Lock', {
          risk: result.risk_score,
          risk_level: result.alert_level || result.trust_level,
          trust_score: result.trust_score,
          modality_scores: result.modality_scores || null,
        });
      }
    } catch (err) {
      console.warn('Verify failed:', err.message);
    }
  }, [navigation]);

  // ── Keystroke timing capture ───────────────────────────────────────────
  const handleKeystroke = useCallback((event) => {
    // Add to buffer
    keystrokeBuf.current.push({
      key: event.key,
      hold_time_ms: event.hold_time_ms,
      flight_time_ms: event.flight_time_ms,
      pressure: event.pressure,
      timestamp: Date.now(),
    });
    setKsCount(keystrokeBuf.current.length);

    // Update note text
    setNoteText((prev) => prev + event.key);
  }, []);

  const handleBackspace = useCallback(() => {
    setNoteText((prev) => prev.slice(0, -1));
  }, []);

  const handleEnter = useCallback(() => {
    setNoteText((prev) => prev + '\n');
  }, []);

  // ── Scroll capture ────────────────────────────────────────────────────
  const lastScrollY = useRef(0);
  const lastScrollT = useRef(Date.now());

  function handleScroll(e) {
    const { contentOffset } = e.nativeEvent;
    const now = Date.now();
    const dy = contentOffset.y - lastScrollY.current;
    const dt = Math.max(1, now - lastScrollT.current);

    // Only record meaningful scroll events (match enrollment ScrollStep logic)
    if (Math.abs(dy) >= 10 && dt >= 20) {
      const velocity = (Math.abs(dy) / dt) * 1000; // px/sec
      scrollBuf.current.push({
        velocity_px_per_sec: velocity,
        direction_deg: dy >= 0 ? 180 : 0,
        distance_px: Math.abs(dy),
        avg_pressure: 0.5,
        timestamp: now,
      });
      if (scrollBuf.current.length > 100) scrollBuf.current = scrollBuf.current.slice(-50);
    }

    lastScrollY.current = contentOffset.y;
    lastScrollT.current = now;
  }

  // ── Logout ─────────────────────────────────────────────────────────────
  async function handleLogout() {
    await AsyncStorage.multiRemove([
      STORAGE_KEYS.USER_ID,
      STORAGE_KEYS.USER_NAME,
      STORAGE_KEYS.IS_ENROLLED,
    ]);
    navigation.replace('Welcome');
  }

  const levelInfo = TRUST_LEVELS[trustLevel] || TRUST_LEVELS.unknown;
  const trustColor = levelInfo.color;
  const trustPercent = trustScore !== null ? (trustScore * 100).toFixed(1) : '—';

  // Trust bar width interpolation
  const trustBarWidth = trustBarAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

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

      {/* Trust Score Card */}
      <View style={[styles.trustCard, { borderColor: trustColor }]}>
        {/* Trust Indicator */}
        <View style={styles.trustHeader}>
          <Animated.View style={[
            styles.trustDot,
            { backgroundColor: trustColor, transform: [{ scale: pulseAnim }] },
          ]} />
          <Text style={[styles.trustLabel, { color: trustColor }]}>
            {levelInfo.label}
          </Text>
          {engine && (
            <View style={styles.engineBadge}>
              <Text style={styles.engineText}>{engine.toUpperCase()}</Text>
            </View>
          )}
        </View>

        {/* Trust Score Display */}
        <View style={styles.trustScoreRow}>
          <Text style={[styles.trustScoreValue, { color: trustColor }]}>
            {trustPercent}%
          </Text>
          <Text style={styles.trustScoreLabel}>Trust Score</Text>
        </View>

        {/* Trust Bar */}
        <View style={styles.trustBarOuter}>
          <Animated.View style={[
            styles.trustBarInner,
            { width: trustBarWidth, backgroundColor: trustColor },
          ]} />
          {/* Threshold markers */}
          <View style={[styles.thresholdMarker, { left: '25%' }]}>
            <View style={styles.thresholdLine} />
          </View>
          <View style={[styles.thresholdMarker, { left: '50%' }]}>
            <View style={styles.thresholdLine} />
          </View>
          <View style={[styles.thresholdMarker, { left: '75%' }]}>
            <View style={styles.thresholdLine} />
          </View>
        </View>
        <View style={styles.thresholdLabels}>
          <Text style={styles.thresholdText}>LOCK</Text>
          <Text style={styles.thresholdText}>CHALLENGE</Text>
          <Text style={styles.thresholdText}>MONITOR</Text>
          <Text style={styles.thresholdText}>SAFE</Text>
        </View>

        {/* Modality breakdown */}
        {modalScores && (
          <View style={styles.modalityRow}>
            {modalScores.keystroke?.available && (
              <View style={styles.modalityChip}>
                <Text style={styles.modalityIcon}>⌨️</Text>
                <Text style={styles.modalityValue}>
                  {(modalScores.keystroke.similarity * 100).toFixed(0)}%
                </Text>
              </View>
            )}
            {modalScores.scroll?.available && (
              <View style={styles.modalityChip}>
                <Text style={styles.modalityIcon}>📜</Text>
                <Text style={styles.modalityValue}>
                  {(modalScores.scroll.similarity * 100).toFixed(0)}%
                </Text>
              </View>
            )}
            {modalScores.imu?.available && (
              <View style={styles.modalityChip}>
                <Text style={styles.modalityIcon}>📱</Text>
                <Text style={styles.modalityValue}>
                  {(modalScores.imu.similarity * 100).toFixed(0)}%
                </Text>
              </View>
            )}
          </View>
        )}

        <Text style={styles.trustSub}>
          Checks: {verifyCount} | Keystrokes buffered: {ksCount}
          {riskScore !== null ? ` | Risk: ${(riskScore * 100).toFixed(1)}%` : ''}
        </Text>
      </View>

      {/* Manual verify */}
      <TouchableOpacity style={styles.verifyBtn} onPress={runVerification} activeOpacity={0.8}>
        <Text style={styles.verifyBtnText}>Verify Now</Text>
      </TouchableOpacity>

      {/* Typing area — captures keystroke timing passively */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Notes</Text>
        <TouchableOpacity 
          activeOpacity={0.8}
          onPress={() => setShowKeyboard(true)}
          style={[styles.noteInput, { minHeight: 80, justifyContent: 'flex-start' }]}
        >
          <Text style={{ color: '#FFF', fontSize: 16 }}>
            {noteText || <Text style={{ color: '#444' }}>Tap to type... your typing rhythm is being monitored</Text>}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Scrollable content */}
      <ScrollView
        style={[styles.feed, { flex: 1 }]}
        onScroll={handleScroll}
        scrollEventThrottle={200}
      >
        <Text style={styles.feedTitle}>How TrueCred Works</Text>
        {[
          '🛡️ TrueCred uses Gaussian behavioral profiling to continuously verify your identity through how you type, scroll, and hold your device.',
          '📊 During enrollment, your behavioral patterns are captured across multiple modalities. Each builds a statistical profile (mean + variance).',
          '📏 Verification uses Mahalanobis distance — measuring how many standard deviations your live behavior is from your enrolled profile.',
          '🔄 Three modalities are fused: keystroke dynamics (50%), scroll patterns (25%), and IMU/motion data (25%) into a single trust score.',
          '📈 Your trust score is smoothed over time using exponential averaging — preventing sudden false alarms from momentary behavior changes.',
          '🧬 The system slowly adapts to natural changes in your behavior (profile drift) — but ONLY when trust is high, preventing attacker poisoning.',
        ].map((text, i) => (
          <View key={i} style={styles.feedCard}>
            <Text style={styles.feedText}>{text}</Text>
          </View>
        ))}
        <View style={{ height: 40 }} />
      </ScrollView>

      {/* Custom Keyboard for accurate keystroke timing */}
      {showKeyboard && (
        <View style={{ backgroundColor: '#1A1A1A' }}>
          <GboardKeyboard
            onKeystroke={handleKeystroke}
            onBackspace={handleBackspace}
            onEnter={handleEnter}
          />
        </View>
      )}
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

  // Trust score card
  trustCard: {
    marginHorizontal: 20,
    backgroundColor: '#111118',
    borderRadius: 14,
    borderWidth: 1.5,
    padding: 16,
    marginBottom: 10,
  },
  trustHeader: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  trustDot: { width: 14, height: 14, borderRadius: 7 },
  trustLabel: { fontSize: 16, fontWeight: '800', letterSpacing: 1, flex: 1 },
  engineBadge: {
    backgroundColor: '#1A1A2A',
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderWidth: 1,
    borderColor: '#2A2A3A',
  },
  engineText: { color: '#666', fontSize: 10, fontWeight: '700', letterSpacing: 0.5 },

  // Trust score
  trustScoreRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 8,
    marginTop: 10,
    marginBottom: 8,
  },
  trustScoreValue: { fontSize: 36, fontWeight: '800' },
  trustScoreLabel: { color: '#666', fontSize: 14, fontWeight: '600' },

  // Trust bar
  trustBarOuter: {
    height: 8,
    backgroundColor: '#1C1C28',
    borderRadius: 4,
    overflow: 'hidden',
    position: 'relative',
  },
  trustBarInner: {
    height: '100%',
    borderRadius: 4,
  },
  thresholdMarker: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 1,
  },
  thresholdLine: {
    width: 1,
    height: '100%',
    backgroundColor: '#333',
  },
  thresholdLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
    paddingHorizontal: 2,
  },
  thresholdText: { color: '#444', fontSize: 8, fontWeight: '600', letterSpacing: 0.5 },

  // Modality chips
  modalityRow: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 10,
  },
  modalityChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1A1A2A',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
    gap: 4,
    borderWidth: 1,
    borderColor: '#2A2A3A',
  },
  modalityIcon: { fontSize: 14 },
  modalityValue: { color: '#CCC', fontSize: 13, fontWeight: '700' },

  trustSub: { color: '#555', fontSize: 11, marginTop: 8 },

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
