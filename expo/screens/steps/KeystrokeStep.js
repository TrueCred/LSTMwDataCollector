// ─────────────────────────────────────────────────────────────────────────────
// screens/steps/KeystrokeStep.js
// Step A of enrollment: captures typing biometrics via custom Gboard keyboard.
// 3 phrases × 5 valid reps each → 15 completed reps minimum.
// ─────────────────────────────────────────────────────────────────────────────

import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  View, Text, Animated, StyleSheet, Platform, BackHandler,
} from 'react-native';
import GboardKeyboard from '../../components/Gboardkeyboard';
import { PHRASES } from '../../config';
import * as Haptics from 'expo-haptics';

export default function KeystrokeStep({ onComplete }) {
  const [phraseIdx,  setPhraseIdx]  = useState(0);
  const [repIdx,     setRepIdx]     = useState(0);
  const [typed,      setTyped]      = useState('');
  const [events,     setEvents]     = useState([]);  // all collected keystroke events
  const shakeAnim   = useRef(new Animated.Value(0)).current;
  const flashAnim   = useRef(new Animated.Value(0)).current;
  const prevKey     = useRef(null); // last non-error key for flight time tracking

  const currentPhrase = PHRASES[phraseIdx];
  const target        = currentPhrase?.text ?? '';

  // ── Block Android back during collection ────────────────────────────────
  useEffect(() => {
    const sub = BackHandler.addEventListener('hardwareBackPress', () => true);
    return () => sub.remove();
  }, []);

  // ── Shake + red flash animation for wrong key ────────────────────────────
  function triggerError() {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    Animated.sequence([
      Animated.timing(shakeAnim, { toValue: 10,  duration: 50,  useNativeDriver: true }),
      Animated.timing(shakeAnim, { toValue: -10, duration: 50,  useNativeDriver: true }),
      Animated.timing(shakeAnim, { toValue: 10,  duration: 50,  useNativeDriver: true }),
      Animated.timing(shakeAnim, { toValue: 0,   duration: 50,  useNativeDriver: true }),
    ]).start();
    Animated.sequence([
      Animated.timing(flashAnim, { toValue: 1, duration: 80,  useNativeDriver: false }),
      Animated.timing(flashAnim, { toValue: 0, duration: 300, useNativeDriver: false }),
    ]).start();
  }

  // ── Handle each key from the keyboard ────────────────────────────────────
  const handleKeystroke = useCallback((event) => {
    if (!currentPhrase) return;

    const nextChar = target[typed.length];

    if (event.key === ' ') return; // ignore space in phrases
    if (event.key !== nextChar) {
      // Wrong key — don't count, show error
      triggerError();
      // Reset this rep's typed buffer so they start again
      setTyped('');
      prevKey.current = null;
      return;
    }

    // Correct key
    const enriched = {
      ...event,
      phrase_id:        currentPhrase.id,
      repetition_index: repIdx,
    };

    const newTyped = typed + event.key;
    setTyped(newTyped);

    setEvents(prev => [...prev, enriched]);

    // Completed a rep?
    if (newTyped === target) {
      setTyped('');
      prevKey.current = null;
      const nextRep = repIdx + 1;

      if (nextRep >= currentPhrase.reps) {
        // Advance to next phrase
        const nextPhrase = phraseIdx + 1;
        if (nextPhrase >= PHRASES.length) {
          // All done!
          onComplete(events.concat(enriched));
        } else {
          setPhraseIdx(nextPhrase);
          setRepIdx(0);
        }
      } else {
        setRepIdx(nextRep);
      }
    }
  }, [currentPhrase, target, typed, repIdx, phraseIdx, events, onComplete]);

  const handleBackspace = useCallback(() => {
    setTyped(prev => prev.slice(0, -1));
  }, []);

  if (!currentPhrase) return null;

  const totalReps    = PHRASES.reduce((s, p) => s + p.reps, 0);
  const completedReps = PHRASES.slice(0, phraseIdx).reduce((s, p) => s + p.reps, 0) + repIdx;
  const progress     = completedReps / totalReps;

  const flashColor = flashAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['rgba(255,0,0,0)', 'rgba(255,0,0,0.25)'],
  });

  return (
    <View style={styles.container}>
      {/* Progress bar */}
      <View style={styles.progressBar}>
        <View style={[styles.progressFill, { width: `${progress * 100}%` }]} />
      </View>

      <View style={styles.header}>
        <Text style={styles.stepLabel}>
          Phrase {phraseIdx + 1}/{PHRASES.length} — Rep {repIdx + 1}/{currentPhrase.reps}
        </Text>
        <Text style={styles.totalLabel}>
          {completedReps}/{totalReps} reps complete
        </Text>
      </View>

      <Text style={styles.instruction}>Type the phrase exactly as shown:</Text>

      {/* Target phrase display */}
      <Animated.View style={[
        styles.phraseContainer,
        { transform: [{ translateX: shakeAnim }] },
      ]}>
        {/* Flash overlay */}
        <Animated.View style={[StyleSheet.absoluteFill, { backgroundColor: flashColor, borderRadius: 12 }]} />
        <Text style={styles.phraseText}>{target}</Text>
      </Animated.View>

      {/* Typed input display */}
      <View style={styles.typedContainer}>
        <Text style={styles.typedText}>
          {typed || <Text style={styles.placeholder}>Start typing…</Text>}
        </Text>
        {typed.length > 0 && (
          <Text style={styles.cursor}>|</Text>
        )}
      </View>

      {/* Hint about wrong keys */}
      <Text style={styles.hint}>Wrong key? The rep resets automatically.</Text>

      {/* Custom keyboard takes up the bottom */}
      <View style={styles.keyboardWrapper}>
        <GboardKeyboard
          onKeystroke={handleKeystroke}
          onBackspace={handleBackspace}
          onEnter={() => {}}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0D0D0D' },
  progressBar: {
    height: 3,
    backgroundColor: '#1C1C1C',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#1A73E8',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 14,
    paddingBottom: 4,
  },
  stepLabel:   { color: '#4FA3FF', fontSize: 14, fontWeight: '600' },
  totalLabel:  { color: '#666',    fontSize: 13 },
  instruction: { color: '#888', fontSize: 14, paddingHorizontal: 20, marginBottom: 10 },
  phraseContainer: {
    marginHorizontal: 20,
    backgroundColor: '#161616',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#2A2A2A',
    paddingVertical: 18,
    paddingHorizontal: 20,
    alignItems: 'center',
    overflow: 'hidden',
  },
  phraseText: {
    color: '#FFFFFF',
    fontSize: 32,
    fontFamily: Platform.select({ ios: 'Courier New', android: 'monospace' }),
    letterSpacing: 6,
    fontWeight: '700',
  },
  typedContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 12,
    marginBottom: 6,
    minHeight: 36,
  },
  typedText: {
    color: '#4FA3FF',
    fontSize: 26,
    fontFamily: Platform.select({ ios: 'Courier New', android: 'monospace' }),
    letterSpacing: 6,
  },
  placeholder: {
    color: '#333',
    fontSize: 16,
  },
  cursor: {
    color: '#4FA3FF',
    fontSize: 26,
    marginLeft: 2,
  },
  hint: {
    color: '#444',
    fontSize: 12,
    textAlign: 'center',
    marginBottom: 8,
  },
  keyboardWrapper: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
  },
});