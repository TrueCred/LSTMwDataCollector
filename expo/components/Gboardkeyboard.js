// ─────────────────────────────────────────────────────────────────────────────
// components/GboardKeyboard.js
// Custom keyboard styled to match Gboard dark mode.
// Reports keystroke biometric events via onKeystroke callback.
// ─────────────────────────────────────────────────────────────────────────────

import React, { useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  PixelRatio,
  Vibration,
} from 'react-native';
import * as Haptics from 'expo-haptics';

// ── Layout definitions ────────────────────────────────────────────────────────

const ALPHA_ROWS = [
  ['q','w','e','r','t','y','u','i','o','p'],
  ['a','s','d','f','g','h','j','k','l'],
  ['⌫','z','x','c','v','b','n','m','⌫_R'], // ⌫_R is a spacer slot on right
  ['123','🎤','SPACE','.','⏎'],
];

// Numeric / symbol layout
const NUM_ROWS = [
  ['1','2','3','4','5','6','7','8','9','0'],
  ['@','#','$','_','&','-','+','(',')','/'],
  ['⌫','*','"',"'",':', ';','!','?','⌫_R'],
  ['ABC','🎤','SPACE','.','⏎'],
];

const KEY_HEIGHT = 48;
const KEY_GAP    = 6;

// ── Helper: get key flex / width hint ─────────────────────────────────────────
function keyFlex(key) {
  if (key === 'SPACE')  return 5;
  if (key === '⌫_R')   return 0; // invisible spacer
  return 1;
}

// ── Single key component ──────────────────────────────────────────────────────
const Key = React.memo(({ label, onPressIn, onPressOut, flex, special }) => {
  const [pressed, setPressed] = useState(false);

  if (label === '⌫_R') {
    return <View style={{ flex: 0.5 }} />;
  }

  const handlePressIn = useCallback((e) => {
    setPressed(true);
    onPressIn(label, e.nativeEvent.force ?? 0.5);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  }, [label, onPressIn]);

  const handlePressOut = useCallback((e) => {
    setPressed(false);
    onPressOut(label, e.nativeEvent.force ?? 0.5);
  }, [label, onPressOut]);

  const isSpace  = label === 'SPACE';
  const isEnter  = label === '⏎';
  const isAction = label === '123' || label === 'ABC' || label === '🎤';
  const isDelete = label === '⌫';

  return (
    <TouchableOpacity
      activeOpacity={1}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      style={[
        styles.key,
        { flex: flex ?? keyFlex(label) },
        isSpace  && styles.spaceKey,
        isEnter  && styles.enterKey,
        isAction && styles.actionKey,
        isDelete && styles.deleteKey,
        pressed  && styles.keyPressed,
      ]}
    >
      <Text style={[
        styles.keyText,
        isEnter  && styles.enterText,
        isAction && styles.actionText,
        isSpace  && styles.spaceText,
      ]}>
        {isSpace ? '' : label}
      </Text>
    </TouchableOpacity>
  );
});

// ── Main GboardKeyboard ───────────────────────────────────────────────────────

/**
 * Props:
 *   onKeystroke(event)  — called with { key, hold_time_ms, flight_time_ms, pressure, pixel_density }
 *   onBackspace()       — called when ⌫ pressed
 *   onEnter()           — called when ⏎ pressed
 */
export default function GboardKeyboard({ onKeystroke, onBackspace, onEnter }) {
  const [isNumeric, setIsNumeric]         = useState(false);
  const pressTimestamps                   = useRef({}); // key → pressIn timestamp
  const lastReleaseTimestamp              = useRef(null);
  const pixelDensity                      = PixelRatio.get() * 160;

  const rows = isNumeric ? NUM_ROWS : ALPHA_ROWS;

  const handlePressIn = useCallback((key, force) => {
    pressTimestamps.current[key] = Date.now();
  }, []);

  const handlePressOut = useCallback((key, force) => {
    const now        = Date.now();
    const pressTime  = pressTimestamps.current[key] ?? now;
    const hold_ms    = now - pressTime;
    const flight_ms  = lastReleaseTimestamp.current != null
      ? pressTime - lastReleaseTimestamp.current
      : 0;
    lastReleaseTimestamp.current = now;

    // Handle special keys
    if (key === '⌫') {
      onBackspace?.();
      return;
    }
    if (key === '⏎') {
      onEnter?.();
      return;
    }
    if (key === '123') {
      setIsNumeric(true);
      return;
    }
    if (key === 'ABC') {
      setIsNumeric(false);
      return;
    }
    if (key === '🎤') return; // no-op

    const actualKey = key === 'SPACE' ? ' ' : key;

    onKeystroke?.({
      key:            actualKey,
      hold_time_ms:   hold_ms,
      flight_time_ms: flight_ms,
      pressure:       force,
      pixel_density:  pixelDensity,
    });
  }, [onKeystroke, onBackspace, onEnter, pixelDensity]);

  return (
    <View style={styles.keyboard}>
      {rows.map((row, rowIdx) => (
        <View key={rowIdx} style={styles.row}>
          {/* Row 2 (index 1) has a small indent for aesthetics */}
          {rowIdx === 1 && <View style={{ flex: 0.5 }} />}
          {row.map((key) => (
            <Key
              key={key}
              label={key}
              onPressIn={handlePressIn}
              onPressOut={handlePressOut}
            />
          ))}
          {rowIdx === 1 && <View style={{ flex: 0.5 }} />}
        </View>
      ))}
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  keyboard: {
    backgroundColor: '#1A1A1A',
    paddingHorizontal: 4,
    paddingVertical: 8,
    gap: KEY_GAP,
  },
  row: {
    flexDirection: 'row',
    gap: KEY_GAP,
    justifyContent: 'center',
    alignItems: 'center',
  },
  key: {
    height: KEY_HEIGHT,
    backgroundColor: '#2A2A2A',
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
    flex: 1,
    // Subtle shadow for depth
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.6,
    shadowRadius: 2,
    elevation: 3,
  },
  keyPressed: {
    backgroundColor: '#3D3D3D',
    transform: [{ scale: 0.95 }],
  },
  spaceKey: {
    flex: 5,
  },
  enterKey: {
    backgroundColor: '#1A73E8',
    flex: 1.2,
  },
  actionKey: {
    backgroundColor: '#3C3C3C',
    flex: 1.2,
  },
  deleteKey: {
    backgroundColor: '#3C3C3C',
    flex: 1.2,
  },
  keyText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '400',
  },
  enterText: {
    color: '#FFFFFF',
    fontSize: 18,
  },
  actionText: {
    color: '#CCCCCC',
    fontSize: 13,
    fontWeight: '500',
  },
  spaceText: {
    color: '#888',
    fontSize: 12,
  },
});