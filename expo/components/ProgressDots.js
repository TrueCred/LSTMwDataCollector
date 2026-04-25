// ─────────────────────────────────────────────────────────────────────────────
// components/ProgressDots.js
// Three-step progress indicator for the enrollment wizard.
// ─────────────────────────────────────────────────────────────────────────────

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

export default function ProgressDots({ current, total = 3, labels }) {
  return (
    <View style={styles.container}>
      {Array.from({ length: total }, (_, i) => {
        const active  = i === current;
        const done    = i < current;
        return (
          <View key={i} style={styles.step}>
            <View style={[
              styles.dot,
              done   && styles.dotDone,
              active && styles.dotActive,
            ]}>
              {done && <Text style={styles.check}>✓</Text>}
              {!done && <Text style={[styles.num, active && styles.numActive]}>{i + 1}</Text>}
            </View>
            {labels && (
              <Text style={[styles.label, active && styles.labelActive]}>
                {labels[i]}
              </Text>
            )}
            {i < total - 1 && (
              <View style={[styles.connector, done && styles.connectorDone]} />
            )}
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    gap: 0,
  },
  step: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  dot: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#2A2A2A',
    borderWidth: 2,
    borderColor: '#444',
    alignItems: 'center',
    justifyContent: 'center',
  },
  dotActive: {
    borderColor: '#1A73E8',
    backgroundColor: '#0D3A6E',
  },
  dotDone: {
    backgroundColor: '#1A73E8',
    borderColor: '#1A73E8',
  },
  num: {
    color: '#666',
    fontSize: 12,
    fontWeight: '700',
  },
  numActive: {
    color: '#4FA3FF',
  },
  check: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: '700',
  },
  label: {
    color: '#666',
    fontSize: 11,
    marginLeft: 4,
  },
  labelActive: {
    color: '#4FA3FF',
  },
  connector: {
    width: 32,
    height: 2,
    backgroundColor: '#333',
    marginHorizontal: 2,
  },
  connectorDone: {
    backgroundColor: '#1A73E8',
  },
});