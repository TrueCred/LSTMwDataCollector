// screens/LockScreen.js — Shown when behavioral mismatch detected
// Now includes modality diagnostics showing WHY the user was flagged
import React, { useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity,
  StyleSheet, Alert, ActivityIndicator, ScrollView,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { loginUser } from '../utils/api';
import { STORAGE_KEYS, APP_NAME, TRUST_LEVELS } from '../config';

export default function LockScreen({ route, navigation }) {
  const risk        = route?.params?.risk ?? 0;
  const riskLevel   = route?.params?.risk_level ?? 'critical';
  const trustScore  = route?.params?.trust_score ?? null;
  const modality    = route?.params?.modality_scores ?? null;
  const [password, setPassword] = useState('');
  const [loading,  setLoading]  = useState(false);

  async function handleUnlock() {
    if (!password) return;

    setLoading(true);
    try {
      const name = await AsyncStorage.getItem(STORAGE_KEYS.USER_NAME);
      await loginUser(name, password);
      // Password correct — go back to dashboard
      navigation.replace('Dashboard');
    } catch {
      Alert.alert('Wrong password', 'Please enter the correct password to unlock.');
    } finally {
      setLoading(false);
    }
  }

  async function handleLogout() {
    await AsyncStorage.multiRemove([
      STORAGE_KEYS.USER_ID,
      STORAGE_KEYS.USER_NAME,
      STORAGE_KEYS.IS_ENROLLED,
    ]);
    navigation.replace('Welcome');
  }

  // Helper to get color based on similarity
  function getSimColor(sim) {
    if (sim >= 0.75) return '#00D68F';
    if (sim >= 0.50) return '#FFAA00';
    if (sim >= 0.25) return '#FF6B35';
    return '#FF3366';
  }

  // Helper to get label for similarity
  function getSimLabel(sim) {
    if (sim >= 0.75) return 'MATCH';
    if (sim >= 0.50) return 'PARTIAL';
    if (sim >= 0.25) return 'WEAK';
    return 'MISMATCH';
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      <View style={styles.content}>
        <Text style={styles.lockIcon}>🔒</Text>
        <Text style={styles.title}>Account Locked</Text>
        <Text style={styles.subtitle}>
          {APP_NAME} detected unusual behavior
        </Text>

        <View style={styles.riskCard}>
          <Text style={styles.riskLabel}>Threat Level</Text>
          <Text style={styles.riskValue}>{(risk * 100).toFixed(1)}%</Text>
          {trustScore !== null && (
            <Text style={styles.trustValue}>Trust: {(trustScore * 100).toFixed(1)}%</Text>
          )}
          <Text style={styles.riskLevel}>{riskLevel.toUpperCase()}</Text>
        </View>

        {/* Modality Diagnostics */}
        {modality && (
          <View style={styles.diagnosticsCard}>
            <Text style={styles.diagTitle}>🔍 What triggered the lock?</Text>
            <Text style={styles.diagSubtitle}>
              Each modality's similarity to your enrolled profile:
            </Text>

            {/* Keystroke */}
            {modality.keystroke?.available && (
              <View style={styles.diagRow}>
                <View style={styles.diagLeft}>
                  <Text style={styles.diagIcon}>⌨️</Text>
                  <View>
                    <Text style={styles.diagName}>Typing Rhythm</Text>
                    <Text style={styles.diagDetail}>
                      Distance: {modality.keystroke.distance.toFixed(2)}σ
                    </Text>
                  </View>
                </View>
                <View style={styles.diagRight}>
                  <Text style={[
                    styles.diagScore,
                    { color: getSimColor(modality.keystroke.similarity) },
                  ]}>
                    {(modality.keystroke.similarity * 100).toFixed(0)}%
                  </Text>
                  <Text style={[
                    styles.diagLabel,
                    { color: getSimColor(modality.keystroke.similarity) },
                  ]}>
                    {getSimLabel(modality.keystroke.similarity)}
                  </Text>
                </View>
              </View>
            )}

            {/* Scroll */}
            {modality.scroll?.available && (
              <View style={styles.diagRow}>
                <View style={styles.diagLeft}>
                  <Text style={styles.diagIcon}>📜</Text>
                  <View>
                    <Text style={styles.diagName}>Scroll Pattern</Text>
                    <Text style={styles.diagDetail}>
                      Distance: {modality.scroll.distance.toFixed(2)}σ
                    </Text>
                  </View>
                </View>
                <View style={styles.diagRight}>
                  <Text style={[
                    styles.diagScore,
                    { color: getSimColor(modality.scroll.similarity) },
                  ]}>
                    {(modality.scroll.similarity * 100).toFixed(0)}%
                  </Text>
                  <Text style={[
                    styles.diagLabel,
                    { color: getSimColor(modality.scroll.similarity) },
                  ]}>
                    {getSimLabel(modality.scroll.similarity)}
                  </Text>
                </View>
              </View>
            )}

            {/* IMU */}
            {modality.imu?.available && (
              <View style={styles.diagRow}>
                <View style={styles.diagLeft}>
                  <Text style={styles.diagIcon}>📱</Text>
                  <View>
                    <Text style={styles.diagName}>Device Motion</Text>
                    <Text style={styles.diagDetail}>
                      Distance: {modality.imu.distance.toFixed(2)}σ
                    </Text>
                  </View>
                </View>
                <View style={styles.diagRight}>
                  <Text style={[
                    styles.diagScore,
                    { color: getSimColor(modality.imu.similarity) },
                  ]}>
                    {(modality.imu.similarity * 100).toFixed(0)}%
                  </Text>
                  <Text style={[
                    styles.diagLabel,
                    { color: getSimColor(modality.imu.similarity) },
                  ]}>
                    {getSimLabel(modality.imu.similarity)}
                  </Text>
                </View>
              </View>
            )}

            {modality.fused_similarity !== undefined && (
              <View style={styles.fusedRow}>
                <Text style={styles.fusedLabel}>Fused Similarity</Text>
                <Text style={[
                  styles.fusedValue,
                  { color: getSimColor(modality.fused_similarity) },
                ]}>
                  {(modality.fused_similarity * 100).toFixed(1)}%
                </Text>
              </View>
            )}
          </View>
        )}

        <Text style={styles.explanation}>
          Your typing and interaction pattern doesn't match the enrolled user.
          Enter your password to verify your identity and unlock.
        </Text>

        <TextInput
          style={styles.input}
          placeholder="Enter password to unlock"
          placeholderTextColor="#555"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          returnKeyType="go"
          onSubmitEditing={handleUnlock}
        />

        <TouchableOpacity
          style={[styles.unlockBtn, (!password || loading) && styles.btnDisabled]}
          onPress={handleUnlock}
          disabled={!password || loading}
          activeOpacity={0.85}
        >
          {loading
            ? <ActivityIndicator color="#fff" />
            : <Text style={styles.unlockText}>Unlock</Text>}
        </TouchableOpacity>

        <TouchableOpacity onPress={handleLogout} style={styles.logoutLink}>
          <Text style={styles.logoutText}>Log out instead</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0A0F',
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingHorizontal: 28,
    paddingVertical: 40,
  },
  content: { alignItems: 'center', gap: 14 },
  lockIcon: { fontSize: 64 },
  title:    { color: '#FF3366', fontSize: 28, fontWeight: '800' },
  subtitle: { color: '#AAA', fontSize: 15, textAlign: 'center' },
  riskCard: {
    backgroundColor: '#1A0A10',
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: '#FF3366',
    padding: 20,
    width: '100%',
    alignItems: 'center',
    gap: 4,
  },
  riskLabel: { color: '#FF6688', fontSize: 12, fontWeight: '700', letterSpacing: 1.5, textTransform: 'uppercase' },
  riskValue: { color: '#FF3366', fontSize: 36, fontWeight: '800' },
  trustValue: { color: '#FF8899', fontSize: 16, fontWeight: '600' },
  riskLevel: { color: '#FF6688', fontSize: 14, fontWeight: '700', letterSpacing: 1 },

  // Diagnostics card
  diagnosticsCard: {
    backgroundColor: '#111118',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#2A2A3A',
    padding: 16,
    width: '100%',
    gap: 12,
  },
  diagTitle: { color: '#FFF', fontSize: 16, fontWeight: '700' },
  diagSubtitle: { color: '#777', fontSize: 12 },
  diagRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#0D0D14',
    borderRadius: 10,
    padding: 12,
    borderWidth: 1,
    borderColor: '#1E1E2A',
  },
  diagLeft: { flexDirection: 'row', alignItems: 'center', gap: 10, flex: 1 },
  diagIcon: { fontSize: 22 },
  diagName: { color: '#CCC', fontSize: 14, fontWeight: '600' },
  diagDetail: { color: '#666', fontSize: 11, marginTop: 2 },
  diagRight: { alignItems: 'flex-end' },
  diagScore: { fontSize: 22, fontWeight: '800' },
  diagLabel: { fontSize: 10, fontWeight: '700', letterSpacing: 0.5, marginTop: 1 },
  fusedRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#1E1E2A',
  },
  fusedLabel: { color: '#888', fontSize: 13, fontWeight: '600' },
  fusedValue: { fontSize: 18, fontWeight: '800' },

  explanation: { color: '#777', fontSize: 13, textAlign: 'center', lineHeight: 19 },
  input: {
    backgroundColor: '#16161F',
    borderWidth: 1,
    borderColor: '#2A2A3A',
    borderRadius: 10,
    color: '#FFF',
    fontSize: 17,
    paddingHorizontal: 16,
    paddingVertical: 12,
    width: '100%',
  },
  unlockBtn: {
    backgroundColor: '#FF3366',
    borderRadius: 12,
    paddingVertical: 16,
    width: '100%',
    alignItems: 'center',
  },
  btnDisabled: { opacity: 0.4 },
  unlockText: { color: '#FFF', fontSize: 17, fontWeight: '700' },
  logoutLink: { marginTop: 4 },
  logoutText: { color: '#666', fontSize: 14 },
});
