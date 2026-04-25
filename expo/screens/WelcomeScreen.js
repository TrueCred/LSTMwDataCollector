// ─────────────────────────────────────────────────────────────────────────────
// screens/WelcomeScreen.js
// Entry point: collect user name, create user on backend, navigate to enrollment.
// Triple-tap title to dump dev info to console.
// ─────────────────────────────────────────────────────────────────────────────

import React, { useState, useRef, useEffect } from 'react';
import {
  View, Text, TextInput, TouchableOpacity,
  StyleSheet, ActivityIndicator, Alert,
  KeyboardAvoidingView, Platform, ScrollView,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { createUser, flushQueue, loadQueue } from '../utils/api';
import { STORAGE_KEYS } from '../config';

export default function WelcomeScreen({ navigation }) {
  const [name,        setName]        = useState('');
  const [loading,     setLoading]     = useState(false);
  const [pendingCount, setPendingCount] = useState(0);

  // ── Dev mode: triple-tap title ──────────────────────────────────────────
  const tapCount = useRef(0);
  const tapTimer = useRef(null);

  async function handleTitleTap() {
    tapCount.current += 1;
    if (tapTimer.current) clearTimeout(tapTimer.current);
    tapTimer.current = setTimeout(() => { tapCount.current = 0; }, 600);
    if (tapCount.current >= 3) {
      tapCount.current = 0;
      const queue = await loadQueue();
      console.log('[DEV] Pending queue:', JSON.stringify(queue, null, 2));
      const stored = await AsyncStorage.getItem(STORAGE_KEYS.USER_ID);
      console.log('[DEV] Stored user_id:', stored);
      Alert.alert('Dev Mode', `Pending uploads: ${queue.length}\nUser ID: ${stored ?? 'none'}\n(See console for full dump)`);
    }
  }

  // ── Load pending count on mount & flush any queued uploads ──────────────
  useEffect(() => {
    (async () => {
      const queue = await loadQueue();
      setPendingCount(queue.length);
      if (queue.length > 0) {
        const uploaded = await flushQueue();
        if (uploaded > 0) {
          const newQueue = await loadQueue();
          setPendingCount(newQueue.length);
        }
      }
    })();
  }, []);

  // ── Start session ────────────────────────────────────────────────────────
  async function handleStart() {
    const trimmed = name.trim();
    if (!trimmed) {
      Alert.alert('Name required', 'Please enter your first name or a nickname.');
      return;
    }

    setLoading(true);
    try {
      const { user_id, name: userName } = await createUser(trimmed);
      await AsyncStorage.setItem(STORAGE_KEYS.USER_ID,   user_id);
      await AsyncStorage.setItem(STORAGE_KEYS.USER_NAME, userName);
      navigation.navigate('Enrollment', { user_id, user_name: userName });
    } catch (err) {
      // Offline fallback: generate a local UUID-ish ID
      console.warn('createUser failed, using local id:', err.message);
      const local_id = `local_${Date.now()}_${Math.random().toString(36).slice(2)}`;
      await AsyncStorage.setItem(STORAGE_KEYS.USER_ID,   local_id);
      await AsyncStorage.setItem(STORAGE_KEYS.USER_NAME, trimmed);
      navigation.navigate('Enrollment', { user_id: local_id, user_name: trimmed });
    } finally {
      setLoading(false);
    }
  }

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        {/* Pending badge */}
        {pendingCount > 0 && (
          <View style={styles.badge}>
            <Text style={styles.badgeText}>📤 {pendingCount} pending upload{pendingCount > 1 ? 's' : ''}</Text>
          </View>
        )}

        {/* Logo / header */}
        <TouchableOpacity onPress={handleTitleTap} activeOpacity={1}>
          <View style={styles.logoContainer}>
            <Text style={styles.logo}>🛡️</Text>
            <Text style={styles.title}>Sentinel Biometrics</Text>
          </View>
        </TouchableOpacity>

        <Text style={styles.subtitle}>Help train passwordless authentication</Text>

        <View style={styles.infoCard}>
          <Text style={styles.infoText}>⏱  Takes about 3 minutes</Text>
          <Text style={styles.infoText}>🔒  No personal data is shared</Text>
          <Text style={styles.infoText}>📱  Uses typing rhythm, scroll, and motion</Text>
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.inputLabel}>First name or nickname</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g. Alex"
            placeholderTextColor="#555"
            value={name}
            onChangeText={setName}
            autoCapitalize="words"
            returnKeyType="go"
            onSubmitEditing={handleStart}
          />
        </View>

        <TouchableOpacity
          style={[styles.btn, (!name.trim() || loading) && styles.btnDisabled]}
          onPress={handleStart}
          disabled={!name.trim() || loading}
          activeOpacity={0.85}
        >
          {loading
            ? <ActivityIndicator color="#fff" />
            : <Text style={styles.btnText}>Start →</Text>}
        </TouchableOpacity>

        <Text style={styles.footer}>Sentinel Biometrics Study · v1.0</Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: '#0D0D0D' },
  container: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 28,
    paddingVertical: 40,
    gap: 20,
  },
  badge: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: '#2A1A00',
    borderWidth: 1,
    borderColor: '#FF9800',
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  badgeText: { color: '#FF9800', fontSize: 12 },
  logoContainer: { alignItems: 'center', gap: 8 },
  logo:    { fontSize: 56 },
  title:   { color: '#FFFFFF', fontSize: 28, fontWeight: '700', letterSpacing: 0.5 },
  subtitle:{ color: '#888', fontSize: 16, textAlign: 'center' },
  infoCard: {
    backgroundColor: '#161616',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#2A2A2A',
    padding: 18,
    width: '100%',
    gap: 10,
  },
  infoText: { color: '#AAA', fontSize: 14 },
  inputGroup: { width: '100%', gap: 6 },
  inputLabel: { color: '#888', fontSize: 13, marginLeft: 4 },
  input: {
    backgroundColor: '#1C1C1C',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 10,
    color: '#FFF',
    fontSize: 18,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  btn: {
    backgroundColor: '#1A73E8',
    borderRadius: 12,
    paddingVertical: 16,
    paddingHorizontal: 40,
    width: '100%',
    alignItems: 'center',
  },
  btnDisabled: { opacity: 0.4 },
  btnText: { color: '#FFF', fontSize: 18, fontWeight: '700' },
  footer: { color: '#333', fontSize: 12, marginTop: 8 },
});