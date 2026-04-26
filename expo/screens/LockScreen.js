// screens/LockScreen.js — Shown when behavioral mismatch detected
import React, { useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity,
  StyleSheet, Alert, ActivityIndicator,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { loginUser } from '../utils/api';
import { STORAGE_KEYS, APP_NAME } from '../config';

export default function LockScreen({ route, navigation }) {
  const risk      = route?.params?.risk ?? 0;
  const riskLevel = route?.params?.risk_level ?? 'critical';
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

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.lockIcon}>🔒</Text>
        <Text style={styles.title}>Account Locked</Text>
        <Text style={styles.subtitle}>
          {APP_NAME} detected unusual behavior
        </Text>

        <View style={styles.riskCard}>
          <Text style={styles.riskLabel}>Threat Level</Text>
          <Text style={styles.riskValue}>{(risk * 100).toFixed(1)}%</Text>
          <Text style={styles.riskLevel}>{riskLevel.toUpperCase()}</Text>
        </View>

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
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0A0F',
    justifyContent: 'center',
    paddingHorizontal: 28,
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
  riskLevel: { color: '#FF6688', fontSize: 14, fontWeight: '700', letterSpacing: 1 },
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
