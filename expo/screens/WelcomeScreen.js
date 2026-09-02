// screens/WelcomeScreen.js — Registration + entry point
import React, { useState, useEffect } from 'react';
import {
  View, Text, TextInput, TouchableOpacity,
  StyleSheet, ActivityIndicator, Alert,
  KeyboardAvoidingView, Platform, ScrollView,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { createUser } from '../utils/api';
import { STORAGE_KEYS, APP_NAME } from '../config';

export default function WelcomeScreen({ navigation }) {
  const [name,     setName]     = useState('');
  const [password, setPassword] = useState('');
  const [loading,  setLoading]  = useState(false);
  const [checking, setChecking] = useState(true);

  // Check if already logged in
  useEffect(() => {
    (async () => {
      const uid = await AsyncStorage.getItem(STORAGE_KEYS.USER_ID);
      const enrolled = await AsyncStorage.getItem(STORAGE_KEYS.IS_ENROLLED);
      if (uid && enrolled === 'true') {
        navigation.replace('Dashboard');
      } else if (uid) {
        // Registered but not enrolled
        const uname = await AsyncStorage.getItem(STORAGE_KEYS.USER_NAME);
        navigation.replace('Enrollment', { user_id: uid, user_name: uname });
      }
      setChecking(false);
    })();
  }, []);

  async function handleRegister() {
    const trimmedName = name.trim();
    if (!trimmedName || !password) {
      Alert.alert('Required', 'Please enter a name and password.');
      return;
    }

    setLoading(true);
    try {
      const { user_id, name: userName } = await createUser(trimmedName, password);
      await AsyncStorage.setItem(STORAGE_KEYS.USER_ID, user_id);
      await AsyncStorage.setItem(STORAGE_KEYS.USER_NAME, userName);
      navigation.replace('Enrollment', { user_id, user_name: userName });
    } catch (err) {
      const msg = err?.response?.data?.detail || err.message;
      Alert.alert('Registration failed', msg);
    } finally {
      setLoading(false);
    }
  }

  if (checking) {
    return (
      <View style={[styles.container, { justifyContent: 'center' }]}>
        <ActivityIndicator size="large" color="#6C5CE7" />
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <View style={styles.logoContainer}>
          <Text style={styles.logo}>🛡️</Text>
          <Text style={styles.title}>{APP_NAME}</Text>
        </View>

        <Text style={styles.subtitle}>Behavioral authentication that protects you</Text>

        <View style={styles.infoCard}>
          <Text style={styles.infoText}>🔐  Learns your unique typing & touch patterns</Text>
          <Text style={styles.infoText}>📱  Continuously verifies it's really you</Text>
          <Text style={styles.infoText}>⚡  Locks out impostors automatically</Text>
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.inputLabel}>Name</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g. Alex"
            placeholderTextColor="#555"
            value={name}
            onChangeText={setName}
            autoCapitalize="words"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.inputLabel}>Password</Text>
          <TextInput
            style={styles.input}
            placeholder="Choose a password"
            placeholderTextColor="#555"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
            returnKeyType="go"
            onSubmitEditing={handleRegister}
          />
        </View>

        <TouchableOpacity
          style={[styles.btn, (!name.trim() || !password || loading) && styles.btnDisabled]}
          onPress={handleRegister}
          disabled={!name.trim() || !password || loading}
          activeOpacity={0.85}
        >
          {loading
            ? <ActivityIndicator color="#fff" />
            : <Text style={styles.btnText}>Register & Enroll</Text>}
        </TouchableOpacity>

        <TouchableOpacity onPress={() => navigation.navigate('Login')} activeOpacity={0.7}>
          <Text style={styles.linkText}>Already have an account? Log in</Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: '#0A0A0F' },
  container: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 28,
    paddingVertical: 40,
    gap: 18,
  },
  logoContainer: { alignItems: 'center', gap: 8 },
  logo:     { fontSize: 56 },
  title:    { color: '#FFFFFF', fontSize: 30, fontWeight: '800', letterSpacing: 1 },
  subtitle: { color: '#888', fontSize: 15, textAlign: 'center', marginBottom: 4 },
  infoCard: {
    backgroundColor: '#13131A',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#1E1E2A',
    padding: 18,
    width: '100%',
    gap: 10,
  },
  infoText: { color: '#AAA', fontSize: 14 },
  inputGroup: { width: '100%', gap: 6 },
  inputLabel: { color: '#888', fontSize: 13, marginLeft: 4 },
  input: {
    backgroundColor: '#16161F',
    borderWidth: 1,
    borderColor: '#2A2A3A',
    borderRadius: 10,
    color: '#FFF',
    fontSize: 17,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  btn: {
    backgroundColor: '#6C5CE7',
    borderRadius: 12,
    paddingVertical: 16,
    width: '100%',
    alignItems: 'center',
  },
  btnDisabled: { opacity: 0.4 },
  btnText: { color: '#FFF', fontSize: 17, fontWeight: '700' },
  linkText: { color: '#6C5CE7', fontSize: 14, marginTop: 4 },
});