// screens/LoginScreen.js — Returning user login
import React, { useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity,
  StyleSheet, ActivityIndicator, Alert,
  KeyboardAvoidingView, Platform, ScrollView,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { loginUser } from '../utils/api';
import { STORAGE_KEYS, APP_NAME } from '../config';

export default function LoginScreen({ navigation }) {
  const [name,     setName]     = useState('');
  const [password, setPassword] = useState('');
  const [loading,  setLoading]  = useState(false);

  async function handleLogin() {
    if (!name.trim() || !password) {
      Alert.alert('Required', 'Enter your name and password.');
      return;
    }

    setLoading(true);
    try {
      const { user_id, name: userName, is_enrolled } = await loginUser(name.trim(), password);
      await AsyncStorage.setItem(STORAGE_KEYS.USER_ID, user_id);
      await AsyncStorage.setItem(STORAGE_KEYS.USER_NAME, userName);

      if (is_enrolled) {
        await AsyncStorage.setItem(STORAGE_KEYS.IS_ENROLLED, 'true');
        navigation.replace('Dashboard');
      } else {
        navigation.replace('Enrollment', { user_id, user_name: userName });
      }
    } catch (err) {
      const msg = err?.response?.data?.detail || 'Login failed. Check your credentials.';
      Alert.alert('Login failed', msg);
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
        <View style={styles.header}>
          <Text style={styles.logo}>🛡️</Text>
          <Text style={styles.title}>Welcome back</Text>
          <Text style={styles.subtitle}>Log in to {APP_NAME}</Text>
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.inputLabel}>Name</Text>
          <TextInput
            style={styles.input}
            placeholder="Your name"
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
            placeholder="Your password"
            placeholderTextColor="#555"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
            returnKeyType="go"
            onSubmitEditing={handleLogin}
          />
        </View>

        <TouchableOpacity
          style={[styles.btn, (!name.trim() || !password || loading) && styles.btnDisabled]}
          onPress={handleLogin}
          disabled={!name.trim() || !password || loading}
          activeOpacity={0.85}
        >
          {loading
            ? <ActivityIndicator color="#fff" />
            : <Text style={styles.btnText}>Log In</Text>}
        </TouchableOpacity>

        <TouchableOpacity onPress={() => navigation.navigate('Welcome')} activeOpacity={0.7}>
          <Text style={styles.linkText}>Don't have an account? Register</Text>
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
  header: { alignItems: 'center', gap: 6, marginBottom: 8 },
  logo:     { fontSize: 48 },
  title:    { color: '#FFFFFF', fontSize: 26, fontWeight: '800' },
  subtitle: { color: '#888', fontSize: 15 },
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
