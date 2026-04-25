import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';

export default function ReviewStep({
  userName,
  keystrokesCount,
  scrollCount,
  imuCount,
  submitting,
  onSubmit,
}) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Review & Submit</Text>
      <Text style={styles.subtitle}>Check your captured data, then submit enrollment.</Text>

      <View style={styles.card}>
        <Text style={styles.row}>User: <Text style={styles.value}>{userName || 'Unknown'}</Text></Text>
        <Text style={styles.row}>Keystrokes: <Text style={styles.value}>{keystrokesCount}</Text></Text>
        <Text style={styles.row}>Scroll events: <Text style={styles.value}>{scrollCount}</Text></Text>
        <Text style={styles.row}>IMU samples: <Text style={styles.value}>{imuCount}</Text></Text>
      </View>

      <TouchableOpacity style={[styles.btn, submitting && styles.btnDisabled]} onPress={onSubmit} disabled={submitting}>
        {submitting ? (
          <ActivityIndicator color="#FFFFFF" />
        ) : (
          <Text style={styles.btnText}>Submit Enrollment</Text>
        )}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0D0D0D',
    padding: 16,
    justifyContent: 'center',
  },
  title: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: '700',
    textAlign: 'center',
  },
  subtitle: {
    color: '#AAAAAA',
    textAlign: 'center',
    marginTop: 8,
    marginBottom: 16,
  },
  card: {
    backgroundColor: '#151515',
    borderWidth: 1,
    borderColor: '#252525',
    borderRadius: 12,
    padding: 16,
    gap: 12,
  },
  row: {
    color: '#D3D3D3',
    fontSize: 16,
  },
  value: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  btn: {
    marginTop: 18,
    backgroundColor: '#1A73E8',
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: 'center',
  },
  btnDisabled: {
    opacity: 0.55,
  },
  btnText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 16,
  },
});
