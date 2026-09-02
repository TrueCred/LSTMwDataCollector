import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import { IMU_DURATION_S } from '../../config';

const SAMPLE_INTERVAL_MS = 100;

function toDegrees(rad) {
  return rad * (180 / Math.PI);
}

export default function IMUStep({ onComplete }) {
  const [samples, setSamples] = useState([]);
  const [running, setRunning] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(IMU_DURATION_S);

  const gyroRef = useRef({ x: 0, y: 0, z: 0 });
  const samplesRef = useRef([]);
  const secondsLeftRef = useRef(IMU_DURATION_S);
  const finishedRef = useRef(false);
  const gyroSubRef = useRef(null);
  const accelSubRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    return () => stopCollection();
  }, []);

  function stopCollection() {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (gyroSubRef.current) {
      gyroSubRef.current.remove();
      gyroSubRef.current = null;
    }

    if (accelSubRef.current) {
      accelSubRef.current.remove();
      accelSubRef.current = null;
    }

    setRunning(false);
  }

  function finishCollection(collected) {
    if (finishedRef.current) return;
    finishedRef.current = true;
    stopCollection();
    onComplete(collected);
  }

  function startCollection() {
    if (running) return;

    finishedRef.current = false;
    samplesRef.current = [];
    secondsLeftRef.current = IMU_DURATION_S;
    setSamples([]);
    setSecondsLeft(IMU_DURATION_S);
    setRunning(true);

    Gyroscope.setUpdateInterval(SAMPLE_INTERVAL_MS);
    Accelerometer.setUpdateInterval(SAMPLE_INTERVAL_MS);

    gyroSubRef.current = Gyroscope.addListener((gyro) => {
      gyroRef.current = gyro;
    });

    accelSubRef.current = Accelerometer.addListener((accel) => {
      const { x, y, z } = accel;
      const pitch = toDegrees(Math.atan2(-x, Math.sqrt((y * y) + (z * z))));
      const roll = toDegrees(Math.atan2(y, z));
      const g = gyroRef.current;

      const sample = {
        gyro_x: g.x ?? 0,
        gyro_y: g.y ?? 0,
        gyro_z: g.z ?? 0,
        tilt_pitch: Number.isFinite(pitch) ? pitch : 0,
        tilt_roll: Number.isFinite(roll) ? roll : 0,
        timestamp: Date.now(),
      };

      samplesRef.current = samplesRef.current.concat(sample);
      setSamples(samplesRef.current);
    });

    timerRef.current = setInterval(() => {
      if (finishedRef.current) return;

      const next = secondsLeftRef.current - 1;
      secondsLeftRef.current = next > 0 ? next : 0;
      setSecondsLeft(secondsLeftRef.current);

      if (secondsLeftRef.current <= 0) {
        finishCollection(samplesRef.current);
      }
    }, 1000);
  }

  function completeNow() {
    finishCollection(samplesRef.current);
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Motion Baseline</Text>
      <Text style={styles.subtitle}>
        Hold your phone naturally and move as you usually do for {IMU_DURATION_S} seconds.
      </Text>

      <View style={styles.card}>
        <Text style={styles.metric}>Samples: {samples.length}</Text>
        <Text style={styles.metric}>Time left: {secondsLeft}s</Text>
      </View>

      {!running ? (
        <TouchableOpacity style={styles.btn} onPress={startCollection}>
          <Text style={styles.btnText}>Start IMU Capture</Text>
        </TouchableOpacity>
      ) : (
        <View style={styles.runningRow}>
          <ActivityIndicator color="#1A73E8" />
          <Text style={styles.runningText}>Collecting sensor data...</Text>
        </View>
      )}

      <TouchableOpacity
        style={[styles.btn, styles.secondaryBtn, (samples.length === 0 || running) && styles.btnDisabled]}
        disabled={samples.length === 0 || running}
        onPress={completeNow}
      >
        <Text style={styles.btnText}>Continue</Text>
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
    marginTop: 8,
    marginBottom: 18,
    textAlign: 'center',
  },
  card: {
    backgroundColor: '#151515',
    borderWidth: 1,
    borderColor: '#252525',
    borderRadius: 12,
    padding: 16,
    gap: 8,
    marginBottom: 16,
  },
  metric: {
    color: '#D6D6D6',
    fontSize: 16,
  },
  btn: {
    backgroundColor: '#1A73E8',
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: 'center',
    marginTop: 10,
  },
  secondaryBtn: {
    backgroundColor: '#24508B',
  },
  btnDisabled: {
    opacity: 0.4,
  },
  btnText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
  runningRow: {
    marginTop: 10,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 8,
  },
  runningText: {
    color: '#8FB6ED',
  },
});
