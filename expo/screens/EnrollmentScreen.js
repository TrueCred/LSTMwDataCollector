import React, { useLayoutEffect, useState } from 'react';
import { Alert, StyleSheet, View } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

import ProgressDots from '../components/ProgressDots';
import KeystrokeStep from './steps/KeystrokeStep';
import ScrollStep from './steps/ScrollStep';
import IMUStep from './steps/IMUStep';
import ReviewStep from './steps/ReviewStep';
import { enrollUser, pushToQueue } from '../utils/api';
import { STORAGE_KEYS } from '../config';

const STEP_LABELS = ['Type', 'Scroll', 'Motion', 'Submit'];

export default function EnrollmentScreen({ route, navigation }) {
  const userId = route?.params?.user_id ?? null;
  const userName = route?.params?.user_name ?? 'Unknown';

  const [step, setStep] = useState(0);
  const [keystrokes, setKeystrokes] = useState([]);
  const [scrolls, setScrolls] = useState([]);
  const [imu, setImu] = useState([]);
  const [submitting, setSubmitting] = useState(false);

  useLayoutEffect(() => {
    navigation.setOptions({
      headerLeft: () => null,
      gestureEnabled: false,
    });
  }, [navigation]);

  function onKeystrokeComplete(data) {
    setKeystrokes(data);
    setStep(1);
  }

  function onScrollComplete(data) {
    setScrolls(data);
    setStep(2);
  }

  function onImuComplete(data) {
    setImu(data);
    setStep(3);
  }

  async function handleSubmit() {
    if (submitting) return;

    const payload = {
      user_id: userId,
      user_name: userName,
      keystrokes,
      scrolls,
      imu,
    };

    setSubmitting(true);
    try {
      await enrollUser(payload);
      await AsyncStorage.setItem(STORAGE_KEYS.IS_ENROLLED, 'true');
      Alert.alert(
        'Enrollment complete',
        'Your behavioral profile has been saved. Welcome to TrueCred!',
        [
          {
            text: 'Continue',
            onPress: () => navigation.replace('Dashboard'),
          },
        ]
      );
    } catch (err) {
      await pushToQueue(payload);
      await AsyncStorage.setItem(STORAGE_KEYS.IS_ENROLLED, 'true');
      Alert.alert(
        'Saved Offline',
        'Network issue detected. Your enrollment was queued and will upload later.',
        [
          {
            text: 'OK',
            onPress: () => navigation.replace('Dashboard'),
          },
        ]
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <View style={styles.container}>
      <ProgressDots current={step} total={4} labels={STEP_LABELS} />

      {step === 0 && <KeystrokeStep onComplete={onKeystrokeComplete} />}
      {step === 1 && <ScrollStep onComplete={onScrollComplete} />}
      {step === 2 && <IMUStep onComplete={onImuComplete} />}
      {step === 3 && (
        <ReviewStep
          userName={userName}
          keystrokesCount={keystrokes.length}
          scrollCount={scrolls.length}
          imuCount={imu.length}
          submitting={submitting}
          onSubmit={handleSubmit}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0D0D0D',
  },
});
