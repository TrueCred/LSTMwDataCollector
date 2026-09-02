// ─────────────────────────────────────────────────────────────────────────────
// utils/sensors.js
// Helpers to start/stop IMU recording using expo-sensors.
// ─────────────────────────────────────────────────────────────────────────────

import { Gyroscope, DeviceMotion } from 'expo-sensors';

const INTERVAL_MS = 20; // 50 Hz

/**
 * Start collecting IMU samples.
 * @param {(sample: object) => void} onSample  called for each fused sample
 * @returns {{ stop: () => void }}              call stop() to unsubscribe
 */
export function startIMUCollection(onSample) {
  Gyroscope.setUpdateInterval(INTERVAL_MS);
  DeviceMotion.setUpdateInterval(INTERVAL_MS);

  // Keep last gyro reading so we can fuse with DeviceMotion tilt
  let lastGyro = { x: 0, y: 0, z: 0 };

  const gyroSub = Gyroscope.addListener(({ x, y, z }) => {
    lastGyro = { x, y, z };
  });

  const motionSub = DeviceMotion.addListener((motion) => {
    const rotation = motion.rotation ?? {};
    const sample = {
      gyro_x:     lastGyro.x,
      gyro_y:     lastGyro.y,
      gyro_z:     lastGyro.z,
      // DeviceMotion rotation is in radians; convert to degrees
      tilt_pitch: ((rotation.beta  ?? 0) * 180) / Math.PI,
      tilt_roll:  ((rotation.gamma ?? 0) * 180) / Math.PI,
      timestamp:  Date.now(),
    };
    onSample(sample);
  });

  return {
    stop: () => {
      gyroSub.remove();
      motionSub.remove();
    },
  };
}