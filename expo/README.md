# TrueCred Mobile App

React Native app built with Expo. Handles account setup, behavioral enrollment, and continuous trust monitoring while the user is in an active session.

## What it does

After a normal username/password login, TrueCred builds a behavioral profile during enrollment:

1. **Keystroke step** — User types three fixed phrases (`vkerjpwu`, `kkjjkkjj`, `13792846`) on a custom keyboard. Hold time, flight time, and pressure are recorded.
2. **Scroll step** — User scrolls through content. The app captures velocity, direction, and touch path.
3. **IMU step** — Device gyroscope and orientation are sampled for ~30 seconds while the user holds the phone naturally.
4. **Review & submit** — Data is sent to `POST /enroll` on the backend.

On the dashboard, the app keeps collecting keystrokes, scrolls, and IMU samples. Every 15 seconds it calls `POST /verify`. The backend returns a trust score; if it drops below the lock threshold, navigation redirects to the lock screen.

## Screens

| Screen | File | Purpose |
|--------|------|---------|
| Welcome | `screens/WelcomeScreen.js` | Entry point, routes to login or enrollment |
| Login | `screens/LoginScreen.js` | Account creation and sign-in |
| Enrollment | `screens/EnrollmentScreen.js` | Multi-step calibration wizard |
| Dashboard | `screens/DashboardScreen.js` | Main UI with live trust indicator |
| Lock | `screens/LockScreen.js` | Shown when trust score is too low |

## Configuration

Edit `config.js`:

```js
export const API_BASE_URL =
  process.env.EXPO_PUBLIC_API_BASE_URL || 'http://172.18.240.244:8000';
```

Set `EXPO_PUBLIC_API_BASE_URL` to your backend's IP when running on a physical device.

Other tunables: verification interval (`VERIFY_INTERVAL_MS`), scroll minimums, IMU duration, and trust thresholds.

## Running

```bash
npm install
npx expo start
```

Press `a` for Android emulator or scan the QR code with Expo Go on a phone. Sensors work best on real hardware.

## Offline behavior

If enrollment fails due to network issues, the payload is queued in AsyncStorage (`PENDING_UPLOADS`) and retried later via `utils/api.js`.

## Key files

- `utils/sensors.js` — IMU collection helpers
- `utils/api.js` — HTTP client for backend endpoints
- `components/Gboardkeyboard.js` — Custom keyboard for consistent keystroke capture
- `screens/steps/` — Individual enrollment step components
