import React, { useMemo, useRef, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  PixelRatio,
} from 'react-native';
import { MIN_SCROLLS } from '../../config';

const DPI = PixelRatio.get() * 160;

export default function ScrollStep({ onComplete }) {
  const [events, setEvents] = useState([]);
  const [lastY, setLastY] = useState(0);
  const [lastT, setLastT] = useState(Date.now());
  const finished = useRef(false);

  const progress = Math.min(1, events.length / MIN_SCROLLS);

  const rows = useMemo(
    () => Array.from({ length: 50 }, (_, i) => `Scroll row ${i + 1} · Keep scrolling up and down`),
    []
  );

  function handleScroll(e) {
    if (finished.current) return;

    const y = e.nativeEvent.contentOffset.y;
    const t = Date.now();

    const dy = y - lastY;
    const dt = Math.max(1, t - lastT);

    if (Math.abs(dy) >= 20 && dt >= 30) {
      const velocity = Math.abs(dy) / dt * 1000;
      const direction = dy >= 0 ? 180 : 0;

      const event = {
        velocity_px_per_sec: velocity,
        direction_deg: direction,
        distance_px: Math.abs(dy),
        avg_pressure: 0.5,
        pixel_density_dpi: DPI,
        timestamp: t,
      };

      setEvents((prev) => prev.concat(event));
    }

    setLastY(y);
    setLastT(t);
  }

  function handleNext() {
    if (finished.current) return;
    finished.current = true;
    onComplete(events);
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Scroll Biometrics</Text>
      <Text style={styles.subtitle}>Scroll up and down naturally until you reach the target.</Text>

      <View style={styles.progressOuter}>
        <View style={[styles.progressInner, { width: `${progress * 100}%` }]} />
      </View>
      <Text style={styles.countLabel}>{events.length}/{MIN_SCROLLS} scroll events</Text>

      <ScrollView
        style={styles.scroller}
        contentContainerStyle={styles.scrollerContent}
        onScroll={handleScroll}
        scrollEventThrottle={16}
        showsVerticalScrollIndicator={false}
      >
        {rows.map((row) => (
          <View key={row} style={styles.rowCard}>
            <Text style={styles.rowText}>{row}</Text>
          </View>
        ))}
      </ScrollView>

      <TouchableOpacity
        style={[styles.btn, events.length < MIN_SCROLLS && styles.btnDisabled]}
        disabled={events.length < MIN_SCROLLS}
        onPress={handleNext}
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
  },
  title: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: '700',
  },
  subtitle: {
    color: '#AAAAAA',
    marginTop: 6,
    marginBottom: 12,
  },
  progressOuter: {
    height: 4,
    borderRadius: 4,
    backgroundColor: '#1F1F1F',
    overflow: 'hidden',
  },
  progressInner: {
    height: '100%',
    backgroundColor: '#1A73E8',
  },
  countLabel: {
    color: '#7C7C7C',
    marginTop: 8,
    marginBottom: 10,
  },
  scroller: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#262626',
    borderRadius: 12,
    backgroundColor: '#111111',
  },
  scrollerContent: {
    padding: 10,
    gap: 8,
  },
  rowCard: {
    backgroundColor: '#181818',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#252525',
    padding: 14,
  },
  rowText: {
    color: '#D6D6D6',
    fontSize: 14,
  },
  btn: {
    marginTop: 12,
    backgroundColor: '#1A73E8',
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: 'center',
  },
  btnDisabled: {
    opacity: 0.4,
  },
  btnText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 16,
  },
});
