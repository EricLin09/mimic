//! Allocation-free coded latency probes. Original sequence/detector/tests migrated
//! from b603bd6:rt-proto/src/lib.rs; realtime wiring remains the caller's job.

/// Fixed bipolar probe; each chip lasts four samples to tolerate audio filtering.
pub const PROBE_CHIPS: [f32; 31] = [
    1., 1., 1., 1., 1., -1., -1., 1., 1., -1., 1., -1., 1., -1., -1., -1., -1., 1., -1., -1., 1.,
    -1., -1., -1., 1., 1., 1., -1., 1., -1., 1.,
];
pub const PROBE_LEN: usize = 124;
pub fn probe_sample(index: usize) -> f32 {
    PROBE_CHIPS[index / 4] * 0.25
}
pub struct ProbeDetector {
    data: [f32; PROBE_LEN],
    cursor: usize,
    count: usize,
}
impl Default for ProbeDetector {
    fn default() -> Self {
        Self {
            data: [0.0; PROBE_LEN],
            cursor: 0,
            count: 0,
        }
    }
}
impl ProbeDetector {
    pub fn push(&mut self, x: f32) -> bool {
        self.data[self.cursor] = x;
        self.cursor = (self.cursor + 1) % PROBE_LEN;
        self.count = (self.count + 1).min(PROBE_LEN);
        if self.count < PROBE_LEN {
            return false;
        }
        let mut dot = 0.0_f32;
        let mut energy = 0.0_f32;
        for i in 0..PROBE_LEN {
            let s = self.data[(self.cursor + i) % PROBE_LEN];
            dot += s * probe_sample(i);
            energy += s * s;
        }
        energy > 0.001 * PROBE_LEN as f32
            && dot.abs() > 0.78 * (energy * PROBE_LEN as f32 * 0.0625).sqrt()
    }
}

/// A generated sample and its optional leading-edge marker in this source's clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeSample {
    pub sample: f32,
    pub start_sample: Option<u64>,
}

/// Periodic source in sample units; the first probe begins at sample zero.
///
/// No clock reading, I/O, allocation, locks or unbounded history. Every start is
/// returned by `next_sample` or delivered through `fill`'s callback; the caller
/// can forward markers into its own fixed queue. This avoids silently losing
/// earlier starts when one output callback spans several probe intervals.
pub struct ProbeSource {
    interval_samples: u64,
    next_sample_index: u64,
    last_start_sample: Option<u64>,
    pulses_emitted: u64,
}
impl ProbeSource {
    pub fn new(interval_samples: u64) -> Self {
        assert!(
            interval_samples >= PROBE_LEN as u64,
            "probe intervals must not overlap"
        );
        Self {
            interval_samples,
            next_sample_index: 0,
            last_start_sample: None,
            pulses_emitted: 0,
        }
    }
    pub fn next_sample(&mut self) -> ProbeSample {
        let absolute = self.next_sample_index;
        self.next_sample_index = absolute
            .checked_add(1)
            .expect("probe sample clock exhausted");
        let position = absolute % self.interval_samples;
        let start_sample = if position == 0 {
            self.last_start_sample = Some(absolute);
            self.pulses_emitted += 1;
            Some(absolute)
        } else {
            None
        };
        let sample = if position < PROBE_LEN as u64 {
            probe_sample(position as usize)
        } else {
            0.0
        };
        ProbeSample {
            sample,
            start_sample,
        }
    }
    pub fn fill(&mut self, out: &mut [f32], mut on_start: impl FnMut(u64)) {
        for value in out {
            let emitted = self.next_sample();
            *value = emitted.sample;
            if let Some(start) = emitted.start_sample {
                on_start(start);
            }
        }
    }
    pub fn next_sample_index(&self) -> u64 {
        self.next_sample_index
    }
    pub fn last_start_sample(&self) -> Option<u64> {
        self.last_start_sample
    }
    pub fn pulses_emitted(&self) -> u64 {
        self.pulses_emitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn probe_recognizes_delayed_scaled_and_inverted() {
        for scale in [0.4, 1.0, -0.7] {
            let mut d = ProbeDetector::default();
            for _ in 0..500 {
                assert!(!d.push(0.0));
            }
            let mut seen = false;
            for i in 0..PROBE_LEN {
                seen |= d.push(probe_sample(i) * scale);
            }
            assert!(seen);
        }
    }
    #[test]
    fn steady_tone_is_not_a_probe() {
        let mut d = ProbeDetector::default();
        for i in 0..48000 {
            assert!(!d.push((i as f32 * 0.05).sin() * 0.3));
        }
    }
    #[test]
    fn probe_source_periodic_markers_and_silence() {
        let interval = PROBE_LEN as u64 + 17;
        let mut source = ProbeSource::new(interval);
        assert_eq!(source.last_start_sample(), None);
        for index in 0..(interval * 3 + PROBE_LEN as u64) {
            let item = source.next_sample();
            let position = index % interval;
            assert_eq!(item.start_sample, (position == 0).then_some(index));
            assert_eq!(
                item.sample,
                if position < PROBE_LEN as u64 {
                    probe_sample(position as usize)
                } else {
                    0.0
                }
            );
        }
        assert_eq!(source.last_start_sample(), Some(interval * 3));
        assert_eq!(source.pulses_emitted(), 4);
        assert_eq!(source.next_sample_index(), interval * 3 + PROBE_LEN as u64);
    }
    #[test]
    fn probe_source_fill_preserves_all_starts_across_callback_sizes() {
        let interval = PROBE_LEN as u64 + 13;
        let mut source = ProbeSource::new(interval);
        let mut markers = Vec::new();
        let mut samples = Vec::new();
        for size in [17, 480, 13, 300] {
            let mut block = vec![0.0; size];
            source.fill(&mut block, |start| markers.push(start));
            samples.extend(block);
        }
        let expected: Vec<_> = (0..samples.len() as u64)
            .step_by(interval as usize)
            .collect();
        assert_eq!(markers, expected);
        assert_eq!(source.pulses_emitted(), markers.len() as u64);
        for (i, &sample) in samples.iter().enumerate() {
            let position = i as u64 % interval;
            assert_eq!(
                sample,
                if position < PROBE_LEN as u64 {
                    probe_sample(position as usize)
                } else {
                    0.0
                }
            );
        }
    }
    #[test]
    #[should_panic(expected = "probe intervals must not overlap")]
    fn probe_source_rejects_overlapping_interval() {
        let _ = ProbeSource::new(PROBE_LEN as u64 - 1);
    }
}
