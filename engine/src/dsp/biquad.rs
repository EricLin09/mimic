//! One biquad section (transposed direct form II). The gate uses it as a side-chain high-pass:
//! handling rumble and breath pops live below the voice band and should not open the gate.

use std::f32::consts::{FRAC_1_SQRT_2, PI};

#[derive(Clone, Debug)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// Second-order Butterworth high-pass (RBJ cookbook, Q = 1/sqrt 2).
    pub fn highpass(rate: f32, cutoff: f32) -> Self {
        let w0 = 2.0 * PI * cutoff / rate;
        let (sn, cs) = w0.sin_cos();
        let alpha = sn / (2.0 * FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 + cs) / 2.0 / a0,
            b1: -(1.0 + cs) / a0,
            b2: (1.0 + cs) / 2.0 / a0,
            a1: -2.0 * cs / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    /// RMS of the filtered signal; the filter state carries over to the next call.
    pub fn rms(&mut self, x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        let sum: f32 = x
            .iter()
            .map(|&v| {
                let y = self.tick(v);
                y * y
            })
            .sum();
        (sum / x.len() as f32).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settled_level(hz: f32) -> f32 {
        let mut f = Biquad::highpass(48_000.0, 150.0);
        let x: Vec<f32> = (0..48_000).map(|i| (2.0 * PI * hz * i as f32 / 48_000.0).sin()).collect();
        f.rms(&x[..24_000]);
        f.rms(&x[24_000..])
    }

    #[test]
    fn passes_the_voice_band_and_rejects_rumble() {
        let unit_sine = FRAC_1_SQRT_2;
        assert!((settled_level(1000.0) / unit_sine - 1.0).abs() < 0.02);
        // 40 Hz sits more than 20 dB down
        assert!(settled_level(40.0) / unit_sine < 0.1);
    }
}
