//! Allocation-free, full-band correlation-adaptive crossfade.
//!
//! Input signals must be finite, time-aligned and approximately equal-power.
//! This normalizes predicted statistical power, not arbitrary short-time RMS.
//! The gain cap intentionally relaxes exact power preservation near rho = -1.

const POWER_FLOOR: f64 = 1e-20;
const GAIN_LIMIT: f64 = 2.0;

/// Reusable crossfade state. All storage is allocated by `new`.
pub struct AdaptiveXfade {
    n: usize,
    fade_in: Vec<f64>,
}

impl AdaptiveXfade {
    /// Build a complementary sine-squared fade of at least two samples.
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "crossfade requires at least two samples");
        let fade_in = (0..n)
            .map(|i| {
                let angle = std::f64::consts::FRAC_PI_2 * i as f64 / (n - 1) as f64;
                angle.sin().powi(2)
            })
            .collect();
        Self { n, fade_in }
    }

    /// Blend exactly `n` samples without allocation, locks or unsafe code.
    ///
    /// Correlation is estimated after mean removal in double precision.
    /// Near-zero centered energy falls back to rho = 0. Negative correlation
    /// is retained, with common normalization capped at +6.0206 dB. The cap
    /// means the power identity is no longer exact where it is active.
    /// Endpoints retain the original signal samples exactly.
    ///
    /// # Panics
    /// Panics if any slice length differs from the configured length.
    pub fn blend(&mut self, prev_tail: &[f32], new_head: &[f32], out: &mut [f32]) {
        assert_eq!(prev_tail.len(), self.n, "prev_tail length mismatch");
        assert_eq!(new_head.len(), self.n, "new_head length mismatch");
        assert_eq!(out.len(), self.n, "output length mismatch");
        let rho = estimate_correlation(prev_tail, new_head);
        out[0] = prev_tail[0];
        for i in 1..self.n - 1 {
            let (a, b) = gains(self.fade_in[i], rho);
            out[i] = (a * f64::from(prev_tail[i]) + b * f64::from(new_head[i])) as f32;
        }
        out[self.n - 1] = new_head[self.n - 1];
    }
}

fn estimate_correlation(x: &[f32], y: &[f32]) -> f64 {
    let mean_x = x.iter().map(|&v| f64::from(v)).sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().map(|&v| f64::from(v)).sum::<f64>() / y.len() as f64;
    let (mut energy_x, mut energy_y, mut cross) = (0.0, 0.0, 0.0);
    for (&x, &y) in x.iter().zip(y) {
        let xc = f64::from(x) - mean_x;
        let yc = f64::from(y) - mean_y;
        energy_x += xc * xc;
        energy_y += yc * yc;
        cross += xc * yc;
    }
    let power = energy_x * energy_y;
    if power <= POWER_FLOOR {
        0.0
    } else {
        (cross / power.sqrt()).clamp(-1.0, 1.0)
    }
}

fn gains(v: f64, rho: f64) -> (f64, f64) {
    let u = 1.0 - v;
    let denominator = u * u + v * v + 2.0 * rho * u * v;
    let normalizer = (1.0 / denominator.max(POWER_FLOOR).sqrt()).min(GAIN_LIMIT);
    (u * normalizer, v * normalizer)
}

#[cfg(test)]
#[path = "xfade_fixture.rs"]
mod fixture;

#[cfg(test)]
mod tests {
    use super::{AdaptiveXfade, estimate_correlation, fixture, gains};
    use std::hint::black_box;
    use std::time::Instant;

    fn lcg(state: &mut u32) -> f64 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f64::from(*state >> 8) / 8_388_608.0 - 1.0
    }

    fn inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut seed = 20_261_027_u32;
        let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let phase = 2.0 * std::f64::consts::PI * 200.0 * i as f64 / 48_000.0;
            x.push((0.3 * phase.sin() + 0.07 * lcg(&mut seed) + 0.02) as f32);
            y.push((0.3 * (phase + 0.19).sin() + 0.09 * lcg(&mut seed) - 0.01) as f32);
        }
        (x, y)
    }

    fn check_fixture(name: &str, x: &[f32], y: &[f32], reference: &[f32], rho: f64) {
        let mut mixer = AdaptiveXfade::new(x.len());
        let mut output = vec![0.0; x.len()];
        mixer.blend(x, y, &mut output);
        let max_error = output
            .iter()
            .zip(reference)
            .map(|(&actual, &expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        let measured_rho = estimate_correlation(x, y);
        assert!((measured_rho - rho).abs() < 1e-10, "rho mismatch: {name}");
        assert!(max_error < 1e-5, "reference mismatch: {name}: {max_error}");
        assert_eq!(output[0], x[0]);
        assert_eq!(output[x.len() - 1], y[x.len() - 1]);
        println!("xfade {name}: rho={measured_rho:.12}, max_abs_error={max_error:.9e}");
    }

    #[test]
    fn xfade_python_reference_lcg() {
        let (x, y) = inputs(fixture::N);
        check_fixture(
            "positive",
            &x,
            &y,
            &fixture::POSITIVE,
            fixture::RHO_POSITIVE,
        );
        let opposite: Vec<_> = x.iter().map(|&v| -v).collect();
        check_fixture(
            "negative",
            &x,
            &opposite,
            &fixture::NEGATIVE,
            fixture::RHO_NEGATIVE,
        );
        let tiny_x: Vec<_> = x.iter().map(|&v| (f64::from(v) * 1e-8) as f32).collect();
        let tiny_y: Vec<_> = y.iter().map(|&v| (f64::from(v) * 1e-8) as f32).collect();
        check_fixture(
            "near_zero",
            &tiny_x,
            &tiny_y,
            &fixture::NEAR_ZERO,
            fixture::RHO_NEAR_ZERO,
        );
    }

    #[test]
    fn xfade_rho_one_is_amplitude_complementary() {
        for i in 0..=1000 {
            let v = i as f64 / 1000.0;
            let (a, b) = gains(v, 1.0);
            assert!((a - (1.0 - v)).abs() < 1e-14);
            assert!((b - v).abs() < 1e-14);
            assert!((a + b - 1.0).abs() < 1e-14);
        }
        let (x, _) = inputs(257);
        let mut out = vec![0.0; x.len()];
        AdaptiveXfade::new(x.len()).blend(&x, &x, &mut out);
        assert!(out.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-7));
    }

    #[test]
    fn xfade_rho_zero_is_equal_power() {
        for i in 0..=1000 {
            let (a, b) = gains(i as f64 / 1000.0, 0.0);
            assert!((a * a + b * b - 1.0).abs() < 1e-14);
        }
        let x = [-1.0, 1.0, -1.0, 1.0, 0.0];
        let y = [-1.0, -1.0, 1.0, 1.0, 0.0];
        assert_eq!(estimate_correlation(&x, &y), 0.0);
        let mut mixer = AdaptiveXfade::new(x.len());
        let mut out = [0.0; 5];
        mixer.blend(&x, &y, &mut out);
        for i in 0..5 {
            let (a, b) = gains(mixer.fade_in[i], 0.0);
            let expected = (a * f64::from(x[i]) + b * f64::from(y[i])) as f32;
            assert!((out[i] - expected).abs() < 1e-7);
        }
    }

    #[test]
    fn xfade_negative_and_silence_safety() {
        let (a, b) = gains(0.5, -1.0);
        assert_eq!((a, b), (1.0, 1.0));
        let (a, b) = gains(0.5, -0.5);
        assert!((a * a + b * b - a * b - 1.0).abs() < 1e-14);
        assert_eq!(estimate_correlation(&[0.0; 4], &[0.0; 4]), 0.0);
        assert_eq!(estimate_correlation(&[0.25; 4], &[0.5; 4]), 0.0);
        let mut out = [1.0; 2];
        AdaptiveXfade::new(2).blend(&[0.0; 2], &[0.0; 2], &mut out);
        assert_eq!(out, [0.0; 2]);
    }

    #[test]
    #[should_panic(expected = "crossfade requires at least two samples")]
    fn xfade_rejects_short_fade() {
        let _ = AdaptiveXfade::new(1);
    }

    #[test]
    #[should_panic(expected = "output length mismatch")]
    fn xfade_rejects_length_mismatch() {
        AdaptiveXfade::new(2).blend(&[0.0; 2], &[0.0; 2], &mut [0.0; 1]);
    }

    #[test]
    fn xfade_release_benchmark() {
        let n = 2400;
        let (x, y) = inputs(n);
        let mut mixer = AdaptiveXfade::new(n);
        let mut output = vec![0.0; n];
        for _ in 0..100 {
            mixer.blend(black_box(&x), black_box(&y), black_box(&mut output));
        }
        let mut durations = Vec::with_capacity(5000);
        for _ in 0..5000 {
            let start = Instant::now();
            black_box(&mut mixer).blend(black_box(&x), black_box(&y), black_box(&mut output));
            durations.push(start.elapsed().as_nanos());
        }
        durations.sort_unstable();
        println!(
            "xfade n={n} release={}: median={:.3} us, p95={:.3} us, max={:.3} us, iterations={}",
            !cfg!(debug_assertions),
            durations[durations.len() / 2] as f64 / 1000.0,
            durations[durations.len() * 95 / 100] as f64 / 1000.0,
            durations[durations.len() - 1] as f64 / 1000.0,
            durations.len()
        );
        assert!(output.iter().all(|v| v.is_finite()));
    }
}
