//! RMVPE salience decoding and RVC's coarse pitch quantisation.

pub const CLASSES: usize = 360;
const CENTS_BASE: f32 = 1997.379_4;
const F0_MIN: f32 = 50.0;
const F0_MAX: f32 = 1100.0;

/// Local weighted average around the arg-max bin (+-4), then cents -> Hz.
/// Frames whose peak salience is <= `threshold` are unvoiced (0 Hz).
pub fn decode(salience: &[f32], frames: usize, threshold: f32) -> Vec<f32> {
    (0..frames)
        .map(|t| {
            let row = &salience[t * CLASSES..(t + 1) * CLASSES];
            let (center, peak) = row
                .iter()
                .enumerate()
                .fold((0, f32::MIN), |best, (i, &v)| if v > best.1 { (i, v) } else { best });
            if peak <= threshold {
                return 0.0;
            }
            let (mut num, mut den) = (0.0f32, 0.0f32);
            for i in center.saturating_sub(4)..(center + 5).min(CLASSES) {
                num += row[i] * (CENTS_BASE + 20.0 * i as f32);
                den += row[i];
            }
            let cents = num / den;
            let f0 = 10.0 * 2f32.powf(cents / 1200.0);
            if f0 == 10.0 { 0.0 } else { f0 }
        })
        .collect()
}

/// 255-bucket mel-scale pitch used by the synthesizer's pitch embedding (1 = unvoiced).
pub fn coarse(f0: f32) -> i64 {
    let mel = 1127.0 * (1.0 + f0 / 700.0).ln();
    ((mel - F0_MIN) * 254.0 / (F0_MAX - F0_MIN) + 1.0).clamp(1.0, 255.0).round() as i64
}

/// 3-point median inside voiced runs: a frame whose two neighbours are voiced takes the median
/// of the three, which removes one-frame octave errors; voicing itself is never changed.
pub fn median3_voiced(f0: &mut [f32]) {
    if f0.len() < 3 {
        return;
    }
    let src = f0.to_vec();
    for i in 1..src.len() - 1 {
        let (a, b, c) = (src[i - 1], src[i], src[i + 1]);
        if a > 0.0 && b > 0.0 && c > 0.0 {
            let (lo, hi) = (a.min(b), a.max(b));
            f0[i] = hi.min(lo.max(c));
        }
    }
}
