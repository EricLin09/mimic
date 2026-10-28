pub mod biquad;
pub mod denoise;
pub mod f0;
pub mod gate;
pub mod mel;
pub mod noise;
pub mod resample;
pub mod sola;
pub mod vad;
pub mod xfade;

/// Root-mean-square level.
pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

/// Cosine similarity over the length the two slices share; None when either side is (near) silent.
pub fn cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    (na > 1e-4 && nb > 1e-4).then(|| dot / (na * nb))
}

/// Nearest-rank percentile of an ascending slice (0 when empty).
pub fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f32 * p).round() as usize]
}
