//! Sinc resampler, a direct port of torchaudio's `Resample` (sinc_interp_hann,
//! lowpass_filter_width 6, rolloff 0.99). `Resampler` is stateless (zero-padded edges, like
//! Applio resampling each block on its own); `StreamResampler` carries the filter history
//! across blocks so a continuous stream has no edge error at block boundaries.

pub struct Resampler {
    orig: usize,
    new: usize,
    width: usize,
    taps: usize,
    /// `new` phases x `taps` coefficients
    kernel: Vec<f32>,
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

impl Resampler {
    pub fn new(orig_freq: u32, new_freq: u32) -> Self {
        Self::with_width(orig_freq, new_freq, 6)
    }

    /// `lowpass_width` sinc zero crossings each side (torchaudio's lowpass_filter_width): 6 is
    /// torchaudio's default and the realtime path's; Applio's training preprocess uses 128.
    pub fn with_width(orig_freq: u32, new_freq: u32, lowpass_width: u32) -> Self {
        const ROLLOFF: f64 = 0.99;
        let lowpass_width = lowpass_width as f64;
        let g = gcd(orig_freq as usize, new_freq as usize);
        let (orig, new) = (orig_freq as usize / g, new_freq as usize / g);
        let base_freq = orig.min(new) as f64 * ROLLOFF;
        let width = (lowpass_width * orig as f64 / base_freq).ceil() as usize;
        let taps = 2 * width + orig;
        let scale = base_freq / orig as f64;
        let mut kernel = Vec::with_capacity(new * taps);
        for phase in 0..new {
            for k in 0..taps {
                let idx = (k as f64 - width as f64) / orig as f64;
                let mut t = (-(phase as f64) / new as f64 + idx) * base_freq;
                t = t.clamp(-lowpass_width, lowpass_width);
                let window = (t * std::f64::consts::PI / lowpass_width / 2.0).cos().powi(2);
                let t = t * std::f64::consts::PI;
                let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
                kernel.push((sinc * window * scale) as f32);
            }
        }
        Self { orig, new, width, taps, kernel }
    }

    /// Output length is ceil(len * new / orig), like torchaudio.
    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        if self.orig == self.new {
            return input.to_vec();
        }
        let len = input.len();
        // zero padding: `width` before, `width + orig` after
        let mut padded = vec![0.0f32; self.width + len + self.width + self.orig];
        padded[self.width..self.width + len].copy_from_slice(input);
        let frames = (padded.len() - self.taps) / self.orig + 1;
        let target = (self.new * len).div_ceil(self.orig);
        let mut out = Vec::with_capacity(frames * self.new);
        for j in 0..frames {
            let window = &padded[j * self.orig..j * self.orig + self.taps];
            for phase in 0..self.new {
                let k = &self.kernel[phase * self.taps..(phase + 1) * self.taps];
                out.push(k.iter().zip(window).map(|(a, b)| a * b).sum());
            }
        }
        out.truncate(target);
        out
    }
}

/// Streaming variant: identical filter, but the last `taps - orig` input samples are kept
/// between calls, so block edges see real neighbours instead of zeros. Output is delayed by
/// `width` input samples (under 0.5 ms at 48 kHz); with block lengths that are multiples of
/// `orig` every call after the first returns exactly len * new / orig samples.
pub struct StreamResampler {
    r: Resampler,
    history: Vec<f32>,
}

impl StreamResampler {
    pub fn new(orig_freq: u32, new_freq: u32) -> Self {
        let r = Resampler::new(orig_freq, new_freq);
        let history = vec![0.0; r.taps.saturating_sub(r.orig)];
        Self { r, history }
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let r = &self.r;
        if r.orig == r.new {
            return input.to_vec();
        }
        let mut buf = Vec::with_capacity(self.history.len() + input.len());
        buf.extend_from_slice(&self.history);
        buf.extend_from_slice(input);
        let frames = if buf.len() >= r.taps { (buf.len() - r.taps) / r.orig + 1 } else { 0 };
        let mut out = Vec::with_capacity(frames * r.new);
        for j in 0..frames {
            let window = &buf[j * r.orig..j * r.orig + r.taps];
            for phase in 0..r.new {
                let k = &r.kernel[phase * r.taps..(phase + 1) * r.taps];
                out.push(k.iter().zip(window).map(|(a, b)| a * b).sum());
            }
        }
        self.history = buf[frames * r.orig..].to_vec();
        out
    }
}
