//! Synchronous overlap-add: find where the new render best continues the previous tail
//! (normalised cross-correlation over the search window), then crossfade: sin^2, Applio's
//! phase-vocoder blend (keeps the magnitude through the seam even when the two renders
//! disagree in phase), or the correlation-adaptive equal-power blend from `xfade`.

use std::sync::Arc;

use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};

use super::xfade::AdaptiveXfade;

pub struct Sola {
    crossfade: usize,
    search: usize,
    tail: Vec<f32>,
    fade_in: Vec<f32>,
    fade_out: Vec<f32>,
    /// ramp applied to the first render after silence (defaults to `fade_in`)
    onset_fade: Vec<f32>,
    /// crossfade samples still to drain after the input went silent
    draining: usize,
    last_offset: usize,
    vocoder: Option<Vocoder>,
    /// takes precedence over the vocoder when set
    adaptive: Option<AdaptiveXfade>,
}

struct Vocoder {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    buf_a: Vec<f32>,
    buf_b: Vec<f32>,
    spec_a: Vec<Complex32>,
    spec_b: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl Vocoder {
    fn new(fade_in: &[f32], fade_out: &[f32]) -> Self {
        let n = fade_in.len();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n);
        Self {
            window: fade_in.iter().zip(fade_out).map(|(i, o)| (i * o).sqrt()).collect(),
            buf_a: fft.make_input_vec(),
            buf_b: fft.make_input_vec(),
            spec_a: fft.make_output_vec(),
            spec_b: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            fft,
        }
    }

    /// Port of Applio's `phase_vocoder(a, b, fade_out, fade_in)`: time-domain sin^4 blend plus a
    /// resynthesis of each bin at the frequency implied by the phase difference between the two
    /// renders, so partials neither cancel nor comb-filter across the seam.
    fn blend(&mut self, a: &[f32], b: &[f32], fade_out: &[f32], fade_in: &[f32], out: &mut [f32]) {
        let n = a.len();
        for i in 0..n {
            self.buf_a[i] = a[i] * self.window[i];
            self.buf_b[i] = b[i] * self.window[i];
        }
        self.fft.process_with_scratch(&mut self.buf_a, &mut self.spec_a, &mut self.scratch).expect("fixed size");
        self.fft.process_with_scratch(&mut self.buf_b, &mut self.spec_b, &mut self.scratch).expect("fixed size");
        let mut acc = vec![0.0f64; n];
        let bins = n / 2 + 1;
        for k in 0..bins {
            let (fa, fb) = (self.spec_a[k], self.spec_b[k]);
            let mut amp = (fa.norm() + fb.norm()) as f64;
            // negative-frequency energy, except DC and (even n) Nyquist
            if k != 0 && !(n % 2 == 0 && k == n / 2) {
                amp *= 2.0;
            }
            if amp == 0.0 {
                continue;
            }
            let phi_a = fa.im.atan2(fa.re) as f64;
            let mut delta = fb.im.atan2(fb.re) as f64 - phi_a;
            delta -= std::f64::consts::TAU * (delta / std::f64::consts::TAU + 0.5).floor();
            let step = (std::f64::consts::TAU * k as f64 + delta) / n as f64;
            let (ds, dc) = step.sin_cos();
            let (mut zi, mut zr) = phi_a.sin_cos();
            for v in acc.iter_mut() {
                *v += amp * zr;
                let nr = zr * dc - zi * ds;
                zi = zr * ds + zi * dc;
                zr = nr;
            }
        }
        for i in 0..n {
            out[i] = a[i] * fade_out[i] * fade_out[i]
                + b[i] * fade_in[i] * fade_in[i]
                + (acc[i] / n as f64) as f32 * self.window[i];
        }
    }
}

impl Sola {
    pub fn new(crossfade: usize, search: usize, phase_vocoder: bool) -> Self {
        let fade_in: Vec<f32> = (0..crossfade)
            .map(|i| {
                let x = if crossfade > 1 { i as f64 / (crossfade - 1) as f64 } else { 1.0 };
                (0.5 * std::f64::consts::PI * x).sin().powi(2) as f32
            })
            .collect();
        let fade_out: Vec<f32> = fade_in.iter().map(|v| 1.0 - v).collect();
        let vocoder = (phase_vocoder && crossfade >= 16).then(|| Vocoder::new(&fade_in, &fade_out));
        Self { crossfade, search, tail: vec![0.0; crossfade], onset_fade: fade_in.clone(), fade_in, fade_out, draining: 0, last_offset: 0, vocoder, adaptive: None }
    }

    /// Length of the sin^2 fade-in at the first render after silence; it only has to hide the
    /// step from zero, so a few ms keep plosives at a phrase start intact.
    pub fn onset_fade(mut self, samples: usize) -> Self {
        let n = samples.max(1);
        self.onset_fade = (0..n)
            .map(|i| {
                let x = if n > 1 { i as f64 / (n - 1) as f64 } else { 1.0 };
                (0.5 * std::f64::consts::PI * x).sin().powi(2) as f32
            })
            .collect();
        self
    }

    /// Correlation-adaptive equal-power crossfade instead of the vocoder / sin^2 blend.
    pub fn adaptive(mut self, on: bool) -> Self {
        self.adaptive = (on && self.crossfade >= 2).then(|| AdaptiveXfade::new(self.crossfade));
        self
    }

    /// Similarity (cosine) between the previous render's tail and the start of `audio`, i.e. how
    /// alike two renders are where they get crossfaded; None right after silence.
    pub fn overlap_similarity(&self, audio: &[f32]) -> Option<f32> {
        if self.tail.iter().all(|&v| v == 0.0) || audio.len() < self.crossfade {
            return None;
        }
        super::cosine(&audio[..self.crossfade], &self.tail)
    }

    /// Offset chosen by the last `process` call, in samples.
    pub fn last_offset(&self) -> usize {
        self.last_offset
    }

    /// `audio` must hold at least block + crossfade + search samples.
    pub fn process(&mut self, audio: &[f32], block: usize) -> Vec<f32> {
        let cf = self.crossfade;
        assert!(audio.len() >= block + cf + self.search, "render too short for SOLA");
        let onset = self.tail.iter().all(|&v| v == 0.0);
        let offset = if onset {
            0
        } else {
            (0..=self.search)
                .map(|k| {
                    let seg = &audio[k..k + cf];
                    let num: f32 = seg.iter().zip(&self.tail).map(|(a, b)| a * b).sum();
                    let den = (seg.iter().map(|a| a * a).sum::<f32>() + 1e-8).sqrt();
                    (k, num / den)
                })
                .fold((0, f32::MIN), |best, cur| if cur.1 > best.1 { cur } else { best })
                .0
        };
        self.last_offset = offset;
        let aligned = &audio[offset..offset + block + cf];
        let mut out = aligned[..block].to_vec();
        let n = cf.min(block);
        if onset {
            for (o, w) in out.iter_mut().zip(&self.onset_fade) {
                *o *= w;
            }
        } else if let (Some(a), true) = (self.adaptive.as_mut(), n == cf) {
            a.blend(&self.tail, &aligned[..cf], &mut out[..cf]);
        } else if let (Some(v), true) = (self.vocoder.as_mut(), n == cf) {
            v.blend(&self.tail, &aligned[..cf], &self.fade_out, &self.fade_in, &mut out[..cf]);
        } else {
            for i in 0..n {
                out[i] = aligned[i] * self.fade_in[i] + self.tail[i] * self.fade_out[i];
            }
        }
        self.tail.copy_from_slice(&aligned[block..block + cf]);
        self.draining = cf;
        out
    }

    /// Gate closed: play out the remaining tail once with a fade, then silence.
    pub fn silence(&mut self, block: usize) -> Vec<f32> {
        let mut out = vec![0.0; block];
        if self.draining > 0 {
            let start = self.crossfade - self.draining;
            let n = block.min(self.draining);
            for i in 0..n {
                out[i] = self.tail[start + i] * self.fade_out[start + i];
            }
            self.draining -= n;
        }
        if self.draining == 0 {
            self.tail.iter_mut().for_each(|v| *v = 0.0);
        }
        out
    }
}
