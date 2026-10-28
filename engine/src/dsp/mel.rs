//! RMVPE front end: torch.stft(center=True, reflect pad, periodic Hann 1024, hop 160) ->
//! magnitude -> librosa mel basis (exported as rmvpe_mel.f32) -> ln(max(x, 1e-5)).

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};

pub const N_FFT: usize = 1024;
pub const HOP: usize = 160;
pub const N_MELS: usize = 128;
const BINS: usize = N_FFT / 2 + 1;

pub struct MelExtractor {
    basis: Vec<f32>,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    frame: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl MelExtractor {
    pub fn new(basis_file: &Path) -> Result<Self> {
        let bytes = std::fs::read(basis_file).with_context(|| format!("reading {}", basis_file.display()))?;
        ensure!(bytes.len() == N_MELS * BINS * 4, "unexpected mel basis size {}", bytes.len());
        let basis = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let window = (0..N_FFT)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32)
            .collect();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        Ok(Self {
            basis,
            window,
            frame: fft.make_input_vec(),
            spectrum: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            fft,
        })
    }

    /// Log-mel of `audio` (16 kHz); returns ([mel][frame] row-major, frames).
    pub fn log_mel(&mut self, audio: &[f32]) -> (Vec<f32>, usize) {
        let pad = N_FFT / 2;
        let n = audio.len();
        assert!(n > pad, "need more than {pad} samples");
        let reflect = |i: isize| -> f32 {
            let idx = if i < 0 { -i } else if i >= n as isize { 2 * (n as isize - 1) - i } else { i };
            audio[idx as usize]
        };
        let frames = n / HOP + 1;
        let mut mel = vec![0.0f32; N_MELS * frames];
        let mut mag = [0.0f32; BINS];
        for t in 0..frames {
            let start = (t * HOP) as isize - pad as isize;
            for (i, (v, w)) in self.frame.iter_mut().zip(&self.window).enumerate() {
                *v = reflect(start + i as isize) * w;
            }
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch)
                .expect("fft sizes are fixed");
            for (m, c) in mag.iter_mut().zip(&self.spectrum) {
                *m = (c.re * c.re + c.im * c.im).sqrt();
            }
            for m in 0..N_MELS {
                let row = &self.basis[m * BINS..(m + 1) * BINS];
                let e: f32 = row.iter().zip(&mag).map(|(a, b)| a * b).sum();
                mel[m * frames + t] = e.max(1e-5).ln();
            }
        }
        (mel, frames)
    }
}

/// Reflect-pad the frame axis up to a multiple of 32, as RMVPE's mel2hidden does.
pub fn pad_frames_to_32(mel: &[f32], frames: usize) -> (Vec<f32>, usize) {
    let padded = frames.div_ceil(32) * 32;
    if padded == frames {
        return (mel.to_vec(), frames);
    }
    let mut out = vec![0.0f32; N_MELS * padded];
    for m in 0..N_MELS {
        let src = &mel[m * frames..(m + 1) * frames];
        let dst = &mut out[m * padded..(m + 1) * padded];
        dst[..frames].copy_from_slice(src);
        for i in frames..padded {
            dst[i] = src[2 * (frames - 1) - i];
        }
    }
    (out, padded)
}
