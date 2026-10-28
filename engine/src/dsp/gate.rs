//! Input gate. A block is converted only while it is loud enough above the rumble band *and*
//! Silero VAD hears a voice in it (or, when asked, the block holds a steady pitch: sustained
//! vowels and glides, which Silero often refuses); hysteresis and a hold keep quiet syllables and
//! phrase tails in.
//! Loud non-speech (rubbing the headset, a phone put down on the desk) fails the second test.
//! RNNoise, which also lives here, cleans a copy of the 48 kHz input for these tests (and the audio
//! that gets converted only when asked); its own voice probability is not used, it calls broadband
//! noise speech.

use anyhow::Result;

use super::biquad::Biquad;
use super::denoise::Denoiser;
use super::vad::{CHUNK_MS, SileroVad};

/// After the input stops qualifying, keep converting this long: the lookahead zone of the last
/// voiced block and the natural decay of the phrase are still waiting to be rendered.
pub const HOLD_MS: f32 = 400.0;
/// Speech evidence one block needs before the gate opens: one VAD chunk. The chunk that straddles
/// the block end (up to 24 ms) is scored with the next block; together with the chunk itself that
/// stays inside the engine's 60 ms lookahead, and the context is kept across closed blocks, so an
/// onset that arrived late in a closed block is still rendered by the next open one.
pub const MIN_SPEECH_MS: f32 = CHUNK_MS;
/// an open gate accepts this much less voice probability before it starts closing
pub const VAD_HYSTERESIS: f32 = 0.15;
/// side-chain high-pass for the level test
pub const HPF_HZ: f32 = 150.0;
/// an open gate closes this far (x0.5 = 6 dB) below its opening level
const LEVEL_HYSTERESIS: f32 = 0.5;
const RATE16: f32 = 16_000.0;
/// Steady-pitch test: YIN over 40 ms windows every 20 ms of the gate copy. A sustained vowel, a
/// glide or humming stays periodic; noise, keys and bumps do not.
const YIN_WINDOW: usize = 640;
const YIN_HOP: usize = 320;
/// pitch periods searched: 1000 Hz .. 60 Hz
const TAU_MIN: usize = 16;
const TAU_MAX: usize = 267;
/// Strict on purpose: held vowels score 0.001-0.01, while a 150 Hz rattle in the noise probe that
/// fooled 0.15 for 80 ms scored 0.04-0.08.
const YIN_THRESHOLD: f32 = 0.05;
/// consecutive periodic windows whose pitch moves less than this per hop form one steady run
const STEADY_CENTS: f32 = 100.0;
/// a steady run this long (five windows) counts as voice
pub const MIN_PITCH_MS: f32 = 120.0;

pub struct Gate {
    denoiser: Denoiser,
    hpf: Biquad,
    vad: Option<SileroVad>,
    pitch: PitchRun,
    hold_blocks: usize,
    hold_left: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GateReport {
    /// convert this block
    pub open: bool,
    /// level test passed (with hysteresis while open)
    pub loud: bool,
    /// voice test passed (always, without a VAD model)
    pub speech: bool,
    /// RMS of the 16 kHz block above `HPF_HZ`
    pub level: f32,
    /// highest VAD chunk probability in the block (0 without a model)
    pub vad: f32,
    /// milliseconds of the block at or above the voice threshold in force
    pub speech_ms: f32,
    /// longest steady-pitch run reaching into the block, ms (0 unless the pitch test is on)
    pub pitch_ms: f32,
    /// opened by the pitch test alone: Silero would have kept the block out
    pub by_pitch: bool,
}

impl Gate {
    /// `vad` None = level-only gate.
    pub fn new(block_ms: f32, vad: Option<SileroVad>) -> Self {
        Self {
            denoiser: Denoiser::new(),
            hpf: Biquad::highpass(RATE16, HPF_HZ),
            vad,
            pitch: PitchRun::new(),
            hold_blocks: ((HOLD_MS / block_ms).ceil() as usize).max(1),
            hold_left: 0,
        }
    }

    /// RNNoise over the 48 kHz block, in place.
    pub fn denoise(&mut self, block48: &mut [f32]) {
        self.denoiser.process(block48);
    }

    /// Decides on the block's 16 kHz audio (after denoising, if any). `open_db` is the level
    /// threshold in dBFS, `vad` the voice threshold 0..1 (0 = level only); `pitch` also lets a
    /// steady pitch count as voice.
    pub fn decide(&mut self, audio16: &[f32], open_db: f32, vad: f32, pitch: bool) -> Result<GateReport> {
        let level = self.hpf.rms(audio16);
        let was_open = self.hold_left > 0;
        let open_at = 10f32.powf(open_db / 20.0);
        let loud = level >= if was_open { open_at * LEVEL_HYSTERESIS } else { open_at };
        let (prob, speech_ms, speech) = match self.vad.as_mut() {
            Some(v) if vad > 0.0 => {
                v.push(audio16)?;
                let th = if was_open { (vad - VAD_HYSTERESIS).max(0.0) } else { vad };
                let ms = v.speech_ms(th);
                (v.max(), ms, ms >= MIN_SPEECH_MS)
            }
            Some(v) => {
                // keep the recurrent state warm so switching the test back on works at once
                v.push(audio16)?;
                (v.max(), 0.0, true)
            }
            None => (0.0, 0.0, true),
        };
        let pitch_ms = if pitch {
            self.pitch.push(audio16)
        } else {
            self.pitch.reset();
            0.0
        };
        let by_pitch = !speech && pitch_ms >= MIN_PITCH_MS;
        let speech = speech || by_pitch;
        if loud && speech {
            self.hold_left = self.hold_blocks;
        } else if was_open {
            self.hold_left -= 1;
        }
        Ok(GateReport { open: (loud && speech) || was_open, loud, speech, level, vad: prob, speech_ms, pitch_ms, by_pitch: loud && by_pitch })
    }

    /// Forget the hold, the level filter and the VAD's recurrent state (after warm-up).
    pub fn reset(&mut self) {
        self.hold_left = 0;
        self.hpf.reset();
        self.pitch.reset();
        if let Some(v) = self.vad.as_mut() {
            v.reset();
        }
    }
}

/// Running YIN analysis of the gate copy on a continuous 20 ms grid, for the steady-pitch test.
struct PitchRun {
    buf: Vec<f32>,
    /// windows in the current steady run, and the pitch period of its last window
    run: usize,
    last_tau: f32,
}

impl PitchRun {
    fn new() -> Self {
        Self { buf: Vec::with_capacity(8192), run: 0, last_tau: 0.0 }
    }

    /// Feed a block; returns the longest steady run reaching into it, in ms of audio.
    fn push(&mut self, audio16: &[f32]) -> f32 {
        self.buf.extend_from_slice(audio16);
        let (need, mut start, mut best) = (YIN_WINDOW + TAU_MAX, 0, 0);
        while start + need <= self.buf.len() {
            let tau = yin_period(&self.buf[start..start + need]);
            self.run = match tau {
                Some(t) if self.run > 0 && (1200.0 * (t / self.last_tau).log2()).abs() < STEADY_CENTS => self.run + 1,
                Some(_) => 1,
                None => 0,
            };
            if let Some(t) = tau {
                self.last_tau = t;
            }
            best = best.max(self.run);
            start += YIN_HOP;
        }
        self.buf.drain(..start);
        if best == 0 { 0.0 } else { (YIN_WINDOW + (best - 1) * YIN_HOP) as f32 / RATE16 * 1000.0 }
    }

    fn reset(&mut self) {
        self.buf.clear();
        self.run = 0;
    }
}

/// YIN pitch period in samples (1000..60 Hz), parabolic-interpolated; None when not periodic.
/// `x` holds YIN_WINDOW + TAU_MAX samples.
fn yin_period(x: &[f32]) -> Option<f32> {
    let mut cmnd = [1.0f32; TAU_MAX + 1];
    let mut sum = 0.0f32;
    for tau in 1..=TAU_MAX {
        let d: f32 = x[..YIN_WINDOW].iter().zip(&x[tau..tau + YIN_WINDOW]).map(|(a, b)| (a - b) * (a - b)).sum();
        sum += d;
        cmnd[tau] = if sum > 0.0 { d * tau as f32 / sum } else { 1.0 };
    }
    let mut tau = (TAU_MIN..=TAU_MAX).find(|&t| cmnd[t] < YIN_THRESHOLD)?;
    while tau < TAU_MAX && cmnd[tau + 1] < cmnd[tau] {
        tau += 1;
    }
    let (a, b) = (cmnd[tau - 1], cmnd[tau]);
    let c = if tau < TAU_MAX { cmnd[tau + 1] } else { b };
    let den = a - 2.0 * b + c;
    Some(if den.abs() > 1e-9 { tau as f32 + 0.5 * (a - c) / den } else { tau as f32 })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::models::{Device, ort_env};

    const BLOCK16: usize = 3840; // 240 ms

    fn noise(len: usize, amp: f32, seed: &mut u32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * amp
            })
            .collect()
    }

    fn silero() -> Option<SileroVad> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let p = root.join("models").join("silero_vad.onnx");
        if !p.exists() {
            eprintln!("no {} - skipping", p.display());
            return None;
        }
        let env = ort_env(&root.join("runtime"), Device::Cpu).expect("ort env");
        Some(SileroVad::load(env, &p).expect("silero vad"))
    }

    /// Eight harmonics on a pitch track `f0(t)` (Hz, t in seconds), 16 kHz.
    fn tone(len: usize, f0: impl Fn(f32) -> f32, amp: f32) -> Vec<f32> {
        let mut phase = 0.0f32;
        (0..len)
            .map(|i| {
                phase += std::f32::consts::TAU * f0(i as f32 / RATE16) / RATE16;
                amp * (1..=8).map(|k| (k as f32 * phase).sin() / k as f32).sum::<f32>() / 2.0
            })
            .collect()
    }

    #[test]
    fn yin_finds_a_harmonic_tone_and_not_noise() {
        let tau = yin_period(&tone(YIN_WINDOW + TAU_MAX, |_| 200.0, 0.3)).expect("a 200 Hz tone is periodic");
        assert!((RATE16 / tau - 200.0).abs() < 2.0, "{} Hz", RATE16 / tau);
        let mut seed = 3;
        assert!(yin_period(&noise(YIN_WINDOW + TAU_MAX, 0.3, &mut seed)).is_none());
    }

    #[test]
    fn a_glide_is_one_steady_run_and_noise_is_none() {
        let mut p = PitchRun::new();
        let glide = tone(BLOCK16 * 4, |t| 150.0 * 2f32.powf(t), 0.3); // an octave a second
        let ms: Vec<f32> = glide.chunks(BLOCK16).map(|b| p.push(b)).collect();
        assert!(ms.iter().all(|&m| m >= MIN_PITCH_MS) && ms.windows(2).all(|w| w[1] > w[0]), "{ms:?}");
        let mut p = PitchRun::new();
        let mut seed = 9;
        assert_eq!(p.push(&noise(BLOCK16, 0.3, &mut seed)), 0.0);
    }

    #[test]
    fn silence_stays_closed_and_the_hold_counts_down() {
        let mut g = Gate::new(240.0, None);
        let z = vec![0.0f32; BLOCK16];
        assert!(!g.decide(&z, -48.0, 0.0, false).unwrap().open);
        // a loud block opens, then two blocks of hold (400 ms / 240 ms)
        let mut seed = 7;
        let loud = noise(BLOCK16, 0.3, &mut seed);
        assert!(g.decide(&loud, -48.0, 0.0, false).unwrap().open);
        assert!(g.decide(&z, -48.0, 0.0, false).unwrap().open);
        assert!(g.decide(&z, -48.0, 0.0, false).unwrap().open);
        assert!(!g.decide(&z, -48.0, 0.0, false).unwrap().open);
    }

    #[test]
    fn loud_white_noise_is_not_speech() {
        let Some(vad) = silero() else { return };
        let mut g = Gate::new(240.0, Some(vad));
        let mut seed = 42;
        for _ in 0..8 {
            let b = noise(BLOCK16, 0.3, &mut seed);
            let r = g.decide(&b, -48.0, 0.5, true).unwrap();
            assert!(r.loud, "level test should pass: {r:?}");
            assert_eq!(r.pitch_ms, 0.0, "noise has no steady pitch: {r:?}");
            assert!(!r.open, "noise opened the gate: {r:?}");
        }
    }
}
