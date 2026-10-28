//! Input gate. A block is converted only while it is loud enough above the rumble band *and*
//! Silero VAD hears a voice in it; hysteresis and a hold keep quiet syllables and phrase tails in.
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

pub struct Gate {
    denoiser: Denoiser,
    hpf: Biquad,
    vad: Option<SileroVad>,
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
}

impl Gate {
    /// `vad` None = level-only gate.
    pub fn new(block_ms: f32, vad: Option<SileroVad>) -> Self {
        Self {
            denoiser: Denoiser::new(),
            hpf: Biquad::highpass(RATE16, HPF_HZ),
            vad,
            hold_blocks: ((HOLD_MS / block_ms).ceil() as usize).max(1),
            hold_left: 0,
        }
    }

    /// RNNoise over the 48 kHz block, in place.
    pub fn denoise(&mut self, block48: &mut [f32]) {
        self.denoiser.process(block48);
    }

    /// Decides on the block's 16 kHz audio (after denoising, if any). `open_db` is the level
    /// threshold in dBFS, `vad` the voice threshold 0..1 (0 = level only).
    pub fn decide(&mut self, audio16: &[f32], open_db: f32, vad: f32) -> Result<GateReport> {
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
        if loud && speech {
            self.hold_left = self.hold_blocks;
        } else if was_open {
            self.hold_left -= 1;
        }
        Ok(GateReport { open: (loud && speech) || was_open, loud, speech, level, vad: prob, speech_ms })
    }

    /// Forget the hold, the level filter and the VAD's recurrent state (after warm-up).
    pub fn reset(&mut self) {
        self.hold_left = 0;
        self.hpf.reset();
        if let Some(v) = self.vad.as_mut() {
            v.reset();
        }
    }
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

    #[test]
    fn silence_stays_closed_and_the_hold_counts_down() {
        let mut g = Gate::new(240.0, None);
        let z = vec![0.0f32; BLOCK16];
        assert!(!g.decide(&z, -48.0, 0.0).unwrap().open);
        // a loud block opens, then two blocks of hold (400 ms / 240 ms)
        let mut seed = 7;
        let loud = noise(BLOCK16, 0.3, &mut seed);
        assert!(g.decide(&loud, -48.0, 0.0).unwrap().open);
        assert!(g.decide(&z, -48.0, 0.0).unwrap().open);
        assert!(g.decide(&z, -48.0, 0.0).unwrap().open);
        assert!(!g.decide(&z, -48.0, 0.0).unwrap().open);
    }

    #[test]
    fn loud_white_noise_is_not_speech() {
        let Some(vad) = silero() else { return };
        let mut g = Gate::new(240.0, Some(vad));
        let mut seed = 42;
        for _ in 0..8 {
            let b = noise(BLOCK16, 0.3, &mut seed);
            let r = g.decide(&b, -48.0, 0.5).unwrap();
            assert!(r.loud, "level test should pass: {r:?}");
            assert!(!r.open, "noise opened the gate: {r:?}");
        }
    }
}
