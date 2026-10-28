//! RNNoise front end (the pure-Rust port `nnnoiseless`): 48 kHz, 10 ms frames. It takes
//! stationary noise, keyboard clicks and handling thumps out of the input before the voice
//! conversion sees them. Its own voice probability is not used: it calls broadband noise speech,
//! so the gate asks Silero VAD instead.

use nnnoiseless::DenoiseState;

/// 480 samples at 48 kHz; every block size the engine produces is a multiple of it.
pub const FRAME: usize = DenoiseState::FRAME_SIZE;
const PCM_SCALE: f32 = 32768.0;

pub struct Denoiser {
    state: Box<DenoiseState<'static>>,
    frame_in: Vec<f32>,
    frame_out: Vec<f32>,
}

impl Default for Denoiser {
    fn default() -> Self {
        Self::new()
    }
}

impl Denoiser {
    pub fn new() -> Self {
        let mut d = Self { state: DenoiseState::new(), frame_in: vec![0.0; FRAME], frame_out: vec![0.0; FRAME] };
        // the first output frame carries a fade-in artefact: spend it on silence
        d.frame(&mut vec![0.0; FRAME]);
        d
    }

    /// Denoises `audio` (−1..1) in place, whole 10 ms frames only; a trailing partial frame is
    /// left untouched rather than delayed.
    pub fn process(&mut self, audio: &mut [f32]) {
        for chunk in audio.chunks_exact_mut(FRAME) {
            self.frame(chunk);
        }
    }

    fn frame(&mut self, chunk: &mut [f32]) {
        for (d, &s) in self.frame_in.iter_mut().zip(chunk.iter()) {
            *d = s * PCM_SCALE;
        }
        self.state.process_frame(&mut self.frame_out, &self.frame_in);
        for (d, &s) in chunk.iter_mut().zip(self.frame_out.iter()) {
            *d = s / PCM_SCALE;
        }
    }
}
