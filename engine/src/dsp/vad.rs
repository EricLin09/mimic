//! Silero VAD (v5 ONNX graph, MIT) on the CPU: a speech probability every 32 ms at 16 kHz. It is
//! trained against exactly the sounds a level gate lets through, handling noise, bumps, clicks,
//! music, so the gate can ask "is this a voice" rather than "is this loud".

use std::path::Path;

use anyhow::{Result, bail};
use ort::environment::Environment;
use ort::value::Tensor;

use crate::models::cpu_session;

/// 32 ms at 16 kHz
pub const CHUNK: usize = 512;
pub const CHUNK_MS: f32 = 32.0;
/// samples of the previous chunk the v5 graph wants in front of the new one
const CONTEXT: usize = 64;
const STATE: usize = 2 * 128;

pub struct SileroVad {
    session: ort::session::Session,
    state: Vec<f32>,
    context: Vec<f32>,
    /// 16 kHz samples waiting for a whole chunk
    carry: Vec<f32>,
    /// probability of every chunk scored by the last `push`
    probs: Vec<f32>,
}

impl SileroVad {
    pub fn load(env: &Environment, path: &Path) -> Result<Self> {
        let session = cpu_session(env, path, 1)?;
        let names: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        for want in ["input", "state", "sr"] {
            if !names.contains(&want) {
                bail!("{} is not a Silero VAD v5 graph (inputs {:?})", path.display(), names);
            }
        }
        Ok(Self { session, state: vec![0.0; STATE], context: vec![0.0; CONTEXT], carry: Vec::new(), probs: Vec::new() })
    }

    /// Feeds 16 kHz audio. Every whole chunk is scored; the remainder (< 32 ms) waits for the
    /// next call, so `probs()` covers this call's audio up to that tail.
    pub fn push(&mut self, audio16: &[f32]) -> Result<()> {
        self.probs.clear();
        self.carry.extend_from_slice(audio16);
        let whole = self.carry.len() / CHUNK;
        let mut input = Vec::with_capacity(CONTEXT + CHUNK);
        for i in 0..whole {
            input.clear();
            input.extend_from_slice(&self.context);
            input.extend_from_slice(&self.carry[i * CHUNK..(i + 1) * CHUNK]);
            let x = Tensor::from_array(([1usize, CONTEXT + CHUNK], input.clone()))?;
            let st = Tensor::from_array(([2usize, 1, 128], self.state.clone()))?;
            let sr = Tensor::from_array(((), vec![16_000i64]))?;
            let out = self.session.run(ort::inputs!["input" => x, "state" => st, "sr" => sr])?;
            let (_, p) = out["output"].try_extract_tensor::<f32>()?;
            self.probs.push(p[0]);
            let (_, s) = out["stateN"].try_extract_tensor::<f32>()?;
            self.state.copy_from_slice(s);
            self.context.copy_from_slice(&input[input.len() - CONTEXT..]);
        }
        self.carry.drain(..whole * CHUNK);
        Ok(())
    }

    /// Highest chunk probability from the last `push` (0 when it scored no whole chunk).
    pub fn max(&self) -> f32 {
        self.probs.iter().copied().fold(0.0, f32::max)
    }

    /// Milliseconds of the last `push` whose chunks scored at least `threshold`.
    pub fn speech_ms(&self, threshold: f32) -> f32 {
        self.probs.iter().filter(|&&p| p >= threshold).count() as f32 * CHUNK_MS
    }

    /// Forget the recurrent state and any partial chunk.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.context.fill(0.0);
        self.carry.clear();
        self.probs.clear();
    }
}
