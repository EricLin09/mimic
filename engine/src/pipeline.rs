//! Block-by-block conversion, ported from Applio's realtime path
//! (rvc/realtime/core.py `Realtime`/`VoiceChanger`, pipeline.py `_voice_conversion`).

use std::time::Instant;

use anyhow::{Result, ensure};

use crate::dsp::gate::Gate;
use crate::dsp::mel::{self, MelExtractor};
use crate::dsp::resample::{Resampler, StreamResampler};
use crate::dsp::sola::Sola;
use crate::dsp::{cosine, f0};
use crate::models::{Device, FEAT_DIM, FixedDims, Models, Paths};

pub const DEVICE_RATE: u32 = 48_000;
/// Speaking pitch assumed for the speaker when centring on the voice: a typical adult male median.
/// With spread 1 it only decides where `pitch` 0 lands; the slider moves from there.
pub const SPEAKER_F0_HZ: f32 = 120.0;
/// hop at 16 kHz: the pitch / synthesis grid runs at 100 fps
const WINDOW: usize = 160;
/// Maps the speaker's pitch range onto the voice's in the log domain:
/// f -> out_median * (f / in_median)^spread. spread 1 keeps the speaker's intonation width,
/// below 1 flattens it, above 1 widens it; the semitone `pitch` shift is applied on top.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F0Map {
    pub in_median: f32,
    pub out_median: f32,
    pub spread: f32,
}

impl F0Map {
    pub fn apply(&self, f0: f32) -> f32 {
        if f0 > 0.0 { self.out_median * (f0 / self.in_median).powf(self.spread) } else { 0.0 }
    }
}

#[derive(Clone, Debug)]
pub struct Params {
    pub pitch: f32,
    pub index_rate: f32,
    pub protect: f32,
    /// input gate in dBFS (block RMS above 150 Hz, after denoising): below it, with 6 dB of
    /// hysteresis and a short hold, the block is not converted at all
    pub silence_db: f32,
    /// RNNoise also cleans the audio that gets converted. Off by default: it rewrites clean speech
    /// too and only wins in very loud steady noise (docs/denoise-eval.md).
    pub denoise_audio: bool,
    /// The gate judges an RNNoise-cleaned copy of the input (level and Silero), so fans and keys
    /// do not open it; `silence_db` is calibrated on that copy.
    pub denoise_gate: bool,
    /// voice threshold 0..1 on Silero VAD's 32 ms chunk probability: the gate only opens, and
    /// only stays open (0.15 lower), while part of the block scores above it, so loud non-speech
    /// (rubbing the headset, a phone put down on the desk) is not converted. 0 = level only.
    pub vad: f32,
    /// Follow the speaker's loudness contour (not their absolute level): 0 keeps the model's own
    /// dynamics, 0.5 takes half of each 10 ms deviation from the long-term level difference.
    pub envelope: f32,
    /// pitch-range mapping onto the voice; wins over `align_pitch`
    pub f0_map: Option<F0Map>,
    /// Centre the speaker on the voice's median f0 (its f0_stats.json), assuming the speaker sits at
    /// SPEAKER_F0_HZ, so `pitch` 0 lands on the voice's centre. No effect without the stats file.
    pub align_pitch: bool,
}

impl Default for Params {
    fn default() -> Self {
        // pitch 0 = centred on the voice's own median f0 (docs/f0-stats.md); envelope 0.5 brings the
        // realtime output's loudness tracking up to the offline render's (both from opt-round-1)
        Self {
            pitch: 0.0, index_rate: 0.75, protect: 0.33, silence_db: -48.0, denoise_audio: false, denoise_gate: true, vad: 0.5, envelope: 0.5,
            f0_map: None, align_pitch: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub block_ms: f32,
    pub extra_s: f32,
    pub crossfade_s: f32,
    pub sola_ms: f32,
    /// Seamless rendering (needs synth_det.onnx): noise keyed to absolute time, continuous NSF
    /// phase, flow over the whole window and a decoder warm-up margin. Off = Applio's behaviour.
    pub seamless: bool,
    /// frames decoded ahead of the kept audio and thrown away (decoder left context)
    pub margin_frames: usize,
    /// Applio's phase-vocoder crossfade instead of a plain sin^2 blend
    pub phase_vocoder: bool,
    /// Correlation-adaptive equal-power crossfade (dsp::xfade); wins over `phase_vocoder`.
    /// Off by default until the opt-round-1 comparison says otherwise.
    pub adaptive_xfade: bool,
    /// Output frames this many 10 ms frames behind the newest audio, so the encoders see some
    /// future context for what gets rendered (and crossfaded). Adds that much latency.
    pub lookahead_frames: usize,
    /// Fade-in at the first render after silence. 50 ms (the crossfade) is Applio's behaviour;
    /// a few ms are enough to hide the step from zero and leave phrase-initial plosives alone.
    pub onset_fade_ms: f32,
    /// 3-point median over voiced f0 runs before the frames are frozen (one-frame octave errors).
    pub f0_median: bool,
    /// Extra left context for RMVPE, rounded up with its window to 32 mel frames. Default 80 ms:
    /// on the 240 ms layout the window grows from 32 to 64 frames, so every frame is rewritten once
    /// more before it is frozen (opt-round-1). Does not add output lookahead or latency.
    pub f0_context_ms: usize,
}

impl Default for Layout {
    fn default() -> Self {
        // Listening tests (2026-10-27): 240 ms blocks with 2.5 s context and a 60 ms lookahead
        // scored 4.03 on the distance-to-offline metric, better than Applio's 250 ms (4.14); the
        // lookahead settles contentvec features and RMVPE pitch; seamless renders are
        // time-aligned, so SOLA's search only adds jitter and is off.
        Self {
            block_ms: 240.0, extra_s: 2.5, crossfade_s: 0.05, sola_ms: 0.0,
            seamless: true, margin_frames: 12, phase_vocoder: true, adaptive_xfade: false, lookahead_frames: 6,
            onset_fade_ms: 50.0, f0_median: false, f0_context_ms: 80,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BlockStats {
    pub silent: bool,
    /// highest Silero VAD chunk probability in the block (0 without the model)
    pub vad: f32,
    /// milliseconds of the block at or above the voice threshold in force
    pub speech_ms: f32,
    /// RMS of the new 16 kHz input above 150 Hz, after denoising
    pub level: f32,
    /// median voiced pitch of the new frames, before / after the shift (0 = unvoiced)
    pub f0_in: f32,
    pub f0_out: f32,
    pub ms_pitch: f32,
    pub ms_features: f32,
    pub ms_synth: f32,
    pub ms_total: f32,
    /// cosine similarity of the two renders over the crossfade (1.0 = seamless)
    pub overlap: Option<f32>,
    /// SOLA alignment offset in samples
    pub sola_offset: usize,
    /// previous raw render vs this one over the time they share, at zero lag
    pub consistency: Option<f32>,
    /// same, at the best lag within +-10 ms: (cosine, lag in samples)
    pub consistency_best: Option<(f32, i32)>,
}

/// Every size a block works with, derived from the layout alone. `BlockProcessor::new` and the
/// fixed-shape session setup both take it from here, so the sessions' shapes cannot drift from what
/// the processor feeds them.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub seamless: bool,
    pub block48: usize,
    pub block16: usize,
    pub crossfade48: usize,
    pub sola48: usize,
    /// conversion window, 16 kHz samples
    pub convert16: usize,
    /// 100 fps frames in the window
    pub feature_frames: usize,
    pub skip_head: usize,
    pub lookahead: usize,
    pub margin: usize,
    /// Extra RMVPE left context, bounded by the full conversion window (16 kHz samples).
    pub f0_context16: usize,
}

impl Geometry {
    /// `seamless_available`: the voice has synth_det.onnx.
    pub fn new(layout: &Layout, seamless_available: bool) -> Self {
        let rate = DEVICE_RATE as f32;
        let seamless = layout.seamless && seamless_available;
        // Whole 10 ms steps keep the 48k / 16k / 100 fps grids aligned. Seamless renders also need
        // the block to be a multiple of contentvec's 320-sample stride (20 ms), otherwise the
        // feature grid shifts by half a frame every block and consecutive renders stop lining up.
        let step_ms = if seamless { 20.0 } else { 10.0 };
        let block48 = ((layout.block_ms / step_ms).round().max(1.0) as usize) * (step_ms as usize * 48);
        let crossfade48 = (layout.crossfade_s * rate) as usize;
        let sola48 = (layout.sola_ms / 1000.0 * rate) as usize;
        let extra48 = (layout.extra_s * rate) as usize;
        let (block16, crossfade16, sola16, extra16) = (block48 / 3, crossfade48 / 3, sola48 / 3, extra48 / 3);
        let mut convert16 = block16 + sola16 + extra16 + crossfade16;
        if convert16 % WINDOW != 0 {
            convert16 += WINDOW - convert16 % WINDOW;
        }
        // With the window length at 160 mod 320 the newest contentvec frame ends 80 samples before
        // the window end instead of 240, so the newest output frames get fresher features.
        if convert16 % 320 == 0 {
            convert16 += WINDOW;
        }
        let feature_frames = convert16 / WINDOW;
        let skip_head = extra16 / WINDOW;
        // lookahead only exists in seamless mode; keep the decoder margin in front of it
        let lookahead = if seamless { layout.lookahead_frames.min(skip_head.saturating_sub(layout.margin_frames)) } else { 0 };
        let margin = layout.margin_frames.min(skip_head - lookahead);
        let f0_context16 = layout.f0_context_ms.saturating_mul(16).min(convert16);
        Self { seamless, block48, block16, crossfade48, sola48, convert16, feature_frames, skip_head, lookahead, margin, f0_context16 }
    }

    /// Newest audio RMVPE sees each block, 16 kHz samples: whole 32-frame chunks.
    pub fn f0_window(&self) -> usize {
        let base = (self.block16 + 800).div_ceil(5120) * 5120 - 160;
        let requested = base + self.f0_context16 + 160;
        (requested.div_ceil(5120) * 5120 - 160).min(self.convert16)
    }

    /// The free dimensions of every graph for this geometry; `hop` is the voice's samples per frame.
    pub fn fixed_dims(&self, hop: usize) -> FixedDims {
        let named = |v: &[(&str, usize)]| v.iter().map(|&(k, n)| (k.to_string(), n as i64)).collect();
        // contentvec's CNN front end: 400-sample receptive field, stride 320
        let cv_frames = (self.convert16 - 400) / 320 + 1;
        let rmvpe_frames = (self.f0_window() / mel::HOP + 1).div_ceil(32) * 32;
        // retrieval runs on the duplicated-last-frame features from skip_head / 2 on
        let retrieval_frames = cv_frames + 1 - self.skip_head / 2;
        let decoder_start = self.skip_head - self.lookahead - self.margin;
        FixedDims {
            contentvec: named(&[("samples", self.convert16), ("frames", cv_frames), ("LayerNormalizationfeats_dim_0", 1)]),
            rmvpe: named(&[("frames", rmvpe_frames), ("Sigmoidsalience_dim_0", 1)]),
            retrieval: named(&[("frames", retrieval_frames)]),
            synth: named(&[("frames", self.feature_frames), ("Tanhaudio_dim_0", 1)]),
            synth_det: named(&[
                ("frames", self.feature_frames),
                ("samples", (self.feature_frames - decoder_start) * hop),
                ("Tanhaudio_dim_0", 1),
            ]),
        }
    }
}

pub struct BlockProcessor {
    models: Models,
    mel: MelExtractor,
    down: StreamResampler,
    /// the gate's denoised copy, with its own filter history
    gate_down: StreamResampler,
    up: Resampler,
    sola: Sola,
    pub params: Params,
    /// Fill `consistency` / `consistency_best` in the stats: keeps a copy of every render and
    /// runs a lag search against it, so only the offline tools switch it on.
    pub consistency_metrics: bool,
    block48: usize,
    block16: usize,
    feature_frames: usize,
    /// newest 16 kHz samples RMVPE sees each block (Geometry::f0_window)
    f0_window: usize,
    skip_head: usize,
    return_length: usize,
    convert: Vec<f32>,
    pitch: Vec<i64>,
    pitchf: Vec<f32>,
    seamless: bool,
    margin: usize,
    lookahead: usize,
    f0_median: bool,
    /// the voice's median voiced f0 (f0_stats.json), for `align_pitch`
    voice_f0: Option<f32>,
    env: EnvFollower,
    /// 100 fps frames ingested so far: absolute time for the noise streams
    frames_seen: u64,
    /// NSF phase (cycles) accumulated over frames that have left the pitch cache
    phase_base: f64,
    /// last raw render (48 kHz, before SOLA), for the consistency metric
    prev_render: Option<Vec<f32>>,
    gate: Gate,
}

fn median_voiced(f0: &[f32]) -> f32 {
    let mut v: Vec<f32> = f0.iter().copied().filter(|&f| f > 0.0).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// circular_write: drop the oldest `new.len()` items and append `new`.
fn push_back<T: Copy>(buf: &mut [T], new: &[T]) {
    let n = new.len().min(buf.len());
    buf.copy_within(n.., 0);
    let len = buf.len();
    buf[len - n..].copy_from_slice(&new[new.len() - n..]);
}

/// Nearest-neighbour x2 upsampling of [frames x 768], truncated to `out_frames`.
fn upsample2(feats: &[f32], out_frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(out_frames * FEAT_DIM);
    for i in 0..out_frames {
        let src = i / 2;
        out.extend_from_slice(&feats[src * FEAT_DIM..(src + 1) * FEAT_DIM]);
    }
    out
}

/// Envelope follower. The render keeps the model's own long-term level but takes the speaker's
/// short-term loudness contour: per 10 ms hop the level difference input - output (dB) is compared
/// with its slow average, and `mix` of the deviation is applied, within +-6 dB. Hops where either
/// side is near silence are left alone. Gains ramp linearly from hop to hop and carry over between
/// renders, so the follower adds no steps at seams.
struct EnvFollower {
    /// gain at the hop where the next render starts
    gain: f32,
    /// slow average of the input - output level difference over active hops, dB
    reference: Option<f32>,
}

impl EnvFollower {
    const FLOOR_DB: f32 = -60.0;
    const MAX_DB: f32 = 6.0;
    /// per active hop: about 2 s of speech
    const ALPHA: f32 = 1.0 / 200.0;

    fn new() -> Self {
        Self { gain: 1.0, reference: None }
    }

    /// `input16` starts at the same frame as `audio48`; `carry_hop` is where the next render starts.
    fn apply(&mut self, audio48: &mut [f32], input16: &[f32], mix: f32, carry_hop: usize) {
        const HOP48: usize = DEVICE_RATE as usize / 100;
        let hops = (audio48.len() / HOP48).min(input16.len() / WINDOW);
        if hops == 0 {
            return;
        }
        let db = |x: &[f32]| 20.0 * crate::dsp::rms(x).max(1e-9).log10();
        let diff: Vec<Option<f32>> = (0..hops)
            .map(|h| {
                let i = db(&input16[h * WINDOW..(h + 1) * WINDOW]);
                let o = db(&audio48[h * HOP48..(h + 1) * HOP48]);
                (i > Self::FLOOR_DB && o > Self::FLOOR_DB).then_some(i - o)
            })
            .collect();
        // only this block's own hops move the reference, so overlapping renders do not count twice
        for d in diff[..carry_hop.min(hops)].iter().flatten() {
            let r = self.reference.get_or_insert(*d);
            *r += (d - *r) * Self::ALPHA;
        }
        let Some(reference) = self.reference else { return };
        let gains: Vec<f32> = diff
            .iter()
            .map(|d| d.map_or(1.0, |d| 10f32.powf((mix * (d - reference)).clamp(-Self::MAX_DB, Self::MAX_DB) / 20.0)))
            .collect();
        let mut prev = self.gain;
        for (h, &g) in gains.iter().enumerate() {
            for (k, v) in audio48[h * HOP48..(h + 1) * HOP48].iter_mut().enumerate() {
                *v = (*v * (prev + (g - prev) * (k as f32 + 0.5) / HOP48 as f32)).clamp(-1.0, 1.0);
            }
            prev = g;
        }
        for v in &mut audio48[hops * HOP48..] {
            *v = (*v * prev).clamp(-1.0, 1.0);
        }
        self.gain = gains[carry_hop.clamp(1, hops) - 1];
    }
}

impl BlockProcessor {
    /// Loads the voice and builds the processor. `fixed_shapes` fixes every session's shapes to this
    /// layout, which spares DirectML a recompile per shape (block median 54 -> 41 ms in
    /// docs/fixed-dims-results.md); false keeps dynamic shapes, for A/B runs.
    pub fn load(paths: &Paths, device: Device, layout: &Layout, params: Params, fixed_shapes: bool) -> Result<Self> {
        let dims = if fixed_shapes {
            Geometry::new(layout, paths.has_seamless_synth()).fixed_dims(paths.voice_hop()?)
        } else {
            FixedDims::default()
        };
        let models = Models::load_fixed(paths, device, &dims)?;
        let mel = MelExtractor::new(&paths.mel_basis())?;
        Ok(Self::new(models, mel, layout, params))
    }

    pub fn new(mut models: Models, mel: MelExtractor, layout: &Layout, params: Params) -> Self {
        let rate = DEVICE_RATE as f32;
        let g = Geometry::new(layout, models.has_seamless_synth());
        let up = Resampler::new(models.sample_rate, DEVICE_RATE);
        Self {
            seamless: g.seamless,
            lookahead: g.lookahead,
            f0_median: layout.f0_median,
            voice_f0: models.f0_center_hz,
            env: EnvFollower::new(),
            margin: g.margin,
            // the zero-filled window at start-up covers absolute frames 0..feature_frames
            frames_seen: g.feature_frames as u64,
            prev_render: None,
            consistency_metrics: false,
            gate: Gate::new(layout.block_ms, models.vad.take()),
            phase_base: 0.0,
            models,
            mel,
            down: StreamResampler::new(DEVICE_RATE, 16_000),
            gate_down: StreamResampler::new(DEVICE_RATE, 16_000),
            up,
            sola: Sola::new(g.crossfade48, g.sola48, layout.phase_vocoder)
                .adaptive(layout.adaptive_xfade)
                .onset_fade((layout.onset_fade_ms / 1000.0 * rate) as usize),
            params,
            block48: g.block48,
            block16: g.block16,
            feature_frames: g.feature_frames,
            f0_window: g.f0_window(),
            skip_head: g.skip_head,
            return_length: g.feature_frames - g.skip_head,
            convert: vec![0.0; g.convert16],
            pitch: vec![0; g.feature_frames + 1],
            pitchf: vec![0.0; g.feature_frames + 1],
        }
    }

    /// Input/output block length at 48 kHz.
    pub fn block_size(&self) -> usize {
        self.block48
    }

    pub fn is_seamless(&self) -> bool {
        self.seamless
    }

    /// Extra output delay from the lookahead, in 48 kHz samples.
    pub fn lookahead_samples(&self) -> usize {
        self.lookahead * (DEVICE_RATE as usize / 100)
    }

    /// Seamless synthesis of window frames [head - margin, p_len), returning 48 kHz audio that
    /// starts at frame `head` and stops `lookahead` frames short of the window end (the warm-up
    /// margin and the lookahead tail are decoded, then dropped).
    fn render_seamless(&mut self, phone: &[f32], p_len: usize, head: usize) -> Result<Vec<f32>> {
        use crate::dsp::noise::{LATENT_STREAM, SOURCE_STREAM, gaussian};
        let start = head - self.margin;
        let first_abs = self.frames_seen - p_len as u64; // absolute frame of window frame 0
        let mut zp = vec![0.0f32; 192 * p_len];
        for c in 0..192 {
            for i in 0..p_len {
                zp[c * p_len + i] = gaussian(LATENT_STREAM, (first_abs + i as u64) * 192 + c as u64);
            }
        }
        let hop = self.models.hop;
        let s0 = (first_abs + start as u64) * hop as u64;
        let src: Vec<f32> = (0..(p_len - start) * hop).map(|k| gaussian(SOURCE_STREAM, s0 + k as u64)).collect();
        // NSF phase at the first decoded sample: frames already gone + window frames before `start`
        let per_frame = hop as f64 / self.models.sample_rate as f64;
        let cache_offset = self.pitchf.len() - p_len;
        let before: f64 = self.pitchf[..cache_offset + start].iter().map(|&f| f as f64 * per_frame).sum();
        let phase0 = (self.phase_base + before).fract() as f32;
        let pitch_p = &self.pitch[self.pitch.len() - p_len..];
        let pitchf_p = &self.pitchf[self.pitchf.len() - p_len..];
        let mut audio = self.models.synth_seamless(phone, pitch_p, pitchf_p, start, &zp, &src, phase0)?;
        audio.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
        let audio48 = self.up.process(&audio);
        // resample first so the resampler's zero-padded edges fall inside the discarded parts
        let per_frame = DEVICE_RATE as usize / 100;
        let drop = (self.margin * per_frame).min(audio48.len());
        let end = audio48.len().saturating_sub(self.lookahead * per_frame).max(drop);
        Ok(audio48[drop..end].to_vec())
    }

    /// Slides the pitch cache `shift` frames: the NSF phase of the frames leaving it is folded into
    /// `phase_base`; the newest `shift` slots keep stale values until the caller fills them.
    fn advance_pitch_cache(&mut self, shift: usize) {
        let per_frame = self.models.hop as f64 / self.models.sample_rate as f64;
        let leaving: f64 = self.pitchf[..shift].iter().map(|&f| f as f64 * per_frame).sum();
        self.phase_base = (self.phase_base + leaving).fract();
        self.pitch.copy_within(shift.., 0);
        self.pitchf.copy_within(shift.., 0);
    }

    pub fn process(&mut self, input: &[f32]) -> Result<(Vec<f32>, BlockStats)> {
        self.run_block(input, false)
    }

    /// Runs the whole chain on `input` whatever the gate says, so the GPU shaders and the VAD are
    /// warm before real audio arrives. Call `reset` when done.
    pub fn warm_up(&mut self, input: &[f32]) -> Result<BlockStats> {
        self.run_block(input, true).map(|(_, s)| s)
    }

    /// Forget the audio context, the pitch cache and the gate state.
    pub fn reset(&mut self) {
        self.convert.fill(0.0);
        self.pitch.fill(0);
        self.pitchf.fill(0.0);
        self.prev_render = None;
        self.env = EnvFollower::new();
        self.gate.reset();
    }

    fn run_block(&mut self, input: &[f32], force: bool) -> Result<(Vec<f32>, BlockStats)> {
        ensure!(input.len() == self.block48, "expected {} samples, got {}", self.block48, input.len());
        let start = Instant::now();
        // RNNoise cleans the 48 kHz block when denoising is on; the gate then judges the 16 kHz
        // audio the engine will see: level above the rumble band, and Silero's voice probability
        // RNNoise cleans the 48 kHz block for the gate (and for the conversion only with
        // `denoise_audio`); the gate then judges 16 kHz audio: level above the rumble band and
        // Silero's voice probability
        let (new16, gate16) = match (self.params.denoise_audio, self.params.denoise_gate) {
            (true, _) => {
                let mut block = input.to_vec();
                self.gate.denoise(&mut block);
                let a = self.down.process(&block);
                (a.clone(), a)
            }
            (false, true) => {
                let mut block = input.to_vec();
                self.gate.denoise(&mut block);
                (self.down.process(input), self.gate_down.process(&block))
            }
            (false, false) => {
                let a = self.down.process(input);
                (a.clone(), a)
            }
        };
        push_back(&mut self.convert, &new16);
        let shift = self.block16 / WINDOW;
        self.frames_seen += shift as u64;
        let gate = self.gate.decide(&gate16, self.params.silence_db, self.params.vad)?;
        let mut stats = BlockStats { level: gate.level, vad: gate.vad, speech_ms: gate.speech_ms, ..Default::default() };

        if !gate.open && !force {
            // Nothing is rendered, but the audio stays in the context: the lookahead zone of an
            // onset that arrived late in this block is rendered by the next open one. The pitch
            // cache advances in step (unvoiced) and the NSF phase keeps counting.
            self.advance_pitch_cache(shift);
            let (pl, fl) = (self.pitch.len(), self.pitchf.len());
            self.pitch[pl - shift..].fill(0);
            self.pitchf[fl - shift..].fill(0.0);
            self.prev_render = None;
            self.env.gain = 1.0;
            stats.silent = true;
            stats.ms_total = start.elapsed().as_secs_f32() * 1000.0;
            return Ok((self.sola.silence(self.block48), stats));
        }

        // pitch from the newest audio only, sized to whole 32-frame RMVPE chunks
        let t = Instant::now();
        let f0_frame = self.f0_window;
        let (mel_spec, frames) = self.mel.log_mel(&self.convert[self.convert.len() - f0_frame..]);
        let (mel_spec, padded) = mel::pad_frames_to_32(&mel_spec, frames);
        let salience = self.models.rmvpe(&mel_spec, padded)?;
        let f0_new = f0::decode(&salience, frames, 0.03);
        // boundary frames are unreliable: keep [3:-1]
        let mut interior: Vec<f32> = if f0_new.len() > 4 { f0_new[3..f0_new.len() - 1].to_vec() } else { f0_new };
        if self.f0_median {
            f0::median3_voiced(&mut interior);
        }
        let ratio = 2f32.powf(self.params.pitch / 12.0);
        let map = self.params.f0_map.or(match (self.params.align_pitch, self.voice_f0) {
            (true, Some(c)) => Some(F0Map { in_median: SPEAKER_F0_HZ, out_median: c, spread: 1.0 }),
            _ => None,
        });
        let shifted: Vec<f32> = interior.iter().map(|&f| map.map_or(f, |m| m.apply(f)) * ratio).collect();
        let coarse: Vec<i64> = shifted.iter().map(|&f| f0::coarse(f)).collect();
        self.advance_pitch_cache(shift);
        // Applio rewrites the newest 28 frames every block. A seamless render needs each frame's
        // pitch, and so the NSF phase, fixed once rendered: frames already output stay as they
        // were; the new frames and the lookahead zone take the latest estimate.
        let keep = if self.seamless { (shift + self.lookahead).min(coarse.len()) } else { coarse.len() };
        let (pl, fl) = (self.pitch.len(), self.pitchf.len());
        self.pitch[pl - keep..].copy_from_slice(&coarse[coarse.len() - keep..]);
        self.pitchf[fl - keep..].copy_from_slice(&shifted[shifted.len() - keep..]);
        stats.f0_in = median_voiced(&interior);
        stats.f0_out = median_voiced(&shifted);
        stats.ms_pitch = t.elapsed().as_secs_f32() * 1000.0;

        // content features, retrieval blend, x2 upsample, consonant protection
        let t = Instant::now();
        let (frames50, mut feats) = self.models.contentvec(&self.convert)?;
        feats.extend_from_within((frames50 - 1) * FEAT_DIM..frames50 * FEAT_DIM);
        let frames50 = frames50 + 1;
        let protect = self.params.protect;
        let feats0 = (protect < 0.5).then(|| feats.clone());
        let offset = self.skip_head / 2;
        if self.params.index_rate > 0.0 && offset < frames50 {
            let blended = self.models.retrieval(&feats[offset * FEAT_DIM..], frames50 - offset, self.params.index_rate)?;
            feats[offset * FEAT_DIM..].copy_from_slice(&blended);
        }
        let p_len = self.feature_frames;
        ensure!(2 * frames50 >= p_len, "contentvec gave {frames50} frames for {p_len} pitch frames");
        let mut phone = upsample2(&feats, p_len);
        let pitch_p = &self.pitch[self.pitch.len() - p_len..];
        let pitchf_p = &self.pitchf[self.pitchf.len() - p_len..];
        if let Some(feats0) = feats0 {
            let base = upsample2(&feats0, p_len);
            for (i, &f) in pitchf_p.iter().enumerate() {
                let w = if f > 0.0 { 1.0 } else { protect };
                let (dst, src) = (&mut phone[i * FEAT_DIM..(i + 1) * FEAT_DIM], &base[i * FEAT_DIM..(i + 1) * FEAT_DIM]);
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = *d * w + s * (1.0 - w);
                }
            }
        }
        stats.ms_features = t.elapsed().as_secs_f32() * 1000.0;

        let t = Instant::now();
        let head = if self.seamless {
            self.skip_head - self.lookahead
        } else {
            // Applio: synthesise only the frames after the context, same float32 rounding
            let keep = (self.return_length as f32 / p_len as f32) as f64;
            (p_len as f64 * (1.0 - keep)) as usize
        };
        let mut audio48 = if self.seamless {
            self.render_seamless(&phone, p_len, head)?
        } else {
            let mut audio = self.models.synth(&phone, pitch_p, pitchf_p, head)?;
            audio.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
            self.up.process(&audio)
        };
        stats.ms_synth = t.elapsed().as_secs_f32() * 1000.0;
        if self.params.envelope > 0.0 {
            // the render starts at window frame `head`: its input is the same frames at 16 kHz
            let input = self.convert[(head * WINDOW).min(self.convert.len())..].to_vec();
            self.env.apply(&mut audio48, &input, self.params.envelope, self.block16 / WINDOW);
        }

        if self.consistency_metrics {
            // consecutive renders share (render - block) samples of absolute time
            if let Some(prev) = &self.prev_render {
                if prev.len() > self.block48 {
                    let shared = &prev[self.block48..];
                    stats.consistency = cosine(shared, &audio48);
                    let search = 480i32;
                    stats.consistency_best = (-search..=search)
                        .filter_map(|lag| {
                            let (a, b) = if lag >= 0 {
                                (shared.get(lag as usize..)?, &audio48[..])
                            } else {
                                (shared, audio48.get((-lag) as usize..)?)
                            };
                            if a.len().min(b.len()) > 960 { cosine(a, b).map(|c| (c, lag)) } else { None }
                        })
                        .max_by(|x, y| x.0.total_cmp(&y.0));
                }
            }
            self.prev_render = Some(audio48.clone());
        }
        stats.overlap = self.sola.overlap_similarity(&audio48);
        let out = self.sola.process(&audio48, self.block48);
        stats.sola_offset = self.sola.last_offset();
        stats.ms_total = start.elapsed().as_secs_f32() * 1000.0;
        Ok((out, stats))
    }
}


#[cfg(test)]
mod geometry_tests {
    use super::*;

    #[test]
    fn rmvpe_zero_context_preserves_all_existing_layout_windows() {
        for seamless in [false, true] {
            for block_ms in [20.0, 120.0, 160.0, 200.0, 240.0, 320.0, 500.0] {
                let layout = Layout { block_ms, seamless, f0_context_ms: 0, ..Layout::default() };
                let g = Geometry::new(&layout, true);
                let original = (5120 * ((g.block16 + 800 - 1) / 5120 + 1) - 160).min(g.convert16);
                assert_eq!(g.f0_window(), original);
                assert_eq!(g.fixed_dims(400).rmvpe["frames"], ((original / mel::HOP + 1).div_ceil(32) * 32) as i64);
            }
        }
    }

    #[test]
    fn rmvpe_left_context_is_quantized_and_shared_with_fixed_session() {
        for (f0_context_ms, samples, frames) in [(0, 4960, 32), (80, 10080, 64), (160, 10080, 64), (320, 10080, 64)] {
            let g = Geometry::new(&Layout { f0_context_ms, ..Layout::default() }, true);
            assert_eq!(g.f0_window(), samples);
            assert_eq!(g.fixed_dims(400).rmvpe["frames"], frames);
        }
    }

    #[test]
    fn rmvpe_context_cannot_exceed_conversion_history() {
        let g = Geometry::new(&Layout { f0_context_ms: usize::MAX, ..Layout::default() }, true);
        assert_eq!(g.f0_window(), g.convert16);
        let actual_frames = g.f0_window() / mel::HOP + 1;
        assert_eq!(g.fixed_dims(400).rmvpe["frames"], (actual_frames.div_ceil(32) * 32) as i64);
    }
}
