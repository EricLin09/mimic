//! Training data for a voice (`mimic prepare`): Applio's slicing and 40 kHz targets, with content
//! features and pitch from the engine's own contentvec / RMVPE graphs, so a model is trained on the
//! same feature extractor the realtime path runs. Writes, under the output folder:
//!
//!   audio/<name>.wav   40 kHz mono float32 slice (the training target)
//!   feats/<name>.npy   contentvec features, float32 [frames, 768], 50 fps
//!   f0/<name>.npy      RMVPE pitch in Hz (0 = unvoiced), float32 [frames], 100 fps
//!   retrieval.npy      every slice's features stacked, float32 [N, 768], for export_retrieval.py
//!   prepare.json       the slice list and the settings used
//!
//! plus one 3 s silent sample, `mute`, which the trainer mixes in like Applio's logs/mute.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;

use crate::dsp::f0;
use crate::dsp::mel::{self, MelExtractor};
use crate::dsp::resample::Resampler;
use crate::models::{Device, FEAT_DIM, FeatureNets};
use crate::wav;

/// Training sample rate: RVC v2 40k, 400 samples per 100 fps frame.
pub const TARGET_RATE: u32 = 40_000;
const FEATURE_RATE: u32 = 16_000;
/// Applio's preprocess resamples the dataset with torchaudio at lowpass_filter_width 128.
const TARGET_RESAMPLE_WIDTH: u32 = 128;
/// 40 kHz -> 16 kHz for the feature nets: the realtime path's 48 kHz -> 16 kHz filter (torchaudio's
/// default, 6 zero crossings), so training features see the same band limit as conversion.
const FEATURE_RESAMPLE_WIDTH: u32 = 6;
/// RMVPE's mel front end needs more than half an FFT window of 16 kHz audio.
const MIN_FEATURE_SAMPLES: usize = mel::N_FFT / 2 + 1;
const MUTE_SECONDS: usize = 3;

/// Silence-based slicer: a port of Applio's rvc/train/preprocess/slicer.py (openvpi's audio-slicer)
/// with the settings Applio's preprocess passes it. Cuts fall where Applio's do.
pub struct Slicer {
    threshold: f64,
    hop: usize,
    win: usize,
    min_length: usize,
    min_interval: usize,
    max_sil_kept: usize,
}

impl Slicer {
    /// threshold -42 dB, min length 1500 ms, min interval 400 ms, hop 15 ms, max silence kept 500 ms.
    pub fn applio(sr: u32) -> Self {
        Self::new(sr, -42.0, 1500.0, 400.0, 15.0, 500.0)
    }

    /// Times in milliseconds; rounding follows Python's round (ties to even), as the original does.
    pub fn new(sr: u32, threshold_db: f64, min_length: f64, min_interval: f64, hop_size: f64, max_sil_kept: f64) -> Self {
        let sr = sr as f64;
        let min_interval = sr * min_interval / 1000.0;
        let hop = (sr * hop_size / 1000.0).round_ties_even() as usize;
        Self {
            threshold: 10f64.powf(threshold_db / 20.0),
            hop,
            win: (min_interval.round_ties_even() as usize).min(4 * hop),
            min_length: (sr * min_length / 1000.0 / hop as f64).round_ties_even() as usize,
            min_interval: (min_interval / hop as f64).round_ties_even() as usize,
            max_sil_kept: (sr * max_sil_kept / 1000.0 / hop as f64).round_ties_even() as usize,
        }
    }

    /// RMS of `win`-sample frames every `hop`, zero-padded by win / 2 at both ends (get_rms).
    fn rms(&self, y: &[f32]) -> Vec<f64> {
        let half = (self.win / 2) as isize;
        (0..y.len() / self.hop + 1)
            .map(|i| {
                let start = (i * self.hop) as isize - half;
                let lo = start.max(0) as usize;
                let hi = ((start + self.win as isize).max(0) as usize).min(y.len());
                let energy: f64 = y.get(lo..hi).unwrap_or(&[]).iter().map(|&v| v as f64 * v as f64).sum();
                (energy / self.win as f64).sqrt()
            })
            .collect()
    }

    /// The non-silent stretches of `wave`, each keeping up to `max_sil_kept` of the silence around it.
    pub fn slice<'a>(&self, wave: &'a [f32]) -> Vec<&'a [f32]> {
        // the original compares the sample count with min_length in frames here; kept as it is
        if wave.len() <= self.min_length {
            return vec![wave];
        }
        let rms = self.rms(wave);
        // index of the first minimum of rms[a..b] (b clipped to the end, like a Python slice)
        let argmin = |a: usize, b: usize| (a..b.min(rms.len())).fold(a, |best, k| if rms[k] < rms[best] { k } else { best });
        let (m, keep) = (self.min_interval, self.max_sil_kept);
        let mut tags: Vec<(usize, usize)> = Vec::new();
        let (mut silence_start, mut clip_start): (Option<usize>, usize) = (None, 0);
        for (i, &level) in rms.iter().enumerate() {
            if level < self.threshold {
                silence_start.get_or_insert(i);
                continue;
            }
            let Some(s) = silence_start else { continue };
            let leading = s == 0 && i > keep;
            let middle = i - s >= m && i - clip_start >= self.min_length;
            if !leading && !middle {
                silence_start = None;
                continue;
            }
            if i - s <= keep {
                let pos = argmin(s, i + 1);
                tags.push(if s == 0 { (0, pos) } else { (pos, pos) });
                clip_start = pos;
            } else if i - s <= keep * 2 {
                let pos = argmin(i - keep, s + keep + 1);
                let pos_l = argmin(s, s + keep + 1);
                let pos_r = argmin(i - keep, i + 1);
                if s == 0 {
                    tags.push((0, pos_r));
                    clip_start = pos_r;
                } else {
                    tags.push((pos_l.min(pos), pos_r.max(pos)));
                    clip_start = pos_r.max(pos);
                }
            } else {
                let pos_l = argmin(s, s + keep + 1);
                let pos_r = argmin(i - keep, i + 1);
                tags.push(if s == 0 { (0, pos_r) } else { (pos_l, pos_r) });
                clip_start = pos_r;
            }
            silence_start = None;
        }
        let total = rms.len();
        if let Some(s) = silence_start {
            if total - s >= m {
                let end = total.min(s + keep);
                tags.push((argmin(s, end + 1), total + 1));
            }
        }
        if tags.is_empty() {
            return vec![wave];
        }
        let cut = |begin: usize, end: usize| {
            let (lo, hi) = ((begin * self.hop).min(wave.len()), (end * self.hop).min(wave.len()));
            &wave[lo.min(hi)..hi] // empty when the tags cross, as the Python slice would be
        };
        let mut chunks = Vec::new();
        if tags[0].0 > 0 {
            chunks.push(cut(0, tags[0].0));
        }
        for pair in tags.windows(2) {
            chunks.push(cut(pair[0].1, pair[1].0));
        }
        if tags[tags.len() - 1].1 < total {
            chunks.push(cut(tags[tags.len() - 1].1, total));
        }
        chunks
    }
}

/// Applio's "Automatic" cut of one slicer segment: 3 s pieces stepping 2.7 s, the last piece taking
/// the rest (up to 3.3 s). Same float expressions as rvc/train/preprocess/preprocess.py.
pub fn pieces(segment: &[f32], sr: u32) -> Vec<&[f32]> {
    const PERCENTAGE: f64 = 3.0;
    const OVERLAP: f64 = 0.3;
    let sr = sr as f64;
    let mut out = Vec::new();
    for i in 0.. {
        let start = ((sr * (PERCENTAGE - OVERLAP) * i as f64) as usize).min(segment.len());
        let rest = &segment[start..];
        if rest.len() as f64 > (PERCENTAGE + OVERLAP) * sr {
            out.push(&rest[..(PERCENTAGE * sr) as usize]);
        } else {
            out.push(rest);
            break;
        }
    }
    out
}

/// Gain that puts the RMS of the file's active frames (above the slicer threshold) at `target_db`,
/// lowered if needed so the peak stays at or below -0.5 dBFS.
fn loudness_gain(audio: &[f32], slicer: &Slicer, target_db: f32) -> f32 {
    let active: Vec<f64> = slicer.rms(audio).into_iter().filter(|&r| r >= slicer.threshold).collect();
    if active.is_empty() {
        return 1.0;
    }
    let level = (active.iter().map(|r| r * r).sum::<f64>() / active.len() as f64).sqrt();
    let peak = audio.iter().fold(0.0f32, |p, v| p.max(v.abs())).max(1e-9);
    let gain = 10f32.powf(target_db / 20.0) / level as f32;
    gain.min(10f32.powf(-0.5 / 20.0) / peak)
}

/// Writes float32 little-endian data as a NumPy .npy file (format 1.0, C order).
pub fn write_npy(path: &Path, data: &[f32], shape: &[usize]) -> Result<()> {
    ensure!(shape.iter().product::<usize>() == data.len(), "{}: shape {shape:?} for {} values", path.display(), data.len());
    let dims = match shape {
        [n] => format!("({n},)"),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {dims}, }}");
    // magic (6) + version (2) + length (2) + header + '\n' padded to a multiple of 64
    header.push_str(&" ".repeat((64 - (10 + header.len() + 1) % 64) % 64));
    header.push('\n');
    let mut bytes = Vec::with_capacity(10 + header.len() + 4 * data.len());
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

#[derive(Clone, Debug)]
pub struct PrepareOptions {
    pub device: Device,
    /// per-file loudness target (RMS of the active frames, dBFS); None leaves levels alone, as
    /// Applio's preprocess does by default
    pub loudness_db: Option<f32>,
}

#[derive(Serialize)]
struct Item {
    name: String,
    source: String,
    /// first sample within the source file's 40 kHz audio
    start: usize,
    samples: usize,
    feat_frames: usize,
    f0_frames: usize,
    voiced: f32,
}

#[derive(Serialize)]
struct Manifest {
    format: u32,
    sample_rate: u32,
    feature_rate: u32,
    source_dir: String,
    device: String,
    loudness_db: Option<f32>,
    target_resample_width: u32,
    feature_resample_width: u32,
    rmvpe_threshold: f32,
    source_seconds: f64,
    sliced_seconds: f64,
    retrieval_vectors: usize,
    mute: String,
    skipped: Vec<String>,
    items: Vec<Item>,
}

pub struct Summary {
    pub files: usize,
    pub slices: usize,
    pub source_seconds: f64,
    pub sliced_seconds: f64,
    pub retrieval_vectors: usize,
    pub skipped: usize,
}

/// Turns 40 kHz slices into training samples with the shared nets.
struct Extractor {
    nets: FeatureNets,
    mel: MelExtractor,
    down: Resampler,
}

impl Extractor {
    /// (features [frames x 768], frames, f0 per 10 ms) of one 40 kHz slice.
    fn run(&mut self, audio40: &[f32]) -> Result<(Vec<f32>, usize, Vec<f32>)> {
        let a16 = self.down.process(audio40);
        ensure!(a16.len() >= MIN_FEATURE_SAMPLES, "{} samples at 16 kHz is too short", a16.len());
        let (frames, feats) = self.nets.contentvec(&a16)?;
        Ok((feats, frames, self.f0(&a16)?))
    }

    /// RMVPE over the whole slice, as Applio's extraction does: frames = samples / 160 + 1.
    fn f0(&mut self, a16: &[f32]) -> Result<Vec<f32>> {
        let (mel_spec, frames) = self.mel.log_mel(a16);
        let (mel_spec, padded) = mel::pad_frames_to_32(&mel_spec, frames);
        let salience = self.nets.rmvpe(&mel_spec, padded)?;
        Ok(f0::decode(&salience, frames, RMVPE_THRESHOLD))
    }
}

/// Voicing threshold on RMVPE's salience, Applio's training extraction value.
const RMVPE_THRESHOLD: f32 = 0.03;

fn wav_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wav")))
        .collect();
    files.sort();
    Ok(files)
}

/// Slices every WAV in `wav_dir` (one speaker, top level only) and writes the training samples to
/// `out_dir`. `models` holds contentvec.onnx, rmvpe.onnx and rmvpe_mel.f32; `runtime` the ORT DLLs.
pub fn prepare(wav_dir: &Path, out_dir: &Path, models: &Path, runtime: &Path, opts: &PrepareOptions, mut log: impl FnMut(&str)) -> Result<Summary> {
    let files = wav_files(wav_dir)?;
    if files.is_empty() {
        bail!("no .wav files in {}", wav_dir.display());
    }
    for sub in ["audio", "feats", "f0"] {
        std::fs::create_dir_all(out_dir.join(sub)).with_context(|| format!("creating {}", out_dir.join(sub).display()))?;
    }
    let mut ex = Extractor {
        nets: FeatureNets::load(models, runtime, opts.device)?,
        mel: MelExtractor::new(&models.join("rmvpe_mel.f32"))?,
        down: Resampler::with_width(TARGET_RATE, FEATURE_RATE, FEATURE_RESAMPLE_WIDTH),
    };
    let slicer = Slicer::applio(TARGET_RATE);
    let mut resamplers: Vec<(u32, Resampler)> = Vec::new();
    let mut items = Vec::new();
    let mut skipped = Vec::new();
    let mut retrieval: Vec<f32> = Vec::new();
    let (mut source_seconds, mut sliced_seconds) = (0.0f64, 0.0f64);
    let started = Instant::now();

    for (idx0, path) in files.iter().enumerate() {
        let file_name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let (audio, rate) = wav::read_mono(path)?;
        source_seconds += audio.len() as f64 / rate as f64;
        let mut a40 = if rate == TARGET_RATE {
            audio
        } else {
            if !resamplers.iter().any(|(r, _)| *r == rate) {
                resamplers.push((rate, Resampler::with_width(rate, TARGET_RATE, TARGET_RESAMPLE_WIDTH)));
            }
            resamplers.iter().find(|(r, _)| *r == rate).expect("inserted above").1.process(&audio)
        };
        if let Some(target) = opts.loudness_db {
            let gain = loudness_gain(&a40, &slicer, target);
            a40.iter_mut().for_each(|v| *v *= gain);
        }
        let mut idx1 = 0;
        for segment in slicer.slice(&a40) {
            for piece in pieces(segment, TARGET_RATE) {
                let name = format!("{idx0}_{idx1}");
                idx1 += 1;
                // where the piece sits in the file, for traceability (it is a subslice of a40)
                let start = (piece.as_ptr() as usize - a40.as_ptr() as usize) / std::mem::size_of::<f32>();
                let (feats, frames, pitch) = match ex.run(piece) {
                    Ok(v) => v,
                    Err(e) => {
                        skipped.push(format!("{name} ({file_name} at {start}): {e}"));
                        continue;
                    }
                };
                wav::write_mono(&out_dir.join("audio").join(format!("{name}.wav")), piece, TARGET_RATE)?;
                write_npy(&out_dir.join("feats").join(format!("{name}.npy")), &feats, &[frames, FEAT_DIM])?;
                write_npy(&out_dir.join("f0").join(format!("{name}.npy")), &pitch, &[pitch.len()])?;
                let voiced = pitch.iter().filter(|&&f| f > 0.0).count() as f32 / pitch.len().max(1) as f32;
                retrieval.extend_from_slice(&feats);
                sliced_seconds += piece.len() as f64 / TARGET_RATE as f64;
                items.push(Item { name, source: file_name.clone(), start, samples: piece.len(), feat_frames: frames, f0_frames: pitch.len(), voiced });
            }
        }
        if (idx0 + 1) % 50 == 0 || idx0 + 1 == files.len() {
            log(&format!("{}/{} files, {} slices, {:.0} s", idx0 + 1, files.len(), items.len(), started.elapsed().as_secs_f32()));
        }
    }
    ensure!(!items.is_empty(), "no usable slices");

    // silence: the model has to learn to stay quiet (Applio adds logs/mute the same way)
    let mute = vec![0.0f32; MUTE_SECONDS * TARGET_RATE as usize];
    let (feats, frames, _) = ex.run(&mute)?;
    let pitch = vec![0.0f32; MUTE_SECONDS * FEATURE_RATE as usize / mel::HOP + 1];
    wav::write_mono(&out_dir.join("audio").join("mute.wav"), &mute, TARGET_RATE)?;
    write_npy(&out_dir.join("feats").join("mute.npy"), &feats, &[frames, FEAT_DIM])?;
    write_npy(&out_dir.join("f0").join("mute.npy"), &pitch, &[pitch.len()])?;

    let vectors = retrieval.len() / FEAT_DIM;
    write_npy(&out_dir.join("retrieval.npy"), &retrieval, &[vectors, FEAT_DIM])?;
    let summary = Summary { files: files.len(), slices: items.len(), source_seconds, sliced_seconds, retrieval_vectors: vectors, skipped: skipped.len() };
    let manifest = Manifest {
        format: 1,
        sample_rate: TARGET_RATE,
        feature_rate: FEATURE_RATE,
        source_dir: wav_dir.display().to_string(),
        device: format!("{:?}", opts.device),
        loudness_db: opts.loudness_db,
        target_resample_width: TARGET_RESAMPLE_WIDTH,
        feature_resample_width: FEATURE_RESAMPLE_WIDTH,
        rmvpe_threshold: RMVPE_THRESHOLD,
        source_seconds,
        sliced_seconds,
        retrieval_vectors: vectors,
        mute: "mute".into(),
        skipped,
        items,
    };
    std::fs::write(out_dir.join("prepare.json"), serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("writing {}", out_dir.join("prepare.json").display()))?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applio_slicer_settings() {
        let s = Slicer::applio(40_000);
        assert_eq!((s.hop, s.win, s.min_length, s.min_interval, s.max_sil_kept), (600, 2400, 100, 27, 33));
        assert!((s.threshold - 0.007_943_282_347_242_814).abs() < 1e-15);
    }

    #[test]
    fn pieces_follow_applio() {
        let seg = vec![0.0f32; 300_000];
        let lens: Vec<usize> = pieces(&seg, 40_000).iter().map(|p| p.len()).collect();
        // starts 0, 108000; 300000 - 108000 = 192000 > 132000, so a third piece from 216000 takes the rest
        assert_eq!(lens, vec![120_000, 120_000, 84_000]);
        assert_eq!(pieces(&seg[..132_000], 40_000).len(), 1);
        assert_eq!(pieces(&seg[..132_001], 40_000).len(), 2);
    }

    #[test]
    fn npy_header_is_aligned() {
        let dir = std::env::temp_dir().join(format!("mimic-npy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.npy");
        write_npy(&p, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = std::fs::read(&p).unwrap();
        let hlen = u16::from_le_bytes([b[8], b[9]]) as usize;
        assert_eq!((10 + hlen) % 64, 0);
        assert!(std::str::from_utf8(&b[10..10 + hlen]).unwrap().contains("'shape': (2, 3)"));
        assert_eq!(f32::from_le_bytes(b[10 + hlen..14 + hlen].try_into().unwrap()), 1.0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
