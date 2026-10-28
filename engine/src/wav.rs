//! Mono 32-bit float WAV helpers shared by the CLI and the live session.

use std::path::Path;

use anyhow::{Context, Result};

use crate::dsp::resample::Resampler;

/// Reads a WAV as mono (channels averaged), -1..1; returns the samples and the file's rate.
pub fn read_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<_, _>>()?
        }
    };
    let ch = spec.channels as usize;
    let mono = samples.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect();
    Ok((mono, spec.sample_rate))
}

/// `read_mono`, resampled to `rate` unless the file is already at it.
pub fn read_mono_at(path: &Path, rate: u32) -> Result<Vec<f32>> {
    let (audio, file_rate) = read_mono(path)?;
    Ok(if file_rate == rate { audio } else { Resampler::new(file_rate, rate).process(&audio) })
}

/// Writes mono 32-bit float samples at `rate`.
pub fn write_mono(path: &Path, audio: &[f32], rate: u32) -> Result<()> {
    let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
    let mut w = hound::WavWriter::create(path, spec).with_context(|| format!("creating {}", path.display()))?;
    for v in audio {
        w.write_sample(*v)?;
    }
    w.finalize()?;
    Ok(())
}
