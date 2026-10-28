//! ONNX Runtime sessions for the four exported graphs, with typed run helpers.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, ensure};
use ort::environment::Environment;

/// ONNX Runtime allows one environment per process; sessions can be rebuilt freely.
static ENV: OnceLock<Environment> = OnceLock::new();
use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::value::Tensor;
use serde::Deserialize;

use crate::dsp::vad::SileroVad;

pub const FEAT_DIM: usize = 768;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    DirectML,
    Cpu,
}

/// Where the shared nets, the voice and the runtime DLLs live.
#[derive(Clone, Debug)]
pub struct Paths {
    pub models: PathBuf,
    pub voice: String,
    pub runtime: PathBuf,
}

impl Paths {
    fn shared(&self, file: &str) -> PathBuf {
        self.models.join(file)
    }
    fn voice_file(&self, file: &str) -> PathBuf {
        self.models.join("voices").join(&self.voice).join(file)
    }
    pub fn mel_basis(&self) -> PathBuf {
        self.shared("rmvpe_mel.f32")
    }
    /// The voice ships the seam-free synthesizer (synth_det.onnx).
    pub fn has_seamless_synth(&self) -> bool {
        self.voice_file("synth_det.onnx").exists()
    }
    /// Output samples per 100 fps frame, from the voice's synth.json.
    pub fn voice_hop(&self) -> Result<usize> {
        Ok(self.synth_meta()?.hop)
    }
    /// The voice's median voiced f0 from f0_stats.json; None when the voice has no stats.
    pub fn voice_f0_center(&self) -> Result<Option<f32>> {
        #[derive(Deserialize)]
        struct F0Stats {
            median_hz: f32,
        }
        let file = self.voice_file("f0_stats.json");
        if !file.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&file).with_context(|| format!("reading {}", file.display()))?;
        let stats: F0Stats = serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", file.display()))?;
        Ok((stats.median_hz.is_finite() && stats.median_hz > 0.0).then_some(stats.median_hz))
    }
    fn synth_meta(&self) -> Result<SynthMeta> {
        let file = self.voice_file("synth.json");
        let bytes = std::fs::read(&file).with_context(|| format!("reading {}", file.display()))?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}

#[derive(Deserialize)]
struct SynthMeta {
    sample_rate: u32,
    hop: usize,
}

pub struct Models {
    contentvec: Session,
    rmvpe: Session,
    retrieval: Session,
    synth: Session,
    /// seam-free variant (noise and NSF phase as inputs), when exported
    synth_det: Option<Session>,
    /// Silero VAD for the input gate, when models/silero_vad.onnx is present
    pub vad: Option<SileroVad>,
    /// median voiced f0 of the voice, from its f0_stats.json ({"median_hz": ...}), when present
    pub f0_center_hz: Option<f32>,
    pub sample_rate: u32,
    pub hop: usize,
    pub device: Device,
}

/// Level-3 optimised session builder; `path` only names the graph in error messages.
fn session_builder(env: &Environment, path: &Path) -> Result<SessionBuilder> {
    Session::builder(env)
        .with_context(|| format!("loading {}", path.display()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{e}"))
}

/// A CPU session with a fixed intra-op thread count: small side models must not spin up a pool.
pub(crate) fn cpu_session(env: &Environment, path: &Path, threads: usize) -> Result<Session> {
    session_builder(env, path)?
        .with_intra_threads(threads)
        .map_err(|e| anyhow!("{e}"))?
        .commit_from_file(path)
        .with_context(|| format!("loading {}", path.display()))
}

/// The process-wide ONNX Runtime environment, created on first use from the DLLs in `runtime`.
/// With `Device::DirectML` our DirectML.dll is preloaded first, so the older copy in System32 is
/// never picked up; a process that will use DirectML must therefore ask for it on the first call.
pub fn ort_env(runtime: &Path, device: Device) -> Result<&'static Environment> {
    if ENV.get().is_none() {
        if device == Device::DirectML {
            ort::util::preload_dylib(runtime.join("DirectML.dll")).map_err(|e| anyhow!("DirectML.dll: {e}"))?;
        }
        let env = ort::init_from(runtime.join("onnxruntime.dll"))
            .map_err(|e| anyhow!("onnxruntime.dll: {e}"))?
            .with_name("mimic")
            .build()
            .context("creating ONNX Runtime environment")?;
        let _ = ENV.set(env);
    }
    Ok(ENV.get().expect("set above"))
}

impl Models {
    /// Every graph with dynamic shapes; `load_fixed` with empty overrides.
    pub fn load(paths: &Paths, device: Device) -> Result<Self> {
        Self::load_fixed(paths, device, &FixedDims::default())
    }

    /// 16 kHz audio -> (frames, [frames x 768]) at 50 fps.
    pub fn contentvec(&mut self, audio: &[f32]) -> Result<(usize, Vec<f32>)> {
        run_contentvec(&mut self.contentvec, audio)
    }

    /// log-mel [128 x frames] (frames % 32 == 0) -> salience [frames x 360].
    pub fn rmvpe(&mut self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        run_rmvpe(&mut self.rmvpe, mel, frames)
    }

    /// Blend [frames x 768] toward the voice index (top-8, 1/d^2).
    pub fn retrieval(&mut self, feats: &[f32], frames: usize, index_rate: f32) -> Result<Vec<f32>> {
        let input = Tensor::from_array(([frames, FEAT_DIM], feats.to_vec()))?;
        let rate = Tensor::from_array(((), vec![index_rate]))?;
        let out = self.retrieval.run(ort::inputs!["feats" => input, "index_rate" => rate])?;
        let (_, data) = out["blended"].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }

    /// Whole window in, audio for the frames after `skip_head` out (at `sample_rate`).
    pub fn synth(&mut self, phone: &[f32], pitch: &[i64], pitchf: &[f32], skip_head: usize) -> Result<Vec<f32>> {
        let frames = pitch.len();
        ensure!(frames >= 12, "synth needs at least 12 frames, got {frames}");
        let out = self.synth.run(ort::inputs![
            "phone" => Tensor::from_array(([1usize, frames, FEAT_DIM], phone.to_vec()))?,
            "phone_lengths" => Tensor::from_array(([1usize], vec![frames as i64]))?,
            "pitch" => Tensor::from_array(([1usize, frames], pitch.to_vec()))?,
            "pitchf" => Tensor::from_array(([1usize, frames], pitchf.to_vec()))?,
            "sid" => Tensor::from_array(([1usize], vec![0i64]))?,
            "skip_head" => Tensor::from_array(((), vec![skip_head as i64]))?,
        ])?;
        let (_, data) = out["audio"].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }

    pub fn has_seamless_synth(&self) -> bool {
        self.synth_det.is_some()
    }

    /// Seam-free synthesis: caller supplies latent noise [192 x frames] (channel-major),
    /// source noise for every output sample and the NSF start phase in cycles.
    pub fn synth_seamless(
        &mut self,
        phone: &[f32],
        pitch: &[i64],
        pitchf: &[f32],
        skip_head: usize,
        zp_noise: &[f32],
        src_noise: &[f32],
        phase0: f32,
    ) -> Result<Vec<f32>> {
        let frames = pitch.len();
        ensure!(frames >= 12, "synth needs at least 12 frames, got {frames}");
        let session = self.synth_det.as_mut().ok_or_else(|| anyhow!("synth_det.onnx not exported for this voice"))?;
        let out = session.run(ort::inputs![
            "phone" => Tensor::from_array(([1usize, frames, FEAT_DIM], phone.to_vec()))?,
            "phone_lengths" => Tensor::from_array(([1usize], vec![frames as i64]))?,
            "pitch" => Tensor::from_array(([1usize, frames], pitch.to_vec()))?,
            "pitchf" => Tensor::from_array(([1usize, frames], pitchf.to_vec()))?,
            "sid" => Tensor::from_array(([1usize], vec![0i64]))?,
            "skip_head" => Tensor::from_array(((), vec![skip_head as i64]))?,
            "zp_noise" => Tensor::from_array(([1usize, 192, frames], zp_noise.to_vec()))?,
            "src_noise" => Tensor::from_array(([1usize, src_noise.len(), 1], src_noise.to_vec()))?,
            "phase0" => Tensor::from_array(((), vec![phase0]))?,
        ])?;
        let (_, data) = out["audio"].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }
}

fn run_contentvec(session: &mut Session, audio: &[f32]) -> Result<(usize, Vec<f32>)> {
    let input = Tensor::from_array(([1usize, audio.len()], audio.to_vec()))?;
    let out = session.run(ort::inputs!["audio" => input])?;
    let (shape, data) = out["feats"].try_extract_tensor::<f32>()?;
    ensure!(shape.len() == 3 && shape[2] as usize == FEAT_DIM, "contentvec output {:?}", &shape[..]);
    Ok((shape[1] as usize, data.to_vec()))
}

fn run_rmvpe(session: &mut Session, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
    let input = Tensor::from_array(([1usize, 128, frames], mel.to_vec()))?;
    let out = session.run(ort::inputs!["mel" => input])?;
    let (_, data) = out["salience"].try_extract_tensor::<f32>()?;
    Ok(data.to_vec())
}

/// contentvec and RMVPE alone, the two graphs every voice shares: `mimic prepare` extracts training
/// features with them, so training sees exactly what the realtime path computes. Dynamic shapes.
pub struct FeatureNets {
    contentvec: Session,
    rmvpe: Session,
}

impl FeatureNets {
    /// `models` holds contentvec.onnx and rmvpe.onnx; `runtime` the ONNX Runtime DLLs.
    pub fn load(models: &Path, runtime: &Path, device: Device) -> Result<Self> {
        let env = ort_env(runtime, device)?;
        let dynamic = std::collections::BTreeMap::new();
        Ok(Self {
            contentvec: fixed_session(env, &models.join("contentvec.onnx"), device, &dynamic)?,
            rmvpe: fixed_session(env, &models.join("rmvpe.onnx"), device, &dynamic)?,
        })
    }

    /// 16 kHz audio -> (frames, [frames x 768]) at 50 fps.
    pub fn contentvec(&mut self, audio: &[f32]) -> Result<(usize, Vec<f32>)> {
        run_contentvec(&mut self.contentvec, audio)
    }

    /// log-mel [128 x frames] (frames % 32 == 0) -> salience [frames x 360].
    pub fn rmvpe(&mut self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        run_rmvpe(&mut self.rmvpe, mel, frames)
    }
}

/// Per-graph symbolic ONNX dimension overrides. Empty maps preserve dynamic dimensions.
/// Names are graph-local: `frames` has a different value in each graph.
#[derive(Clone, Debug, Default)]
pub struct FixedDims {
    pub contentvec: std::collections::BTreeMap<String, i64>,
    pub rmvpe: std::collections::BTreeMap<String, i64>,
    pub retrieval: std::collections::BTreeMap<String, i64>,
    pub synth: std::collections::BTreeMap<String, i64>,
    pub synth_det: std::collections::BTreeMap<String, i64>,
}

impl FixedDims {
    fn validate(&self) -> Result<()> {
        for (graph, dimensions) in [
            ("contentvec", &self.contentvec), ("rmvpe", &self.rmvpe),
            ("retrieval", &self.retrieval), ("synth", &self.synth),
            ("synth_det", &self.synth_det),
        ] {
            for (name, size) in dimensions {
                ensure!(!name.is_empty() && !name.contains('\0'), "{graph}: invalid dimension name");
                ensure!(*size > 0, "{graph}: dimension {name} must be positive, got {size}");
            }
        }
        Ok(())
    }
}

/// Selects a session for the optional IO-binding interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelGraph {
    Contentvec,
    Rmvpe,
    Retrieval,
    Synth,
    SynthDet,
}

fn fixed_session(
    env: &Environment, path: &Path, device: Device,
    dimensions: &std::collections::BTreeMap<String, i64>,
) -> Result<Session> {
    let mut builder = session_builder(env, path)?;
    if device == Device::DirectML {
        // the DirectML EP needs sequential execution without memory patterns
        builder = builder
            .with_memory_pattern(false).map_err(|e| anyhow!("{e}"))?
            .with_parallel_execution(false).map_err(|e| anyhow!("{e}"))?
            .with_execution_providers([ort::ep::DirectML::default().build()])
            .map_err(|e| anyhow!("{e}"))?;
    }
    for (name, size) in dimensions {
        builder = builder.with_dimension_override(name, *size)
            .map_err(|e| anyhow!("{}: fixing {name}={size}: {e}", path.display()))?;
    }
    builder.commit_from_file(path).with_context(|| format!("loading {}", path.display()))
}

impl Models {
    /// Sessions with the free dimensions in `dims` fixed (DirectML then compiles each graph once
    /// instead of once per shape). Rebuild when the layout changes; `skip_head` is a scalar value,
    /// not a dimension name.
    pub fn load_fixed(paths: &Paths, device: Device, dims: &FixedDims) -> Result<Self> {
        dims.validate()?;
        let env = ort_env(&paths.runtime, device)?;
        let meta = paths.synth_meta()?;
        Ok(Self {
            contentvec: fixed_session(env, &paths.shared("contentvec.onnx"), device, &dims.contentvec)?,
            rmvpe: fixed_session(env, &paths.shared("rmvpe.onnx"), device, &dims.rmvpe)?,
            retrieval: fixed_session(env, &paths.voice_file("retrieval.onnx"), device, &dims.retrieval)?,
            synth: fixed_session(env, &paths.voice_file("synth.onnx"), device, &dims.synth)?,
            synth_det: {
                let p = paths.voice_file("synth_det.onnx");
                if p.exists() { Some(fixed_session(env, &p, device, &dims.synth_det)?) } else { None }
            },
            vad: {
                let p = paths.shared("silero_vad.onnx");
                if p.exists() { Some(SileroVad::load(env, &p)?) } else { None }
            },
            sample_rate: meta.sample_rate,
            hop: meta.hop,
            f0_center_hz: paths.voice_f0_center()?,
            device,
        })
    }

    /// Create a reusable binding for one graph. Bind owned tensors or device outputs.
    /// A binding belongs to this exact Models instance and selected graph.
    pub fn create_binding(&self, graph: ModelGraph) -> Result<ort::session::IoBinding> {
        let session = match graph {
            ModelGraph::Contentvec => &self.contentvec,
            ModelGraph::Rmvpe => &self.rmvpe,
            ModelGraph::Retrieval => &self.retrieval,
            ModelGraph::Synth => &self.synth,
            ModelGraph::SynthDet => self.synth_det.as_ref()
                .ok_or_else(|| anyhow!("synth_det.onnx not exported for this voice"))?,
        };
        session.create_binding().map_err(Into::into)
    }

    /// Synchronize and execute an explicitly bound graph without extracting CPU Vecs.
    /// Output buffers may be reused by the binding: consume downstream before re-running
    /// the producer, or explicitly copy the output. CPU mutation requires re-binding inputs.
    pub fn run_bound<'a>(
        &'a mut self, graph: ModelGraph, binding: &'a ort::session::IoBinding,
    ) -> Result<ort::session::SessionOutputs<'a>> {
        binding.synchronize_inputs()?;
        let session = match graph {
            ModelGraph::Contentvec => &mut self.contentvec,
            ModelGraph::Rmvpe => &mut self.rmvpe,
            ModelGraph::Retrieval => &mut self.retrieval,
            ModelGraph::Synth => &mut self.synth,
            ModelGraph::SynthDet => self.synth_det.as_mut()
                .ok_or_else(|| anyhow!("synth_det.onnx not exported for this voice"))?,
        };
        let output = session.run_binding(binding)?;
        binding.synchronize_outputs()?;
        Ok(output)
    }

    /// Requested output placement for IO binding. Callers must check the returned
    /// tensor's memory_info: ONNX Runtime can fall back to CPU allocation.
    /// DirectML's current configuration uses logical device 0, matching session().
    pub fn binding_memory(&self) -> Result<ort::memory::MemoryInfo<'static>> {
        use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
        let device = if self.device == Device::DirectML {
            AllocationDevice::DIRECTML
        } else { AllocationDevice::CPU };
        MemoryInfo::new(device, 0, AllocatorType::Device, MemoryType::Default).map_err(Into::into)
    }
}

#[cfg(test)]
mod fixed_dims_tests {
    use super::*;
    use crate::dsp::mel::MelExtractor;
    use crate::pipeline::{BlockProcessor, BlockStats, Layout, Params};
    use ort::value::TensorValueType;
    use std::io::Write;
    use std::time::Instant;

    fn named(values: &[(&str, i64)]) -> std::collections::BTreeMap<String, i64> {
        values.iter().map(|(name, size)| ((*name).to_string(), *size)).collect()
    }

    fn default_dimensions() -> FixedDims {
        FixedDims {
            contentvec: named(&[("samples", 44640), ("frames", 139), ("LayerNormalizationfeats_dim_0", 1)]),
            rmvpe: named(&[("frames", 32), ("Sigmoidsalience_dim_0", 1)]),
            retrieval: named(&[("frames", 15)]),
            synth: named(&[("frames", 279), ("Tanhaudio_dim_0", 1)]),
            synth_det: named(&[("frames", 279), ("samples", 18800), ("Tanhaudio_dim_0", 1)]),
        }
    }

    fn paths() -> Paths {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        // MIMIC_VOICE names the voice folder under models/voices
        let voice = std::env::var("MIMIC_VOICE").expect("set MIMIC_VOICE to a voice folder under models/voices");
        Paths { models: root.join("models"), voice, runtime: root.join("runtime") }
    }

    fn output_dir() -> Result<PathBuf> {
        let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.codex-target/fixed-dims-ab");
        std::fs::create_dir_all(&out)?;
        Ok(out)
    }

    #[test]
    fn fixed_dims_validates_overrides() {
        let mut dims = default_dimensions();
        dims.validate().unwrap();
        dims.rmvpe.insert("frames".to_string(), 0);
        assert!(dims.validate().is_err());
        dims.rmvpe = named(&[("frames", 32)]);
        dims.contentvec.insert("bad\0name".to_string(), 1);
        assert!(dims.validate().is_err());
    }

    fn summary(stats: &[BlockStats]) -> serde_json::Value {
        let active: Vec<_> = stats.iter().filter(|s| !s.silent).skip(1).collect();
        let distribution = |select: fn(&BlockStats) -> f32| {
            let mut values: Vec<_> = active.iter().map(|s| select(s)).collect();
            values.sort_by(f32::total_cmp);
            serde_json::json!({"median": crate::dsp::percentile(&values, 0.5),
                "p95": crate::dsp::percentile(&values, 0.95),
                "max": values.last().copied().unwrap_or(0.0)})
        };
        serde_json::json!({
            "blocks_all": stats.len(), "blocks_voiced": active.len() + 1,
            "warm_blocks": active.len(),
            "first_voiced_ms": stats.iter().find(|s| !s.silent).map(|s| s.ms_total),
            "ms_total": distribution(|s| s.ms_total),
            "ms_synth": distribution(|s| s.ms_synth),
            "ms_features": distribution(|s| s.ms_features),
            "ms_pitch": distribution(|s| s.ms_pitch),
        })
    }

    fn convert_for_test(label: &str, audio: &[f32], fixed: bool) -> Result<(Vec<f32>, Vec<BlockStats>, serde_json::Value)> {
        let paths = paths();
        let start = Instant::now();
        let models = if fixed { Models::load_fixed(&paths, Device::DirectML, &default_dimensions())? }
            else { Models::load(&paths, Device::DirectML)? };
        let load_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mel = MelExtractor::new(&paths.mel_basis())?;
        let params = Params { pitch: -3.0, ..Params::default() };
        let mut processor = BlockProcessor::new(models, mel, &Layout::default(), params);
        processor.consistency_metrics = true; // exactly as mimic convert
        let block = processor.block_size();
        let (mut output, mut all_stats) = (Vec::new(), Vec::new());
        let out_dir = output_dir()?;
        let mut csv = std::fs::File::create(out_dir.join(format!("{label}.csv")))?;
        writeln!(csv, "block,silent,ms_total,ms_pitch,ms_features,ms_synth")?;
        println!("fixed-dims run={label} pid={} block={block} load_ms={load_ms:.1}", std::process::id());
        for (index, chunk) in audio.chunks(block).enumerate() {
            let mut input = chunk.to_vec();
            input.resize(block, 0.0);
            let (converted, stats) = processor.process(&input)
                .with_context(|| format!("{label} block {index}"))?;
            writeln!(csv, "{index},{},{},{},{},{}", stats.silent,
                stats.ms_total, stats.ms_pitch, stats.ms_features, stats.ms_synth)?;
            output.extend_from_slice(&converted);
            all_stats.push(stats);
        }
        crate::wav::write_mono(&out_dir.join(format!("{label}.wav")), &output, 48000)?;
        let mut info = summary(&all_stats);
        info["load_ms"] = load_ms.into();
        println!("fixed-dims {label}: {info}");
        Ok((output, all_stats, info))
    }

    /// Explicit opt-in: requires idle GPU and the user's local input/models/runtime.
    #[test]
    #[ignore = "manual DirectML A/B; check no game is running first"]
    fn fixed_dims_ab_directml() -> Result<()> {
        println!("fixed-dims experiment pid={}", std::process::id());
        // MIMIC_AB_INPUT names the speech recording to convert
        let input = std::env::var("MIMIC_AB_INPUT").expect("set MIMIC_AB_INPUT to a speech WAV");
        let audio = crate::wav::read_mono_at(Path::new(&input), 48000)?;
        let (dynamic, dynamic_stats, dynamic_info) = convert_for_test("dynamic", &audio, false)?;
        let (fixed, fixed_stats, fixed_info) = convert_for_test("fixed", &audio, true)?;
        ensure!(dynamic.len() == fixed.len(), "A/B output lengths differ");
        let max_error = dynamic.iter().zip(&fixed).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        let flags_equal = dynamic_stats.iter().zip(&fixed_stats).all(|(a,b)| a.silent == b.silent);
        let report = serde_json::json!({"dynamic": dynamic_info, "fixed": fixed_info,
            "input": input, "pitch": -3,
            "samples": dynamic.len(), "max_abs_error": max_error, "gate_flags_equal": flags_equal});
        std::fs::write(output_dir()?.join("summary.json"), serde_json::to_vec_pretty(&report)?)?;
        println!("fixed-dims A/B: {report}");
        ensure!(flags_equal, "A/B gate decisions changed");
        ensure!(fixed.iter().all(|x| x.is_finite()), "nonfinite fixed output");
        Ok(())
    }

    /// Separate placement/parity smoke test; this is not a GPU-only full pipeline.
    #[test]
    #[ignore = "manual DirectML IO binding; check no game is running first"]
    fn fixed_dims_binding_directml() -> Result<()> {
        use ort::memory::AllocationDevice;
        let mut models = Models::load_fixed(&paths(), Device::DirectML, &default_dimensions())?;
        let audio = vec![0.0_f32; 44640];
        let (frames, baseline) = models.contentvec(&audio)?;
        let mut binding = models.create_binding(ModelGraph::Contentvec)?;
        binding.bind_input("audio", &Tensor::from_array(([1_usize, audio.len()], audio))?)?;
        binding.bind_output_to_device("feats", &models.binding_memory()?)?;
        let output = models.run_bound(ModelGraph::Contentvec, &binding)?.remove("feats")
            .ok_or_else(|| anyhow!("missing contentvec output"))?;
        let tensor = output.downcast::<TensorValueType<f32>>()?;
        let location = tensor.memory_info().allocation_device().as_str().to_string();
        ensure!(location == "DML", "bound output fell back to {location}");
        let cpu = tensor.to(AllocationDevice::CPU, 0)?;
        let (_, samples) = cpu.try_extract_tensor::<f32>()?;
        let error = samples.iter().zip(&baseline).map(|(a,b)| (a-b).abs()).fold(0.0_f32, f32::max);
        let report = serde_json::json!({"contentvec_output_device": location, "frames": frames,
            "max_abs_error_vs_typed": error, "bound_output_shape": format!("{:?}", tensor.shape()),
            "whole_pipeline_device_resident": false});
        std::fs::write(output_dir()?.join("binding.json"), serde_json::to_vec_pretty(&report)?)?;
        println!("fixed-dims IO binding: {report}");
        ensure!(error < 1e-5, "bound contentvec output differs: {error}");
        Ok(())
    }
}