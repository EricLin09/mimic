//! Engine tools: offline conversion (`convert`), device listing (`devices`), the live
//! microphone path (`live`), the input-gate probe (`gate`) and training data (`prepare`).

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use engine::dsp::gate::Gate;

use engine::dsp::percentile;
use engine::dsp::resample::StreamResampler;
use engine::dsp::vad::SileroVad;
use engine::models::ort_env;
use engine::pipeline::{DEVICE_RATE, F0Map};
use engine::prepare::PrepareOptions;
use engine::wav;
use engine::{BlockProcessor, Device, Layout, LiveConfig, LiveSession, Params, Paths, list_devices, record_input};

#[derive(Parser)]
#[command(name = "mimic", about = "Mimic voice engine tools")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// Defaults come from the engine's own `Layout::default` / `Params::default`, so the CLI and the
/// app always agree.
#[derive(Args, Clone)]
struct EngineOpts {
    /// voice folder under models/voices (default: the first one that holds a synth.json)
    #[arg(long)]
    voice: Option<String>,
    /// semitones; 0 = centred on the voice's median f0 when it has f0_stats.json
    #[arg(long, default_value_t = Params::default().pitch, allow_hyphen_values = true)]
    pitch: f32,
    /// keep the speaker's own pitch centre: `--pitch` is then a plain shift (the old behaviour)
    #[arg(long)]
    no_align_pitch: bool,
    #[arg(long, default_value_t = Params::default().index_rate)]
    index: f32,
    #[arg(long, default_value_t = Params::default().protect)]
    protect: f32,
    #[arg(long, default_value_t = Layout::default().block_ms)]
    block_ms: f32,
    #[arg(long, default_value_t = Layout::default().extra_s)]
    extra: f32,
    /// run on the CPU instead of DirectML
    #[arg(long)]
    cpu: bool,
    /// build the sessions with dynamic shapes instead of fixing them to the layout (A/B runs)
    #[arg(long)]
    dynamic_shapes: bool,
    /// Applio's rendering (fresh noise per block, cut before the flow) instead of seamless
    #[arg(long)]
    legacy: bool,
    /// decoder warm-up frames for seamless rendering
    #[arg(long, default_value_t = Layout::default().margin_frames)]
    margin: usize,
    /// crossfade between consecutive renders
    #[arg(long, value_enum, default_value_t = Xfade::Vocoder)]
    xfade: Xfade,
    /// fade-in length at the first render after silence
    #[arg(long, default_value_t = Layout::default().onset_fade_ms)]
    onset_ms: f32,
    /// 3-point median over voiced f0 before the frames are frozen
    #[arg(long)]
    f0_median: bool,
    /// extra RMVPE left context (rounded up to 32 mel frames); does not add lookahead
    #[arg(long, default_value_t = Layout::default().f0_context_ms)]
    f0_context_ms: usize,
    /// render this many 10 ms frames behind the newest audio (seamless mode)
    #[arg(long, default_value_t = Layout::default().lookahead_frames)]
    lookahead: usize,
    /// crossfade length between consecutive renders
    #[arg(long, default_value_t = Layout::default().crossfade_s * 1000.0)]
    crossfade_ms: f32,
    /// SOLA alignment search range (0 = trust the seamless time alignment)
    #[arg(long, default_value_t = Layout::default().sola_ms)]
    sola_ms: f32,
    /// input RMS below this (dBFS) is treated as silence
    #[arg(long, default_value_t = Params::default().silence_db, allow_hyphen_values = true)]
    silence_db: f32,
    /// RNNoise also on the audio that gets converted (before defaults v2 it always was)
    #[arg(long)]
    denoise_audio: bool,
    /// no RNNoise at all: the gate judges the raw input too
    #[arg(long)]
    no_denoise: bool,
    /// voice threshold 0..1 on Silero VAD's chunk probability (0 = level-only gate)
    #[arg(long, default_value_t = Params::default().vad)]
    vad: f32,
    /// follow the speaker's loudness contour (shape only): 0 = the model's own dynamics
    #[arg(long, default_value_t = Params::default().envelope)]
    envelope: f32,
    /// map the speaker's pitch range onto the voice's: in_median_hz:out_median_hz[:spread]
    #[arg(long, value_parser = parse_f0_map)]
    f0_map: Option<F0Map>,
}

fn parse_f0_map(s: &str) -> Result<F0Map, String> {
    let v: Vec<f32> = s.split(':').map(|p| p.trim().parse::<f32>().map_err(|e| format!("{p:?}: {e}"))).collect::<Result<_, _>>()?;
    match v.as_slice() {
        [i, o] => Ok(F0Map { in_median: *i, out_median: *o, spread: 1.0 }),
        [i, o, k] => Ok(F0Map { in_median: *i, out_median: *o, spread: *k }),
        _ => Err("expected in_hz:out_hz[:spread]".into()),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Xfade {
    /// Applio's phase-vocoder blend (the baseline)
    Vocoder,
    /// plain sin^2 amplitude blend
    Sin2,
    /// correlation-adaptive equal-power blend
    Adaptive,
}

impl std::fmt::Display for Xfade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Xfade::Vocoder => "vocoder",
            Xfade::Sin2 => "sin2",
            Xfade::Adaptive => "adaptive",
        })
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert a WAV file block by block, exactly as the realtime path would.
    Convert {
        input: PathBuf,
        output: PathBuf,
        #[command(flatten)]
        engine: EngineOpts,
    },
    /// List audio devices.
    Devices,
    /// Live conversion: microphone (or a WAV paced in real time) -> output device.
    Live {
        /// input device name or substring (default: system default)
        #[arg(long = "in")]
        input: Option<String>,
        /// output device name or substring (default: system default)
        #[arg(long = "out")]
        output: Option<String>,
        /// feed this WAV instead of a microphone
        #[arg(long)]
        in_file: Option<PathBuf>,
        /// save the processed output
        #[arg(long)]
        record: Option<PathBuf>,
        /// stop after this many seconds (0 = until Enter)
        #[arg(long, default_value_t = 0.0)]
        seconds: f32,
        /// zeros queued ahead of the first block (default: warm-up p95 + 10 ms)
        #[arg(long)]
        prime_ms: Option<f32>,
        /// also play the output on this device (the app's "hear yourself" monitor)
        #[arg(long)]
        monitor: Option<String>,
        /// emit coded pulses and measure output->input RTT; requires an explicit loopback
        #[arg(long)]
        measure: bool,
        #[command(flatten)]
        engine: EngineOpts,
    },
    /// Watch the input gate block by block: level, Silero voice probability and the decision,
    /// from a WAV or live from a microphone. Only the small VAD model is loaded, on the CPU.
    Gate {
        /// WAV to analyse; omitted = record from the microphone
        #[arg(long)]
        file: Option<PathBuf>,
        /// input device name or substring (default: system default)
        #[arg(long = "in")]
        input: Option<String>,
        /// microphone capture length
        #[arg(long, default_value_t = 30.0)]
        seconds: f32,
        /// save the captured microphone audio here (48 kHz mono)
        #[arg(long)]
        save: Option<PathBuf>,
        #[arg(long, default_value_t = Layout::default().block_ms)]
        block_ms: f32,
        /// level threshold, dBFS above 150 Hz after denoising
        #[arg(long, default_value_t = Params::default().silence_db, allow_hyphen_values = true)]
        gate_db: f32,
        /// voice threshold 0..1 (0 = level-only gate)
        #[arg(long, default_value_t = Params::default().vad)]
        vad: f32,
        /// judge the level on the raw input instead of the denoised one
        #[arg(long)]
        no_denoise: bool,
        /// print every block, not just the summary
        #[arg(long, short)]
        verbose: bool,
    },
    /// Training data for a voice: slice the WAVs as Applio does, resample to 40 kHz and extract
    /// contentvec features and RMVPE pitch with the engine's own graphs (docs/training.md).
    Prepare {
        /// folder of WAV files, one speaker (top level only)
        wav_dir: PathBuf,
        /// output folder, created if missing
        out_dir: PathBuf,
        /// run the feature nets on the CPU instead of DirectML
        #[arg(long)]
        cpu: bool,
        /// scale each file so its speech RMS sits at this level, dBFS (default: levels untouched, as Applio)
        #[arg(long, allow_hyphen_values = true)]
        loudness: Option<f32>,
    },
}

fn repo_dir(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(sub)
}

/// The voice to load: `--voice`, else the first folder under models/voices that holds a synth.json.
fn voice_id(o: &EngineOpts) -> Result<String> {
    if let Some(v) = &o.voice {
        return Ok(v.clone());
    }
    let dir = repo_dir("models").join("voices");
    let mut ids: Vec<String> = std::fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("synth.json").exists())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    ids.sort();
    ids.into_iter().next().ok_or_else(|| anyhow::anyhow!("no voice in {}: export one, or pass --voice", dir.display()))
}

fn build_processor(o: &EngineOpts) -> Result<BlockProcessor> {
    let paths = Paths { models: repo_dir("models"), voice: voice_id(o)?, runtime: repo_dir("runtime") };
    let device = if o.cpu { Device::Cpu } else { Device::DirectML };

    let layout = Layout {
        block_ms: o.block_ms, extra_s: o.extra, seamless: !o.legacy, margin_frames: o.margin,
        phase_vocoder: o.xfade == Xfade::Vocoder, adaptive_xfade: o.xfade == Xfade::Adaptive,
        lookahead_frames: o.lookahead, crossfade_s: o.crossfade_ms / 1000.0, sola_ms: o.sola_ms,
        onset_fade_ms: o.onset_ms, f0_median: o.f0_median, f0_context_ms: o.f0_context_ms,
    };
    let params = Params {
        pitch: o.pitch, index_rate: o.index, protect: o.protect, silence_db: o.silence_db, denoise_audio: o.denoise_audio && !o.no_denoise, denoise_gate: !o.no_denoise, vad: o.vad,
        envelope: o.envelope, f0_map: o.f0_map, align_pitch: !o.no_align_pitch,
    };
    let t = Instant::now();
    let proc = BlockProcessor::load(&paths, device, &layout, params, !o.dynamic_shapes)?;
    let shapes = if o.dynamic_shapes { "dynamic" } else { "fixed" };
    println!("models loaded on {device:?} with {shapes} shapes in {:.1}s", t.elapsed().as_secs_f32());
    println!(
        "block {:.0} ms, lookahead {} ms, rendering: {}, crossfade: {}",
        proc.block_size() as f32 * 1000.0 / DEVICE_RATE as f32,
        proc.lookahead_samples() * 1000 / DEVICE_RATE as usize,
        if proc.is_seamless() { "seamless" } else { "legacy (Applio)" },
        o.xfade
    );
    Ok(proc)
}

fn dbfs(level: f32) -> f32 {
    20.0 * (level.max(1e-6)).log10()
}

fn convert(input: PathBuf, output: PathBuf, o: EngineOpts) -> Result<()> {
    let mut proc = build_processor(&o)?;
    proc.consistency_metrics = true;
    let audio = wav::read_mono_at(&input, DEVICE_RATE)?;
    if audio.is_empty() {
        bail!("empty input");
    }
    let block = proc.block_size();

    let mut out = Vec::with_capacity(audio.len() + block);
    let mut times = Vec::new();
    let (mut overlaps, mut offsets, mut cons0, mut cons_best, mut lags) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut first = None;
    let mut f0s: Vec<(f32, f32)> = Vec::new();
    for chunk in audio.chunks(block) {
        let mut buf = chunk.to_vec();
        buf.resize(block, 0.0);
        let (y, s) = proc.process(&buf)?;
        if let Some(o) = s.overlap {
            overlaps.push(o);
            offsets.push(s.sola_offset as f32);
        }
        if let (Some(c), Some((cb, lag))) = (s.consistency, s.consistency_best) {
            cons0.push(c);
            cons_best.push(cb);
            lags.push(lag as f32);
        }
        if !s.silent {
            if s.f0_in > 0.0 {
                f0s.push((s.f0_in, s.f0_out));
            }
            // the first voiced block includes DirectML shader compilation
            if first.is_none() {
                first = Some(s.ms_total);
            } else {
                times.push((s.ms_total, s.ms_pitch, s.ms_features, s.ms_synth));
            }
        }
        out.extend_from_slice(&y);
    }

    wav::write_mono(&output, &out, DEVICE_RATE)?;

    let block_ms = block as f32 / DEVICE_RATE as f32 * 1000.0;
    let mut total: Vec<f32> = times.iter().map(|t| t.0).collect();
    total.sort_by(f32::total_cmp);
    let mean = |f: fn(&(f32, f32, f32, f32)) -> f32| times.iter().map(f).sum::<f32>() / times.len().max(1) as f32;
    println!("wrote {} ({:.1}s)", output.display(), out.len() as f32 / DEVICE_RATE as f32);
    println!("block {block_ms:.0} ms, voiced blocks {} (first {:.0} ms, warm-up)", times.len() + 1, first.unwrap_or(0.0));
    println!(
        "per block: median {:.1} ms, p95 {:.1} ms, max {:.1} ms  -> RTF {:.2}",
        percentile(&total, 0.5), percentile(&total, 0.95), total.last().copied().unwrap_or(0.0),
        percentile(&total, 0.5) / block_ms
    );
    println!("stages (mean): pitch {:.1} ms, features {:.1} ms, synth {:.1} ms", mean(|t| t.1), mean(|t| t.2), mean(|t| t.3));
    for v in [&mut overlaps, &mut offsets, &mut cons0, &mut cons_best, &mut lags] {
        v.sort_by(f32::total_cmp);
    }
    println!(
        "render overlap similarity over {} seams: median {:.3}, p10 {:.3} | SOLA offset median {:.0}, p90 {:.0} samples",
        overlaps.len(), percentile(&overlaps, 0.5), percentile(&overlaps, 0.1), percentile(&offsets, 0.5), percentile(&offsets, 0.9)
    );
    println!(
        "render consistency over {} shared regions: zero-lag median {:.3} (p10 {:.3}) | best-lag median {:.3}, lag median {:.0} (p10 {:.0}, p90 {:.0}) samples",
        cons0.len(), percentile(&cons0, 0.5), percentile(&cons0, 0.1), percentile(&cons_best, 0.5),
        percentile(&lags, 0.5), percentile(&lags, 0.1), percentile(&lags, 0.9)
    );
    let (mut fi, mut fo): (Vec<f32>, Vec<f32>) = f0s.iter().copied().unzip();
    fi.sort_by(f32::total_cmp);
    fo.sort_by(f32::total_cmp);
    println!(
        "pitch over {} voiced blocks: in median {:.0} Hz (p10 {:.0}, p90 {:.0}) -> out median {:.0} Hz (p10 {:.0}, p90 {:.0})",
        fi.len(), percentile(&fi, 0.5), percentile(&fi, 0.1), percentile(&fi, 0.9),
        percentile(&fo, 0.5), percentile(&fo, 0.1), percentile(&fo, 0.9)
    );
    Ok(())
}

fn devices() -> Result<()> {
    for d in list_devices()? {
        let kind = match (d.input, d.output) {
            (true, true) => "in+out",
            (true, false) => "in",
            (false, true) => "out",
            _ => "-",
        };
        let flags = format!(
            "{}{}",
            if d.default_input { " [default in]" } else { "" },
            if d.default_output { " [default out]" } else { "" }
        );
        let flags = if d.output && !d.output_ok { format!("{flags} [won't open at 48 kHz now]") } else { flags };
        println!("{kind:6} {:>6} Hz {:>2} ch  {}{}", d.default_rate, d.channels, d.name, flags);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn live(
    input: Option<String>, output: Option<String>, in_file: Option<PathBuf>, record: Option<PathBuf>, seconds: f32, prime_ms: Option<f32>,
    monitor: Option<String>, measure: bool, o: EngineOpts,
) -> Result<()> {
    if measure {
        if in_file.is_some() || record.is_some() || monitor.is_some() || prime_ms.is_some() {
            bail!("--measure cannot be combined with --in-file, --record, --monitor or --prime-ms");
        }
        let input = input.ok_or_else(|| anyhow::anyhow!("--measure requires explicit --in and --out devices"))?;
        let output = output.ok_or_else(|| anyhow::anyhow!("--measure requires explicit --in and --out devices"))?;
        return measure_live(input, output, seconds, &o);
    }
    let proc = build_processor(&o)?;
    let cfg = LiveConfig { input: input.clone(), output: output.clone(), input_file: in_file.clone(), record: record.clone(), prime_ms, monitor_on: monitor.is_some(), monitor };
    println!("warming up the GPU...");
    let session = LiveSession::start(proc, cfg)?;
    if let Some(w) = &session.monitor_warning {
        println!("monitor skipped: {w}");
    }
    println!(
        "warm-up blocks: {} ms; output primed with {:.0} ms",
        session.warmup_ms.iter().map(|v| format!("{v:.0}")).collect::<Vec<_>>().join(" "),
        session.prime_ms
    );
    println!(
        "live: {} -> {} | block {:.0} ms + lookahead {:.0} ms{}",
        in_file.as_ref().map(|p| p.display().to_string()).or(input).unwrap_or_else(|| "default input".into()),
        output.unwrap_or_else(|| "default output".into()),
        session.block_ms, session.lookahead_ms,
        if seconds > 0.0 { format!(" | stopping after {seconds:.0} s") } else { " | press Enter to stop".into() }
    );

    // Enter watcher (only when running open-ended)
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
    if seconds <= 0.0 {
        std::thread::spawn(move || {
            let mut line = String::new();
            // EOF (stdin not a terminal) is not a request to stop
            if matches!(std::io::stdin().lock().read_line(&mut line), Ok(n) if n > 0) {
                let _ = stop_tx.send(());
            }
        });
    }

    let start = Instant::now();
    let mut last_print = Instant::now();
    let mut procs: Vec<f32> = Vec::new();
    let (mut lv_in, mut lv_out, mut f0i, mut f0o, mut lat) = (0f32, 0f32, 0f32, 0f32, 0f32);
    let mut blocks = 0u64;
    let mut all_procs: Vec<f32> = Vec::new();
    loop {
        while let Ok(t) = session.telemetry().try_recv() {
            blocks += 1;
            procs.push(t.proc_ms);
            all_procs.push(t.proc_ms);
            lv_in = lv_in.max(t.input_level);
            lv_out = lv_out.max(t.output_level);
            if t.f0_in > 0.0 {
                f0i = t.f0_in;
                f0o = t.f0_out;
            }
            lat = t.latency_ms;
        }
        if last_print.elapsed() >= Duration::from_secs(1) {
            procs.sort_by(f32::total_cmp);
            print!(
                "\r[{:5.1}s] in {:6.1} dBFS  out {:6.1} dBFS  f0 {:3.0}->{:3.0} Hz  proc med {:4.1} max {:4.1} ms  latency {:4.0} ms  under {} over {}   ",
                start.elapsed().as_secs_f32(), dbfs(lv_in), dbfs(lv_out), f0i, f0o,
                percentile(&procs, 0.5), procs.last().copied().unwrap_or(0.0), lat,
                session.underruns(), session.overruns()
            );
            std::io::stdout().flush()?;
            procs.clear();
            lv_in = 0.0;
            lv_out = 0.0;
            last_print = Instant::now();
        }
        if seconds > 0.0 && start.elapsed().as_secs_f32() >= seconds {
            break;
        }
        if in_file.is_some() && !session.feeding() {
            // the ring drains after the file ends; those underruns are expected and not reported
            let steady = session.underruns();
            std::thread::sleep(Duration::from_millis(600));
            println!("\nfile finished; underruns during playback: {steady}");
            break;
        }
        if stop_rx.try_recv().is_ok() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    println!();
    let (under, over) = (session.underruns(), session.overruns());
    session.stop()?;
    all_procs.sort_by(f32::total_cmp);
    println!(
        "stopped after {} blocks: proc median {:.1} ms, p95 {:.1} ms, max {:.1} ms; underruns {under}, overruns {over}",
        blocks, percentile(&all_procs, 0.5), percentile(&all_procs, 0.95), all_procs.last().copied().unwrap_or(0.0)
    );
    if let Some(r) = record {
        println!("recorded {}", r.display());
    }
    Ok(())
}

fn measure_live(input: String, output: String, seconds: f32, o: &EngineOpts) -> Result<()> {
    use engine::realtime::LatencyMeasurement;
    if !seconds.is_finite() || seconds < 0.0 || !o.block_ms.is_finite() || o.block_ms <= 0.0 || o.block_ms > 2000.0 {
        bail!("--measure requires finite seconds >=0 and block-ms in (0,2000]");
    }
    let block = (o.block_ms * DEVICE_RATE as f32 / 1000.0).round() as usize;
    let lookahead = o.lookahead.checked_mul(DEVICE_RATE as usize / 100).ok_or_else(|| anyhow::anyhow!("lookahead too large"))?;
    println!("MEASURE: coded pulses replace voice conversion; {output:?} -> {input:?}. No models/GPU are loaded.");
    println!("RTT uses output/capture callback sample clocks and includes device/driver buffering; this is not a hardware-clock instrument.");
    let mut session = LatencyMeasurement::start(&input, &output, block, lookahead)?;
    let (stop_tx, stop_rx) = std::sync::mpsc::channel();
    if seconds == 0.0 {
        println!("Press Enter to stop.");
        std::thread::spawn(move || {
            let mut line = String::new();
            if matches!(std::io::stdin().lock().read_line(&mut line), Ok(n) if n > 0) { let _ = stop_tx.send(()); }
        });
    }
    let begin = Instant::now();
    let mut report = Instant::now();
    let mut rtts = Vec::new();
    while session.running() {
        while let Ok(event) = session.round_trips().try_recv() { rtts.push(event.rtt_ms); }
        while let Ok(block) = session.blocks().try_recv() {
            println!("measure block={} algorithm={:.2}ms (block+lookahead) processing={:.4}ms out_queue_ahead={:.2}ms", block.index, block.algorithm_ms, block.processing_ms, block.output_queue_ms);
        }
        if report.elapsed() >= Duration::from_secs(1) {
            let totals = session.totals();
            let mut sorted = rtts.clone(); sorted.sort_by(f32::total_cmp);
            if sorted.is_empty() {
                println!("measure t={:.1}s RTT=no valid pulse valid=0 sent={} timeout={} pending={} dropped={} under={} over={}", begin.elapsed().as_secs_f32(), totals.sent, totals.timeouts, totals.pending, totals.dropped, totals.underruns, totals.overruns);
            } else {
                println!("measure t={:.1}s RTT median={:.3}ms p95={:.3}ms valid={} sent={} timeout={} pending={} dropped={} under={} over={}", begin.elapsed().as_secs_f32(), percentile(&sorted,0.5), percentile(&sorted,0.95), sorted.len(), totals.sent, totals.timeouts, totals.pending, totals.dropped, totals.underruns, totals.overruns);
            }
            report = Instant::now();
        }
        if (seconds > 0.0 && begin.elapsed().as_secs_f32() >= seconds) || stop_rx.try_recv().is_ok() { break; }
        std::thread::sleep(Duration::from_millis(10));
    }
    let stopped = session.stop();
    while let Ok(event) = session.round_trips().try_recv() { rtts.push(event.rtt_ms); }
    rtts.sort_by(f32::total_cmp);
    let totals = session.totals();
    if rtts.is_empty() {
        println!("MEASURE final: RTT unavailable; valid=0 sent={} timeout={} pending={} dropped={}", totals.sent, totals.timeouts, totals.pending, totals.dropped);
    } else {
        println!("MEASURE final: RTT median={:.3}ms p95={:.3}ms valid={} sent={} timeout={} pending={} dropped={}", percentile(&rtts,0.5), percentile(&rtts,0.95), rtts.len(), totals.sent, totals.timeouts, totals.pending, totals.dropped);
    }
    println!("callback sizes: input={:.3}ms output={:.3}ms; algorithm/processing/queue above are NOT included in RTT. Processing here is probe generation, not neural inference.", totals.input_callback_ms, totals.output_callback_ms);
    stopped
}

#[allow(clippy::too_many_arguments)]
fn gate_probe(
    file: Option<PathBuf>, input: Option<String>, seconds: f32, save: Option<PathBuf>, block_ms: f32, gate_db: f32, vad: f32, denoise: bool,
    verbose: bool,
) -> Result<()> {
    let audio = match &file {
        Some(p) => wav::read_mono_at(p, DEVICE_RATE)?,
        None => {
            println!("recording {seconds:.0} s from {} ...", input.as_deref().unwrap_or("the default input"));
            let a = record_input(input.as_deref(), seconds, |t, db| {
                print!("\r[{t:5.1}s] {db:6.1} dBFS   ");
                let _ = std::io::stdout().flush();
            })?;
            println!();
            a
        }
    };
    if audio.is_empty() {
        bail!("no audio");
    }
    if let Some(p) = &save {
        wav::write_mono(p, &audio, DEVICE_RATE)?;
        println!("saved {} ({:.1} s)", p.display(), audio.len() as f32 / DEVICE_RATE as f32);
    }
    let block = (DEVICE_RATE as f32 * block_ms / 1000.0).round() as usize;
    let vad_path = repo_dir("models").join("silero_vad.onnx");
    let model = if vad_path.exists() { Some(SileroVad::load(ort_env(&repo_dir("runtime"), Device::Cpu)?, &vad_path)?) } else { None };
    if model.is_none() {
        println!("no voice model at {} -> level-only gate", vad_path.display());
    }
    let mut gate = Gate::new(block_ms, model);
    let mut down = StreamResampler::new(DEVICE_RATE, 16_000);
    println!("gate {gate_db:.0} dB, voice {vad:.2}, block {block_ms:.0} ms, denoise {}", if denoise { "on" } else { "off" });
    if verbose {
        println!("{:>8} {:>9} {:>6} {:>7}  decision", "t", "level", "voice", "speech");
    }
    let (mut n_open, mut n_hold, mut n_quiet, mut n_noise) = (0u32, 0u32, 0u32, 0u32);
    let mut timeline = String::new();
    for (i, chunk) in audio.chunks(block).enumerate() {
        let mut buf = chunk.to_vec();
        buf.resize(block, 0.0);
        if denoise {
            gate.denoise(&mut buf);
        }
        let a16 = down.process(&buf);
        let r = gate.decide(&a16, gate_db, vad)?;
        let (what, mark) = match (r.open, r.loud, r.speech) {
            (true, true, true) => {
                n_open += 1;
                ("OPEN", '#')
            }
            (true, _, _) => {
                n_hold += 1;
                ("open (hold)", '+')
            }
            (false, false, _) => {
                n_quiet += 1;
                ("closed: quiet", '.')
            }
            (false, true, _) => {
                n_noise += 1;
                ("closed: loud but not voice", 'x')
            }
        };
        timeline.push(mark);
        if verbose {
            println!("{:7.2}s {:7.1} dB {:6.2} {:4.0} ms  {}", i as f32 * block_ms / 1000.0, dbfs(r.level), r.vad, r.speech_ms, what);
        }
    }
    let per_line = (10_000.0 / block_ms).round() as usize; // 10 s per line
    println!("timeline (# open, + hold, . quiet, x loud-but-not-voice; one line = 10 s):");
    for (i, line) in timeline.as_bytes().chunks(per_line.max(1)).enumerate() {
        println!("{:4}s {}", i * 10, std::str::from_utf8(line).unwrap_or(""));
    }
    println!(
        "blocks: {n_open} open, {n_hold} hold, {n_quiet} quiet, {n_noise} loud-but-not-voice (those would have passed a level-only gate)"
    );
    Ok(())
}

fn prepare(wav_dir: PathBuf, out_dir: PathBuf, cpu: bool, loudness: Option<f32>) -> Result<()> {
    let opts = PrepareOptions { device: if cpu { Device::Cpu } else { Device::DirectML }, loudness_db: loudness };
    let t = Instant::now();
    let s = engine::prepare::prepare(&wav_dir, &out_dir, &repo_dir("models"), &repo_dir("runtime"), &opts, |m| println!("{m}"))?;
    println!(
        "{} files ({:.1} min) -> {} slices ({:.1} min), {} retrieval vectors, {} skipped, {:.0} s on {:?} -> {}",
        s.files, s.source_seconds / 60.0, s.slices, s.sliced_seconds / 60.0, s.retrieval_vectors, s.skipped,
        t.elapsed().as_secs_f32(), opts.device, out_dir.display()
    );
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Convert { input, output, engine } => convert(input, output, engine),
        Cmd::Devices => devices(),
        Cmd::Live { input, output, in_file, record, seconds, prime_ms, monitor, measure, engine } => {
            live(input, output, in_file, record, seconds, prime_ms, monitor, measure, engine)
        }
        Cmd::Gate { file, input, seconds, save, block_ms, gate_db, vad, no_denoise, verbose } => {
            gate_probe(file, input, seconds, save, block_ms, gate_db, vad, !no_denoise, verbose)
        }
        Cmd::Prepare { wav_dir, out_dir, cpu, loudness } => prepare(wav_dir, out_dir, cpu, loudness),
    }
}
