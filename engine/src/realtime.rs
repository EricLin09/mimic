//! Realtime path: WASAPI capture -> lock-free ring -> worker thread (BlockProcessor) ->
//! lock-free ring -> WASAPI render. Audio callbacks only move samples; everything else
//! (inference, telemetry, parameter changes) happens on the worker.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rtrb::{Consumer, RingBuffer};

use crate::dsp::{percentile, rms};
use crate::pipeline::DEVICE_RATE;
use crate::wav;
use crate::{BlockProcessor, Params};

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub input: bool,
    pub output: bool,
    pub default_rate: u32,
    pub channels: u16,
    pub default_input: bool,
    pub default_output: bool,
    /// for outputs: a 48 kHz stream builds right now (false = busy or format refused)
    pub output_ok: bool,
}

fn dev_name(d: &cpal::Device) -> String {
    d.description().map(|x| x.name().to_string()).unwrap_or_default()
}

pub fn list_devices() -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let din = host.default_input_device().map(|d| dev_name(&d));
    let dout = host.default_output_device().map(|d| dev_name(&d));
    let mut out = Vec::new();
    for d in host.devices()? {
        let name = dev_name(&d);
        let ic = d.default_input_config().ok();
        let oc = d.default_output_config().ok();
        let cfg = ic.as_ref().or(oc.as_ref());
        let output_ok = oc.as_ref().is_some_and(|c| output_opens(&d, c.channels()));
        out.push(DeviceInfo {
            default_input: din.as_deref() == Some(name.as_str()),
            default_output: dout.as_deref() == Some(name.as_str()),
            input: ic.is_some(),
            output: oc.is_some(),
            default_rate: cfg.map(|c| c.sample_rate()).unwrap_or(0),
            channels: cfg.map(|c| c.channels()).unwrap_or(0),
            output_ok,
            name,
        });
    }
    Ok(out)
}

/// Whether a session could open this output right now: builds the 48 kHz stream it would use and
/// drops it unplayed. Fails while another program holds the device exclusively, while VB-Cable's
/// sibling endpoint streams (its 16ch and stereo inputs share one cable), or when the format is
/// refused.
fn output_opens(d: &cpal::Device, channels: u16) -> bool {
    let cfg = cpal::StreamConfig { channels, sample_rate: DEVICE_RATE, buffer_size: cpal::BufferSize::Default };
    d.build_output_stream::<f32, _, _>(cfg, |data, _| data.fill(0.0), |_| {}, None).is_ok()
}

/// Case-insensitive substring match; an exact match wins.
fn find_device(host: &cpal::Host, wanted: Option<&str>, input: bool) -> Result<cpal::Device> {
    let Some(wanted) = wanted else {
        return (if input { host.default_input_device() } else { host.default_output_device() })
            .ok_or_else(|| anyhow!("no default {} device", if input { "input" } else { "output" }));
    };
    let devices: Vec<cpal::Device> = if input { host.input_devices()?.collect() } else { host.output_devices()?.collect() };
    let names: Vec<String> = devices.iter().map(dev_name).collect();
    if let Some(i) = names.iter().position(|n| n == wanted) {
        return Ok(devices.into_iter().nth(i).unwrap());
    }
    let lower = wanted.to_lowercase();
    let hits: Vec<usize> = names.iter().enumerate().filter(|(_, n)| n.to_lowercase().contains(&lower)).map(|(i, _)| i).collect();
    match hits.as_slice() {
        [i] => Ok(devices.into_iter().nth(*i).unwrap()),
        [] => bail!("no {} device matches {wanted:?}; available: {names:?}", if input { "input" } else { "output" }),
        many => bail!("{wanted:?} matches several devices: {:?}", many.iter().map(|&i| &names[i]).collect::<Vec<_>>()),
    }
}

#[derive(Clone, Debug)]
pub struct Telemetry {
    pub silent: bool,
    /// highest Silero VAD chunk probability in the block (0 without the model)
    pub vad: f32,
    pub input_level: f32,
    pub output_level: f32,
    pub f0_in: f32,
    pub f0_out: f32,
    pub proc_ms: f32,
    pub rtf: f32,
    /// block + lookahead + processing + samples queued for the output device
    pub latency_ms: f32,
    pub underruns: u64,
    pub wave_in: Vec<f32>,
    pub wave_out: Vec<f32>,
}

#[derive(Default)]
struct Counters {
    underruns: AtomicU64,
    overruns: AtomicU64,
    /// samples waiting in the output ring, as seen by the last output callback
    out_queued: AtomicU64,
    /// set once the first processed block was queued; underruns before that are expected
    started: AtomicBool,
    /// RMS of the last capture / render callback buffer (f32 bits), for smooth UI meters
    in_level: AtomicU32,
    out_level: AtomicU32,
}

fn store_level(slot: &AtomicU32, x: &[f32]) {
    slot.store(rms(x).to_bits(), Ordering::Relaxed);
}

#[derive(Clone, Debug, Default)]
pub struct LiveConfig {
    pub input: Option<String>,
    pub output: Option<String>,
    /// feed this WAV as the microphone, paced in real time (testing)
    pub input_file: Option<PathBuf>,
    /// write the processed output here on stop
    pub record: Option<PathBuf>,
    /// zeros queued ahead of the first block; None = p95 warm-up time + 10 ms
    pub prime_ms: Option<f32>,
    /// second output device that plays the same blocks, for hearing yourself while the main
    /// output feeds a virtual microphone; None opens no monitor stream
    pub monitor: Option<String>,
    /// whether the monitor starts audible; `set_monitor` toggles it live
    pub monitor_on: bool,
}

pub struct LiveSession {
    running: Arc<AtomicBool>,
    monitor_on: Arc<AtomicBool>,
    params: Arc<Mutex<Params>>,
    counters: Arc<Counters>,
    telemetry: Receiver<Telemetry>,
    worker: Option<JoinHandle<Result<()>>>,
    audio: Option<JoinHandle<Result<()>>>,
    feeder: Option<JoinHandle<()>>,
    _timer: TimerResolution,
    pub block_ms: f32,
    pub lookahead_ms: f32,
    pub prime_ms: f32,
    pub warmup_ms: Vec<f32>,
    /// why the monitor output was skipped, when one was asked for and would not open
    pub monitor_warning: Option<String>,
}

/// 1 ms scheduler granularity while a session runs; Windows defaults to 15.6 ms, which turns
/// the worker's 1 ms polling sleep into a 15 ms one.
struct TimerResolution;

#[cfg(windows)]
impl TimerResolution {
    fn acquire() -> Self {
        #[link(name = "winmm")]
        unsafe extern "system" {
            fn timeBeginPeriod(period: u32) -> u32;
        }
        unsafe { timeBeginPeriod(1) };
        TimerResolution
    }
}
#[cfg(windows)]
impl Drop for TimerResolution {
    fn drop(&mut self) {
        #[link(name = "winmm")]
        unsafe extern "system" {
            fn timeEndPeriod(period: u32) -> u32;
        }
        unsafe { timeEndPeriod(1) };
    }
}
#[cfg(not(windows))]
impl TimerResolution {
    fn acquire() -> Self {
        TimerResolution
    }
}

#[cfg(windows)]
fn pro_audio_priority() {
    #[link(name = "avrt")]
    unsafe extern "system" {
        fn AvSetMmThreadCharacteristicsW(task_name: *const u16, task_index: *mut u32) -> isize;
    }
    let name: Vec<u16> = "Pro Audio".encode_utf16().chain(std::iter::once(0)).collect();
    let mut index = 0u32;
    // a null handle only means we keep the normal priority
    unsafe { AvSetMmThreadCharacteristicsW(name.as_ptr(), &mut index) };
}
#[cfg(not(windows))]
fn pro_audio_priority() {}

fn downsample_wave(x: &[f32], n: usize) -> Vec<f32> {
    (0..n).map(|i| x[(i * x.len()) / n.max(1)]).collect()
}

/// Output callback body: pops one mono sample per frame of `data` into `mono` (grown only on the
/// first oversized callback), zero-fills whatever the ring could not supply and copies the result
/// to every channel. Returns (frames, samples that were missing).
fn render_mono(data: &mut [f32], channels: usize, mono: &mut Vec<f32>, ring: &mut Consumer<f32>) -> (usize, usize) {
    let frames = data.len() / channels;
    if mono.len() < frames {
        mono.resize(frames, 0.0);
    }
    let (_, missing) = ring.pop_partial_slice(&mut mono[..frames]);
    let short = missing.len();
    missing.fill(0.0);
    for (f, frame) in data.chunks_mut(channels).enumerate() {
        frame.fill(mono[f]);
    }
    (frames, short)
}

impl LiveSession {
    pub fn start(mut proc: BlockProcessor, cfg: LiveConfig) -> Result<Self> {
        let timer = TimerResolution::acquire();
        let block = proc.block_size();
        let block_ms = block as f32 * 1000.0 / DEVICE_RATE as f32;
        let lookahead_ms = proc.lookahead_samples() as f32 * 1000.0 / DEVICE_RATE as f32;

        // Warm-up: the first DirectML run compiles shaders (seconds); do it before audio starts and
        // size the output cushion from the steady-state time. The gate is bypassed (its voice test
        // would reject the noise and never reach the GPU), then everything the noise touched is reset.
        let mut warmup_ms = Vec::new();
        let noise: Vec<f32> = (0..block).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.02).collect();
        for _ in 0..4 {
            let s = proc.warm_up(&noise)?;
            warmup_ms.push(s.ms_total);
        }
        proc.reset();
        let mut sorted = warmup_ms[1..].to_vec();
        sorted.sort_by(f32::total_cmp);
        let (p95, median) = (percentile(&sorted, 0.95), percentile(&sorted, 0.5));
        // The cushion only has to absorb processing-time jitter, not the processing time itself.
        let prime_ms = cfg.prime_ms.unwrap_or((p95 - median + 25.0).max(30.0));
        let prime = (prime_ms / 1000.0 * DEVICE_RATE as f32) as usize;

        let capacity = block * 8 + DEVICE_RATE as usize; // a second of slack either side
        let (in_prod, mut in_cons) = RingBuffer::<f32>::new(capacity);
        let (mut out_prod, mut out_cons) = RingBuffer::<f32>::new(capacity + prime);

        let running = Arc::new(AtomicBool::new(true));
        let params = Arc::new(Mutex::new(proc.params.clone()));
        let counters = Arc::new(Counters::default());
        let (tx, telemetry) = mpsc::sync_channel::<Telemetry>(64);
        let (mut mon_prod, mon_cons) = RingBuffer::<f32>::new(capacity + prime);
        let monitor_on = Arc::new(AtomicBool::new(cfg.monitor_on));
        let monitor_w = monitor_on.clone();
        let mut mon_was_on = false;

        // ---- worker: pops whole blocks, converts, pushes output, reports
        let worker = {
            let running = running.clone();
            let params = params.clone();
            let counters = counters.clone();
            let record = cfg.record.clone();
            thread::Builder::new().name("mimic-worker".into()).spawn(move || -> Result<()> {
                pro_audio_priority();
                let mut buf = vec![0.0f32; block];
                let mut recorded: Vec<f32> = Vec::new();
                // Typical voiced processing time (EMA). Silent blocks skip inference and would be
                // queued ~50 ms earlier than voiced ones; holding them to the same timing keeps
                // the output ring level steady, so the cushion only has to absorb jitter.
                let mut typical_ms = median;
                let mut reprime = false;
                while running.load(Ordering::Relaxed) {
                    let pending = in_cons.slots();
                    if pending < block {
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    // After a stall (a shader compile, a device hiccup) the input ring holds a
                    // backlog. Converting all of it would queue it ahead of fresh audio and leave
                    // the latency raised for good: keep only the newest block, count the rest as
                    // lost, and put the cushion back since the output ring has drained meanwhile.
                    if pending >= 3 * block {
                        let drop = (pending / block - 1) * block;
                        let mut sink = vec![0.0f32; drop];
                        in_cons.pop_entire_slice(&mut sink).expect("checked slots");
                        counters.overruns.fetch_add((drop / block) as u64, Ordering::Relaxed);
                        reprime = true;
                        mon_was_on = false;
                    }
                    let arrived = Instant::now();
                    in_cons.pop_entire_slice(&mut buf).expect("checked slots");
                    if let Ok(p) = params.lock() {
                        proc.params = p.clone();
                    }
                    let (out, stats) = proc.process(&buf)?;
                    if stats.silent {
                        let target = Duration::from_secs_f32(typical_ms / 1000.0);
                        if let Some(wait) = target.checked_sub(arrived.elapsed()) {
                            thread::sleep(wait);
                        }
                    } else {
                        typical_ms += (stats.ms_total - typical_ms) * 0.1;
                    }
                    if !counters.started.load(Ordering::Relaxed) || reprime {
                        // the cushion goes in right before the first block, so it is still there
                        // when the second block's processing time wobbles; after a catch-up the
                        // drained ring gets it again
                        let zeros = vec![0.0f32; prime];
                        let _ = out_prod.push_partial_slice(&zeros);
                        reprime = false;
                    }
                    let (_, rest) = out_prod.push_partial_slice(&out);
                    if !rest.is_empty() {
                        counters.overruns.fetch_add(1, Ordering::Relaxed);
                    }
                    // the monitor gets the same cushion when it switches on, then the same blocks
                    let mon_on = monitor_w.load(Ordering::Relaxed);
                    if mon_on {
                        if !mon_was_on {
                            let _ = mon_prod.push_partial_slice(&vec![0.0f32; prime]);
                        }
                        let _ = mon_prod.push_partial_slice(&out);
                    }
                    mon_was_on = mon_on;
                    counters.started.store(true, Ordering::Relaxed);
                    if record.is_some() {
                        recorded.extend_from_slice(&out);
                    }
                    let queued_ms = counters.out_queued.load(Ordering::Relaxed) as f32 * 1000.0 / DEVICE_RATE as f32;
                    let t = Telemetry {
                        silent: stats.silent,
                        vad: stats.vad,
                        input_level: stats.level,
                        output_level: rms(&out),
                        f0_in: stats.f0_in,
                        f0_out: stats.f0_out,
                        proc_ms: stats.ms_total,
                        rtf: stats.ms_total / block_ms,
                        latency_ms: block_ms + lookahead_ms + stats.ms_total + queued_ms,
                        underruns: counters.underruns.load(Ordering::Relaxed),
                        wave_in: downsample_wave(&buf, 256),
                        wave_out: downsample_wave(&out, 256),
                    };
                    if let Err(TrySendError::Disconnected(_)) = tx.try_send(t) {
                        break;
                    }
                }
                if let Some(path) = record {
                    wav::write_mono(&path, &recorded, DEVICE_RATE)?;
                }
                Ok(())
            })?
        };

        // ---- feeder (file input) or capture stream: exactly one of them owns the input producer
        let mut in_prod = Some(in_prod);
        let feeder = match &cfg.input_file {
            Some(path) => {
                let audio = wav::read_mono_at(path, DEVICE_RATE)?;
                let running = running.clone();
                let counters = counters.clone();
                let mut in_prod = in_prod.take().expect("input producer");
                Some(thread::Builder::new().name("mimic-feeder".into()).spawn(move || {
                    let step = Duration::from_micros(10_000); // 10 ms slices, like a device callback
                    let n = (DEVICE_RATE / 100) as usize;
                    let start = Instant::now();
                    let mut i = 0usize;
                    while running.load(Ordering::Relaxed) && (i + 1) * n <= audio.len() {
                        let due = start + step * i as u32;
                        if let Some(d) = due.checked_duration_since(Instant::now()) {
                            thread::sleep(d);
                        }
                        let (_, rest) = in_prod.push_partial_slice(&audio[i * n..(i + 1) * n]);
                        if !rest.is_empty() {
                            counters.overruns.fetch_add(1, Ordering::Relaxed);
                        }
                        i += 1;
                    }
                })?)
            }
            None => None,
        };

        // ---- audio owner thread: cpal streams live and die here (cpal::Stream is not Send)
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(usize, usize, Option<String>)>>();
        let audio = {
            let running = running.clone();
            let counters = counters.clone();
            let (input_name, output_name) = (cfg.input.clone(), cfg.output.clone());
            let mut in_prod = in_prod.take();
            let monitor_name = cfg.monitor.clone();
            let mut mon_cons = Some(mon_cons);
            thread::Builder::new().name("mimic-audio".into()).spawn(move || -> Result<()> {
                let host = cpal::default_host();
                #[allow(clippy::type_complexity)]
                let build = || -> Result<(cpal::Stream, Option<cpal::Stream>, Option<cpal::Stream>, Option<String>, usize, usize)> {
                    let out_dev = find_device(&host, output_name.as_deref(), false)?;
                    let out_sup = out_dev.default_output_config()?;
                    let out_ch = out_sup.channels() as usize;
                    let out_cfg = cpal::StreamConfig { channels: out_sup.channels(), sample_rate: DEVICE_RATE, buffer_size: cpal::BufferSize::Default };
                    let c1 = counters.clone();
                    let mut mono = vec![0.0f32; 4096];
                    let out_stream = out_dev
                        .build_output_stream::<f32, _, _>(
                            out_cfg,
                            move |data, _| {
                                let (frames, missing) = render_mono(data, out_ch, &mut mono, &mut out_cons);
                                if missing > 0 && c1.started.load(Ordering::Relaxed) {
                                    c1.underruns.fetch_add(1, Ordering::Relaxed);
                                }
                                store_level(&c1.out_level, &mono[..frames]);
                                c1.out_queued.store(out_cons.slots() as u64, Ordering::Relaxed);
                            },
                            {
                                // cpal reports a one-off xrun when a WASAPI stream starts; say it once
                                let warned = AtomicBool::new(false);
                                move |e| {
                                    if !warned.swap(true, Ordering::Relaxed) {
                                        eprintln!("output stream: {e}");
                                    }
                                }
                            },
                            None,
                        )
                        .with_context(|| format!("opening output {:?} at 48 kHz", dev_name(&out_dev)))?;
                    out_stream.play()?;

                    // optional second output for hearing yourself: silence is its normal state
                    // while switched off, so an empty ring is not an underrun here. It must never
                    // take the session down: a monitor that will not open is reported and skipped.
                    let mut mon_warn: Option<String> = None;
                    let mon_stream = match (monitor_name.as_deref(), mon_cons.take()) {
                        (Some(name), Some(mut cons)) => {
                            let host = &host;
                            let open = move || -> Result<cpal::Stream> {
                            let dev = find_device(host, Some(name), false)?;
                            let sup = dev.default_output_config()?;
                            let ch = sup.channels() as usize;
                            let cfg = cpal::StreamConfig { channels: sup.channels(), sample_rate: DEVICE_RATE, buffer_size: cpal::BufferSize::Default };
                            let mut mono = vec![0.0f32; 4096];
                            let s = dev
                                .build_output_stream::<f32, _, _>(
                                    cfg,
                                    move |data, _| {
                                        render_mono(data, ch, &mut mono, &mut cons);
                                    },
                                    |_| {},
                                    None,
                                )
                                .with_context(|| format!("opening monitor {:?} at 48 kHz", dev_name(&dev)))?;
                            s.play()?;
                            Ok(s)
                            };
                            match open() {
                                Ok(s) => Some(s),
                                Err(e) => {
                                    eprintln!("monitor skipped: {e:#}");
                                    mon_warn = Some(format!("{e:#}"));
                                    None
                                }
                            }
                        }
                        _ => None,
                    };

                    let mut in_ch = 0;
                    let in_stream = if let Some(mut prod) = in_prod.take() {
                        let in_dev = find_device(&host, input_name.as_deref(), true)?;
                        let in_sup = in_dev.default_input_config()?;
                        in_ch = in_sup.channels() as usize;
                        let ch = in_ch;
                        let in_cfg = cpal::StreamConfig { channels: in_sup.channels(), sample_rate: DEVICE_RATE, buffer_size: cpal::BufferSize::Default };
                        let c2 = counters.clone();
                        let mut mono = vec![0.0f32; 4096];
                        let s = in_dev
                            .build_input_stream::<f32, _, _>(
                                in_cfg,
                                move |data, _| {
                                    let frames = data.len() / ch;
                                    if mono.len() < frames {
                                        mono.resize(frames, 0.0);
                                    }
                                    for (f, frame) in data.chunks(ch).enumerate() {
                                        mono[f] = frame.iter().sum::<f32>() / ch as f32;
                                    }
                                    store_level(&c2.in_level, &mono[..frames]);
                                    let (_, rest) = prod.push_partial_slice(&mono[..frames]);
                                    if !rest.is_empty() {
                                        c2.overruns.fetch_add(1, Ordering::Relaxed);
                                    }
                                },
                                {
                                    let warned = AtomicBool::new(false);
                                    move |e| {
                                        if !warned.swap(true, Ordering::Relaxed) {
                                            eprintln!("input stream: {e}");
                                        }
                                    }
                                },
                                None,
                            )
                            .with_context(|| format!("opening input {:?} at 48 kHz", dev_name(&in_dev)))?;
                        s.play()?;
                        Some(s)
                    } else {
                        None
                    };
                    Ok((out_stream, in_stream, mon_stream, mon_warn, in_ch, out_ch))
                };
                match build() {
                    Ok((out_stream, in_stream, mon_stream, mon_warn, in_ch, out_ch)) => {
                        let _ = ready_tx.send(Ok((in_ch, out_ch, mon_warn)));
                        while running.load(Ordering::Relaxed) {
                            thread::sleep(Duration::from_millis(20));
                        }
                        drop(in_stream);
                        drop(mon_stream);
                        drop(out_stream);
                        Ok(())
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(anyhow!("{e:#}")));
                        Ok(())
                    }
                }
            })?
        };
        let monitor_warning = match ready_rx.recv_timeout(Duration::from_secs(15)) {
            Ok(Ok((_, _, warn))) => warn,
            Ok(Err(e)) => {
                running.store(false, Ordering::Relaxed);
                return Err(e);
            }
            Err(_) => {
                running.store(false, Ordering::Relaxed);
                bail!("audio devices did not start within 15 s");
            }
        };

        Ok(Self {
            running, monitor_on, params, counters, telemetry,
            worker: Some(worker), audio: Some(audio), feeder, _timer: timer,
            block_ms, lookahead_ms, prime_ms, warmup_ms, monitor_warning,
        })
    }

    /// Mute or unmute the monitor output without touching the main one.
    pub fn set_monitor(&self, on: bool) {
        self.monitor_on.store(on, Ordering::Relaxed);
    }

    pub fn set_params(&self, p: Params) {
        if let Ok(mut g) = self.params.lock() {
            *g = p;
        }
    }

    pub fn telemetry(&self) -> &Receiver<Telemetry> {
        &self.telemetry
    }

    pub fn underruns(&self) -> u64 {
        self.counters.underruns.load(Ordering::Relaxed)
    }

    /// (input, output) RMS of the most recent device callbacks; updates every ~10 ms.
    pub fn levels(&self) -> (f32, f32) {
        (
            f32::from_bits(self.counters.in_level.load(Ordering::Relaxed)),
            f32::from_bits(self.counters.out_level.load(Ordering::Relaxed)),
        )
    }

    pub fn overruns(&self) -> u64 {
        self.counters.overruns.load(Ordering::Relaxed)
    }

    /// True while a file feeder is still delivering audio.
    pub fn feeding(&self) -> bool {
        self.feeder.as_ref().is_some_and(|f| !f.is_finished())
    }

    pub fn stop(mut self) -> Result<()> {
        self.running.store(false, Ordering::Relaxed);
        if let Some(f) = self.feeder.take() {
            let _ = f.join();
        }
        if let Some(a) = self.audio.take() {
            let _ = a.join();
        }
        if let Some(w) = self.worker.take() {
            w.join().map_err(|_| anyhow!("worker panicked"))??;
        }
        Ok(())
    }
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

/// Capture `seconds` of mono 48 kHz audio from an input device (the default one when `name` is
/// None), calling `tick` about twice a second with (elapsed s, level dBFS of the last 0.5 s).
/// Diagnostics only: no models, no output stream.
pub fn record_input(name: Option<&str>, seconds: f32, mut tick: impl FnMut(f32, f32)) -> Result<Vec<f32>> {
    let host = cpal::default_host();
    let dev = find_device(&host, name, true)?;
    let sup = dev.default_input_config()?;
    let ch = sup.channels() as usize;
    let cfg = cpal::StreamConfig { channels: sup.channels(), sample_rate: DEVICE_RATE, buffer_size: cpal::BufferSize::Default };
    let want = (DEVICE_RATE as f32 * seconds) as usize;
    let buf = Arc::new(Mutex::new(Vec::<f32>::with_capacity(want + DEVICE_RATE as usize)));
    let sink = buf.clone();
    let stream = dev
        .build_input_stream::<f32, _, _>(
            cfg,
            move |data, _| {
                let mut b = sink.lock().unwrap();
                b.extend(data.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32));
            },
            |_| {},
            None,
        )
        .with_context(|| format!("opening input {:?} at 48 kHz", dev_name(&dev)))?;
    stream.play()?;
    let start = Instant::now();
    loop {
        thread::sleep(Duration::from_millis(500));
        let (len, level) = {
            let b = buf.lock().unwrap();
            let tail = &b[b.len().saturating_sub(DEVICE_RATE as usize / 2)..];
            (b.len(), crate::dsp::rms(tail))
        };
        tick(start.elapsed().as_secs_f32(), 20.0 * level.max(1e-6).log10());
        if len >= want {
            break;
        }
    }
    drop(stream);
    let mut v = std::mem::take(&mut *buf.lock().unwrap());
    v.truncate(want);
    Ok(v)
}

/// Explicit diagnostic session; normal LiveSession/default conversion is untouched.
pub use latency_measurement::{LatencyMeasurement, MeasurementBlock, MeasurementTotals, RoundTrip};

mod latency_measurement {
    use super::{DEVICE_RATE, TimerResolution, find_device, pro_audio_priority};
    use crate::probe::{PROBE_LEN, ProbeDetector, ProbeSource};
    use anyhow::{Context, Result, anyhow, bail};
    use cpal::traits::{DeviceTrait, StreamTrait};
    use rtrb::{Consumer, Producer, RingBuffer};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
    use std::sync::mpsc::{self, Receiver};
    use std::thread::{self, JoinHandle};
    use std::time::{Duration, Instant};

    const TIMEOUT_NS: u64 = 1_500_000_000;
    const SAMPLE_NS: u64 = 1_000_000_000;
    #[derive(Clone, Copy, Debug, Default)]
    struct Frame {
        sample: f32,
        marker: u64,
    }
    #[derive(Clone, Copy, Debug)]
    pub struct RoundTrip {
        pub probe_id: u64,
        pub rtt_ms: f32,
    }
    #[derive(Clone, Copy, Debug)]
    pub struct MeasurementBlock {
        pub index: u64,
        pub algorithm_ms: f32,
        pub processing_ms: f32,
        /// Samples already queued AHEAD of this block, not this block itself.
        pub output_queue_ms: f32,
    }
    #[derive(Clone, Copy, Debug)]
    pub struct MeasurementTotals {
        pub sent: u64,
        pub timeouts: u64,
        pub pending: u64,
        pub dropped: u64,
        pub underruns: u64,
        pub overruns: u64,
        pub input_callback_ms: f32,
        pub output_callback_ms: f32,
    }
    #[derive(Default)]
    struct State {
        pending_id: AtomicU64,
        sent_ns: AtomicU64,
        sent: AtomicU64,
        timeouts: AtomicU64,
        dropped: AtomicU64,
        underruns: AtomicU64,
        overruns: AtomicU64,
        input_frames: AtomicU64,
        output_frames: AtomicU64,
        started: AtomicBool,
        fatal: AtomicU32,
    }
    impl State {
        fn arm(&self, id: u64, at: u64) {
            if self.pending_id.swap(0, Ordering::AcqRel) != 0 {
                self.timeouts.fetch_add(1, Ordering::Relaxed);
            }
            self.sent_ns.store(at, Ordering::Relaxed);
            self.pending_id.store(id, Ordering::Release);
            self.sent.fetch_add(1, Ordering::Relaxed);
        }
        fn expire(&self, now: u64) {
            let id = self.pending_id.load(Ordering::Acquire);
            let sent = self.sent_ns.load(Ordering::Relaxed);
            if id != 0
                && now.saturating_sub(sent) >= TIMEOUT_NS
                && self
                    .pending_id
                    .compare_exchange(id, 0, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
            {
                self.timeouts.fetch_add(1, Ordering::Relaxed);
            }
        }
        fn detect(&self, at: u64) -> Option<RoundTrip> {
            let id = self.pending_id.load(Ordering::Acquire);
            let sent = self.sent_ns.load(Ordering::Relaxed);
            if id == 0 || at < sent || at - sent >= TIMEOUT_NS {
                return None;
            }
            if self
                .pending_id
                .compare_exchange(id, 0, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                Some(RoundTrip {
                    probe_id: id,
                    rtt_ms: (at - sent) as f32 / 1e6,
                })
            } else {
                None
            }
        }
        fn error(&self, error: cpal::Error, running: &AtomicBool) {
            if matches!(
                error.kind(),
                cpal::ErrorKind::Xrun | cpal::ErrorKind::RealtimeDenied
            ) {
                return;
            }
            let code = if error.kind() == cpal::ErrorKind::DeviceNotAvailable {
                1
            } else {
                2
            };
            self.fatal.store(code, Ordering::Release);
            running.store(false, Ordering::Release);
        }
    }
    fn now_ns(origin: Instant) -> u64 {
        origin.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }
    fn offset_ns(frame: usize) -> u64 {
        frame as u64 * SAMPLE_NS / DEVICE_RATE as u64
    }

    // Shared by the real callback and tests. No allocation, lock, I/O or printing.
    fn render(
        data: &mut [f32],
        channels: usize,
        ring: &mut Consumer<Frame>,
        state: &State,
        callback_ns: u64,
    ) {
        let mut missing = false;
        for (index, frame) in data.chunks_exact_mut(channels).enumerate() {
            let value = match ring.pop() {
                Ok(value) => value,
                Err(_) => {
                    missing = true;
                    Frame::default()
                }
            };
            if value.marker != 0 {
                state.arm(value.marker, callback_ns + offset_ns(index));
            }
            frame.fill(value.sample);
        }
        if missing && state.started.load(Ordering::Relaxed) {
            state.underruns.fetch_add(1, Ordering::Relaxed);
        }
        state
            .output_frames
            .store((data.len() / channels) as u64, Ordering::Relaxed);
    }
    fn capture(
        data: &[f32],
        channels: usize,
        detector: &mut ProbeDetector,
        events: &mut Producer<RoundTrip>,
        state: &State,
        callback_ns: u64,
    ) {
        for (index, frame) in data.chunks_exact(channels).enumerate() {
            let mono = frame.iter().sum::<f32>() / channels as f32;
            if detector.push(mono) {
                // Rolling match completes at the last probe sample. Both sides
                // use callback-entry + sample offset, retaining driver buffering.
                let at = (callback_ns + offset_ns(index)).saturating_sub(offset_ns(PROBE_LEN - 1));
                if let Some(event) = state.detect(at) {
                    if events.push(event).is_err() {
                        state.dropped.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        state
            .input_frames
            .store((data.len() / channels) as u64, Ordering::Relaxed);
    }

    pub struct LatencyMeasurement {
        running: Arc<AtomicBool>,
        state: Arc<State>,
        round_trips: Receiver<RoundTrip>,
        blocks: Receiver<MeasurementBlock>,
        worker: Option<JoinHandle<Result<()>>>,
        audio: Option<JoinHandle<Result<()>>>,
        _timer: TimerResolution,
    }
    impl LatencyMeasurement {
        pub fn start(input: &str, output: &str, block: usize, lookahead: usize) -> Result<Self> {
            if input.trim().is_empty()
                || output.trim().is_empty()
                || block == 0
                || block > DEVICE_RATE as usize * 2
                || block.checked_add(lookahead).is_none()
            {
                bail!("measurement requires explicit input/output and a block in (0,2 seconds]");
            }
            let timer = TimerResolution::acquire();
            let state = Arc::new(State::default());
            let running = Arc::new(AtomicBool::new(true));
            let origin = Instant::now();
            let capacity = DEVICE_RATE as usize + block * 4;
            let (mut out_prod, mut out_cons) = RingBuffer::<Frame>::new(capacity);
            let (mut event_prod, mut event_cons) = RingBuffer::<RoundTrip>::new(64);
            let (event_tx, round_trips) = mpsc::sync_channel(64);
            let (block_tx, blocks) = mpsc::sync_channel(64);
            let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
            let (input, output) = (input.to_owned(), output.to_owned());
            let audio_state = state.clone();
            let audio_running = running.clone();
            let audio = thread::Builder::new()
                .name("mimic-measure-audio".into())
                .spawn(move || -> Result<()> {
                    let build = || -> Result<(cpal::Stream, cpal::Stream)> {
                        let host = cpal::default_host();
                        let out = find_device(&host, Some(&output), false)?;
                        let input = find_device(&host, Some(&input), true)?;
                        let (out_ch, in_ch) = (
                            out.default_output_config()?.channels(),
                            input.default_input_config()?.channels(),
                        );
                        let output_cfg = cpal::StreamConfig {
                            channels: out_ch,
                            sample_rate: DEVICE_RATE,
                            buffer_size: cpal::BufferSize::Default,
                        };
                        let input_cfg = cpal::StreamConfig {
                            channels: in_ch,
                            sample_rate: DEVICE_RATE,
                            buffer_size: cpal::BufferSize::Default,
                        };
                        let (s, e, active) = (
                            audio_state.clone(),
                            audio_state.clone(),
                            audio_running.clone(),
                        );
                        let out_stream = out
                            .build_output_stream::<f32, _, _>(
                                output_cfg,
                                move |data, _| {
                                    render(data, out_ch as usize, &mut out_cons, &s, now_ns(origin))
                                },
                                move |error| e.error(error, &active),
                                Some(Duration::from_secs(5)),
                            )
                            .context("opening measurement output at 48 kHz")?;
                        let (s, e, active) = (
                            audio_state.clone(),
                            audio_state.clone(),
                            audio_running.clone(),
                        );
                        let mut detector = ProbeDetector::default();
                        let in_stream = input
                            .build_input_stream::<f32, _, _>(
                                input_cfg,
                                move |data, _| {
                                    capture(
                                        data,
                                        in_ch as usize,
                                        &mut detector,
                                        &mut event_prod,
                                        &s,
                                        now_ns(origin),
                                    )
                                },
                                move |error| e.error(error, &active),
                                Some(Duration::from_secs(5)),
                            )
                            .context("opening measurement input at 48 kHz")?;
                        in_stream.play()?;
                        out_stream.play()?;
                        Ok((in_stream, out_stream))
                    };
                    match build() {
                        Ok((input, output)) => {
                            let _ = ready_tx.send(Ok(()));
                            while audio_running.load(Ordering::Acquire) {
                                thread::sleep(Duration::from_millis(5));
                            }
                            drop(input);
                            drop(output);
                            Ok(())
                        }
                        Err(error) => {
                            audio_running.store(false, Ordering::Release);
                            let _ = ready_tx.send(Err(error));
                            Ok(())
                        }
                    }
                })?;
            match ready_rx.recv_timeout(Duration::from_secs(15)) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    running.store(false, Ordering::Release);
                    let _ = audio.join();
                    return Err(error);
                }
                Err(_) => {
                    running.store(false, Ordering::Release);
                    bail!("measurement audio did not start within 15 seconds");
                }
            }
            let (worker_state, worker_running) = (state.clone(), running.clone());
            let worker = match thread::Builder::new()
                .name("mimic-measure-worker".into())
                .spawn(move || -> Result<()> {
                    pro_audio_priority();
                    let mut source = ProbeSource::new(DEVICE_RATE as u64 * 2);
                    let mut buffer = vec![Frame::default(); block];
                    let mut index = 0;
                    loop {
                        while let Ok(event) = event_cons.pop() {
                            if event_tx.try_send(event).is_err() {
                                worker_state.dropped.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        worker_state.expire(now_ns(origin));
                        if !worker_running.load(Ordering::Acquire) {
                            // Wait until capture's producer has gone away before the
                            // final drain, so shutdown does not lose its last event.
                            if event_cons.is_abandoned() && event_cons.slots() == 0 {
                                break;
                            }
                            thread::sleep(Duration::from_millis(1));
                            continue;
                        }
                        let free = out_prod.slots();
                        let queued = capacity - free;
                        let low_water = (worker_state.output_frames.load(Ordering::Relaxed)
                            as usize)
                            .max(DEVICE_RATE as usize / 100);
                        if queued >= low_water || free < block {
                            thread::sleep(Duration::from_millis(1));
                            continue;
                        }
                        let begin = Instant::now();
                        for frame in &mut buffer {
                            let next = source.next_sample();
                            *frame = Frame {
                                sample: next.sample,
                                marker: next.start_sample.map_or(0, |start| start + 1),
                            };
                        }
                        for frame in &buffer {
                            if out_prod.push(*frame).is_err() {
                                worker_state.overruns.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        worker_state.started.store(true, Ordering::Relaxed);
                        index += 1;
                        let _ = block_tx.try_send(MeasurementBlock {
                            index,
                            algorithm_ms: (block + lookahead) as f32 * 1000.0 / DEVICE_RATE as f32,
                            processing_ms: begin.elapsed().as_secs_f32() * 1000.0,
                            output_queue_ms: queued as f32 * 1000.0 / DEVICE_RATE as f32,
                        });
                    }
                    Ok(())
                }) {
                Ok(worker) => worker,
                Err(error) => {
                    running.store(false, Ordering::Release);
                    let _ = audio.join();
                    return Err(error.into());
                }
            };
            Ok(Self {
                running,
                state,
                round_trips,
                blocks,
                worker: Some(worker),
                audio: Some(audio),
                _timer: timer,
            })
        }
        pub fn running(&self) -> bool {
            self.running.load(Ordering::Acquire)
        }
        pub fn round_trips(&self) -> &Receiver<RoundTrip> {
            &self.round_trips
        }
        pub fn blocks(&self) -> &Receiver<MeasurementBlock> {
            &self.blocks
        }
        pub fn totals(&self) -> MeasurementTotals {
            MeasurementTotals {
                sent: self.state.sent.load(Ordering::Relaxed),
                timeouts: self.state.timeouts.load(Ordering::Relaxed),
                pending: u64::from(self.state.pending_id.load(Ordering::Acquire) != 0),
                dropped: self.state.dropped.load(Ordering::Relaxed),
                underruns: self.state.underruns.load(Ordering::Relaxed),
                overruns: self.state.overruns.load(Ordering::Relaxed),
                input_callback_ms: self.state.input_frames.load(Ordering::Relaxed) as f32 * 1000.0
                    / DEVICE_RATE as f32,
                output_callback_ms: self.state.output_frames.load(Ordering::Relaxed) as f32
                    * 1000.0
                    / DEVICE_RATE as f32,
            }
        }
        pub fn stop(&mut self) -> Result<()> {
            self.running.store(false, Ordering::Release);
            if let Some(audio) = self.audio.take() {
                audio
                    .join()
                    .map_err(|_| anyhow!("measurement audio thread panicked"))??;
            }
            if let Some(worker) = self.worker.take() {
                worker
                    .join()
                    .map_err(|_| anyhow!("measurement worker panicked"))??;
            }
            match self.state.fatal.load(Ordering::Acquire) {
                0 => Ok(()),
                1 => bail!("measurement device became unavailable"),
                _ => bail!("measurement audio backend failed"),
            }
        }
    }
    impl Drop for LatencyMeasurement {
        fn drop(&mut self) {
            self.running.store(false, Ordering::Release);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn measure_detects_across_callback_lengths_and_emits_once() {
            for output_length in [17, 128, 480, 1024] {
                for input_length in [13, 160, 512, 960] {
                    for channels in [1, 2] {
                        let state = State::default();
                        let (mut producer, mut consumer) = RingBuffer::new(10_000);
                        let (mut events, mut received) = RingBuffer::new(8);
                        let mut source = ProbeSource::new(DEVICE_RATE as u64 * 2);
                        for _ in 0..2048 {
                            let sample = source.next_sample();
                            producer
                                .push(Frame {
                                    sample: sample.sample,
                                    marker: sample.start_sample.map_or(0, |s| s + 1),
                                })
                                .unwrap();
                        }
                        let mut rendered = vec![0.0; 2048 * channels];
                        for (i, block) in rendered.chunks_mut(output_length * channels).enumerate()
                        {
                            render(
                                block,
                                channels,
                                &mut consumer,
                                &state,
                                1_000_000_000 + offset_ns(i * output_length),
                            );
                        }
                        let mut detector = ProbeDetector::default();
                        for (i, block) in rendered.chunks(input_length * channels).enumerate() {
                            capture(
                                block,
                                channels,
                                &mut detector,
                                &mut events,
                                &state,
                                1_030_000_000 + offset_ns(i * input_length),
                            );
                        }
                        let event = received.pop().expect("probe recognized");
                        assert_eq!(event.probe_id, 1);
                        assert!(
                            (event.rtt_ms - 30.0).abs() < 0.15,
                            "out={output_length} in={input_length} channels={channels} RTT={}",
                            event.rtt_ms
                        );
                        assert!(received.pop().is_err());
                        assert_eq!(state.pending_id.load(Ordering::Relaxed), 0);
                    }
                }
            }
        }
        #[test]
        fn measure_timeout_counts_once_and_late_matches_are_rejected() {
            let state = State::default();
            state.arm(1, 100);
            state.expire(100 + TIMEOUT_NS - 1);
            assert_eq!(state.timeouts.load(Ordering::Relaxed), 0);
            assert!(state.detect(100 + TIMEOUT_NS).is_none());
            state.expire(100 + TIMEOUT_NS);
            state.expire(100 + TIMEOUT_NS + 1);
            assert_eq!(state.timeouts.load(Ordering::Relaxed), 1);
            assert!(state.detect(100 + TIMEOUT_NS + 100).is_none());
            state.arm(2, 3_000_000_000);
            assert!(state.detect(2_999_999_999).is_none());
            assert_eq!(state.detect(3_020_000_000).unwrap().probe_id, 2);
        }
        #[test]
        fn measure_replacement_and_dropped_event_are_counted() {
            let state = State::default();
            state.arm(1, 1);
            state.arm(2, 2);
            assert_eq!(state.timeouts.load(Ordering::Relaxed), 1);
            assert_eq!(state.sent.load(Ordering::Relaxed), 2);
            let (mut events, _consumer) = RingBuffer::new(1);
            events
                .push(RoundTrip {
                    probe_id: 9,
                    rtt_ms: 1.0,
                })
                .unwrap();
            let mut detector = ProbeDetector::default();
            let signal: Vec<_> = (0..PROBE_LEN).map(crate::probe::probe_sample).collect();
            capture(&signal, 1, &mut detector, &mut events, &state, 30_000_000);
            assert_eq!(state.dropped.load(Ordering::Relaxed), 1);
            assert_eq!(state.pending_id.load(Ordering::Relaxed), 0);
        }
    }
}
