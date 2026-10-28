//! Tauri bridge to the Mimic engine: start/stop the live session, push parameter changes,
//! list devices, and stream telemetry to the UI as events.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};


use engine::{BlockProcessor, Device, Layout, LiveConfig, LiveSession, Params, Paths, list_devices};
use engine::gpu::{GpuMeter, MIN_SAMPLE_INTERVAL};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiParams {
    pub pitch: f32,
    pub index_rate: f32,
    pub protect: f32,
    pub block_ms: f32,
    /// input gate in dBFS
    pub gate_db: f32,
    pub denoise: bool,
    /// voice threshold 0..1 for the gate (0 = level only)
    pub vad: f32,
    /// follow the speaker's loudness contour 0..1 (0 = the model's own dynamics)
    #[serde(default)]
    pub envelope: f32,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct DeviceChoice {
    pub input: Option<String>,
    pub output: Option<String>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UiDevices {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// listed outputs that would not open right now (busy, format refused)
    pub unusable_outputs: Vec<String>,
    pub input: String,
    pub output: String,
    /// where the monitor plays when the main output is a virtual cable
    pub default_output: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiTelemetry {
    t: f64,
    input_level: f32,
    output_level: f32,
    f0_in: f32,
    f0_out: f32,
    latency_ms: f32,
    rtf: f32,
    /// Process contribution to its busiest GPU engine, 0..1; negative = unavailable.
    gpu: f32,
    underruns: u64,
    /// block RMS after denoising, dBFS, for the gate meter
    gate_db: f32,
    gate_open: bool,
    vad: f32,
    wave_in: Vec<f32>,
    wave_out: Vec<f32>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiState {
    status: String,
    error: Option<String>,
}

#[derive(Default)]
struct Shared {
    session: Option<LiveSession>,
    status: String,
    error: Option<String>,
    params: Option<UiParams>,
    voice: String,
    choice: DeviceChoice,
    monitor: bool,
    /// bumps on every start; the telemetry pump of an older session exits when it changes
    generation: u64,
}

pub struct EngineState(Arc<Mutex<Shared>>);

/// `models/` and `runtime/` live next to a packaged exe; a dev build falls back to the checkout.
fn repo(sub: &str) -> PathBuf {
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.to_path_buf())) {
        let beside = dir.join(sub);
        if beside.exists() {
            return beside;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join(sub)
}

fn set_status(app: &AppHandle, shared: &Arc<Mutex<Shared>>, status: &str, error: Option<String>) {
    match &error {
        Some(e) => eprintln!("[mimic] {status}: {e}"),
        None => eprintln!("[mimic] {status}"),
    }
    if let Ok(mut s) = shared.lock() {
        s.status = status.to_string();
        s.error = error.clone();
    }
    let _ = app.emit("engine-state", UiState { status: status.to_string(), error });
}

fn build_processor(voice: &str, p: &UiParams) -> anyhow::Result<BlockProcessor> {
    let paths = Paths { models: repo("models"), voice: voice.to_string(), runtime: repo("runtime") };
    let layout = Layout { block_ms: p.block_ms, ..Layout::default() };
    let params = Params { pitch: p.pitch, index_rate: p.index_rate, protect: p.protect, silence_db: p.gate_db, denoise_audio: p.denoise, vad: p.vad, envelope: p.envelope, ..Params::default() };
    BlockProcessor::load(&paths, Device::DirectML, &layout, params, true)
}

/// Pick sensible defaults: never a CABLE device for input unless chosen, system default output.
fn resolve_devices(choice: &DeviceChoice) -> anyhow::Result<UiDevices> {
    let all = list_devices()?;
    let inputs: Vec<String> = all.iter().filter(|d| d.input).map(|d| d.name.clone()).collect();
    // VB-Cable's 16-channel endpoint feeds the same cable as "CABLE Input" and cannot stream while
    // that one is in use: offer the stereo endpoint only
    let has_cable_input = all.iter().any(|d| d.output && d.name.starts_with("CABLE Input"));
    let outputs: Vec<String> = all
        .iter()
        .filter(|d| d.output && !(has_cable_input && d.name.starts_with("CABLE In 16ch")))
        .map(|d| d.name.clone())
        .collect();
    let unusable_outputs: Vec<String> =
        all.iter().filter(|d| d.output && !d.output_ok && outputs.contains(&d.name)).map(|d| d.name.clone()).collect();
    let usable: Vec<String> = outputs.iter().filter(|o| !unusable_outputs.contains(o)).cloned().collect();
    // the wanted device if listed, else the fallback if listed, else the first listed
    let pick = |want: &Option<String>, list: &[String], fallback: Option<String>| -> String {
        want.iter().chain(fallback.iter()).find(|w| list.contains(w)).cloned().unwrap_or_else(|| list.first().cloned().unwrap_or_default())
    };
    let default_in = all.iter().find(|d| d.default_input && d.input).map(|d| d.name.clone());
    let default_in = match default_in {
        Some(n) if !n.contains("CABLE") => Some(n),
        _ => inputs.iter().find(|n| !n.contains("CABLE")).cloned(),
    };
    let default_out = all.iter().find(|d| d.default_output && d.output).map(|d| d.name.clone());
    Ok(UiDevices {
        input: pick(&choice.input, &inputs, default_in),
        // an explicit choice is honoured even when it will not open, so the error names it;
        // defaults only ever land on an output that opens
        output: match &choice.output {
            Some(w) if outputs.contains(w) => w.clone(),
            // the app exists to feed games and chat, so without a usable choice OUT is the virtual
            // cable; the headset (hearing yourself) is an explicit preview choice
            _ => {
                let cable = usable.iter().find(|o| o.starts_with("CABLE Input")).cloned();
                pick(&cable, &usable, default_out.clone())
            }
        },
        inputs,
        outputs,
        unusable_outputs,
        default_output: default_out.unwrap_or_default(),
    })
}

/// Virtual cables feed another app; nobody hears them directly.
fn is_virtual(name: &str) -> bool {
    let l = name.to_lowercase();
    l.contains("cable") || l.contains("vb-audio") || l.contains("voicemeeter")
}

/// Where the monitor plays. Only needed while OUT is a virtual cable, and never onto another
/// cable (that would put the voice into the game twice): the system default output when it is a
/// real device, else the output of the same headset as the microphone, else the first real one.
/// The UI applies the same rule (SignalChain.tsx `pickMonitor`).
fn pick_monitor(d: &UiDevices) -> Option<String> {
    if !is_virtual(&d.output) {
        return None;
    }
    let real: Vec<&String> =
        d.outputs.iter().filter(|o| !is_virtual(o) && **o != d.output && !d.unusable_outputs.contains(o)).collect();
    if real.iter().any(|o| **o == d.default_output) {
        return Some(d.default_output.clone());
    }
    let headset = d.input.split_once('(').and_then(|(_, r)| r.split_once(')')).map(|(m, _)| m);
    if let Some(h) = headset {
        if let Some(o) = real.iter().find(|o| o.contains(h)) {
            return Some((*o).clone());
        }
    }
    real.first().map(|o| (*o).clone())
}

fn spawn_start(app: AppHandle, shared: Arc<Mutex<Shared>>, voice: String, params: UiParams, choice: DeviceChoice) {
    set_status(&app, &shared, "loading", None);
    thread::spawn(move || {
        let result = (|| -> anyhow::Result<LiveSession> {
            let devices = resolve_devices(&choice)?;
            let proc = build_processor(&voice, &params)?;
            let monitor = pick_monitor(&devices);
            let monitor_on = shared.lock().map(|s| s.monitor).unwrap_or(false);
            let cfg = LiveConfig { input: Some(devices.input), output: Some(devices.output), monitor, monitor_on, ..LiveConfig::default() };
            LiveSession::start(proc, cfg)
        })();
        match result {
            Ok(session) => {
                // a monitor that would not open leaves the session running; tell the UI why
                let warning = session.monitor_warning.as_ref().map(|w| format!("monitor skipped: {w}"));
                let generation = {
                    let mut s = shared.lock().unwrap();
                    s.session = Some(session);
                    s.params = Some(params.clone());
                    s.voice = voice;
                    s.choice = choice;
                    s.generation += 1;
                    s.generation
                };
                set_status(&app, &shared, "live", warning);
                spawn_pump(app, shared, generation);
            }
            Err(e) => set_status(&app, &shared, "error", Some(format!("{e:#}"))),
        }
    });
}

/// 30 Hz telemetry: device-callback levels every tick, block stats whenever a block lands.
fn spawn_pump(app: AppHandle, shared: Arc<Mutex<Shared>>, generation: u64) {
    thread::spawn(move || {
        let t0 = Instant::now();
        let mut gpu_meter = GpuMeter::new();
        let mut gpu = -1.0_f32;
        let mut gpu_sample_at = Instant::now();
        let (mut f0_in, mut f0_out, mut latency, mut rtf, mut under) = (0f32, 0f32, 0f32, 0f32, 0u64);
        let (mut gate_db, mut gate_open, mut vad) = (-90f32, false, 0f32);
        let (mut wave_in, mut wave_out) = (vec![0.0f32; 256], vec![0.0f32; 256]);
        loop {
            thread::sleep(Duration::from_millis(33));
            // PDH needs two collections at least 200 ms apart; never poll in audio callbacks.
            // Keep a valid reading between polls, but report unavailable after a failed poll.
            if gpu_sample_at.elapsed() >= MIN_SAMPLE_INTERVAL {
                gpu = gpu_meter.as_mut().and_then(GpuMeter::sample).unwrap_or(-1.0);
                gpu_sample_at = Instant::now();
            }
            let payload = {
                let Ok(s) = shared.lock() else { break };
                if s.generation != generation {
                    break;
                }
                let Some(session) = s.session.as_ref() else { break };
                while let Ok(t) = session.telemetry().try_recv() {
                    if t.f0_in > 0.0 {
                        f0_in = t.f0_in;
                        f0_out = t.f0_out;
                    } else if t.silent {
                        f0_in = 0.0;
                        f0_out = 0.0;
                    }
                    latency = t.latency_ms;
                    rtf = t.rtf;
                    gate_db = if t.input_level > 1e-6 { 20.0 * t.input_level.log10() } else { -90.0 };
                    gate_open = !t.silent;
                    vad = t.vad;
                    under = t.underruns;
                    wave_in = t.wave_in;
                    wave_out = t.wave_out;
                }
                let (li, lo) = session.levels();
                UiTelemetry {
                    t: t0.elapsed().as_secs_f64() * 1000.0,
                    input_level: li,
                    output_level: lo,
                    f0_in,
                    f0_out,
                    latency_ms: latency,
                    rtf,
                    gpu,
                    underruns: under,
                    gate_db,
                    gate_open,
                    vad,
                    wave_in: wave_in.clone(),
                    wave_out: wave_out.clone(),
                }
            };
            let _ = app.emit("telemetry", payload);
        }
    });
}

#[tauri::command]
fn engine_status(state: State<EngineState>) -> UiState {
    let s = state.0.lock().unwrap();
    UiState { status: if s.status.is_empty() { "ready".into() } else { s.status.clone() }, error: s.error.clone() }
}

#[tauri::command]
fn list_audio_devices(state: State<EngineState>) -> Result<UiDevices, String> {
    let choice = state.0.lock().unwrap().choice.clone();
    resolve_devices(&choice).map_err(|e| format!("{e:#}"))
}

#[tauri::command]
fn select_devices(state: State<EngineState>, choice: DeviceChoice) {
    state.0.lock().unwrap().choice = choice;
}

#[tauri::command]
fn engine_start(app: AppHandle, state: State<EngineState>, voice: String, params: UiParams) -> Result<(), String> {
    let shared = state.0.clone();
    let (existing, choice) = {
        let mut s = shared.lock().unwrap();
        (s.session.take(), s.choice.clone())
    };
    if let Some(old) = existing {
        let _ = old.stop();
    }
    spawn_start(app, shared, voice, params, choice);
    Ok(())
}

#[tauri::command]
fn engine_stop(app: AppHandle, state: State<EngineState>) {
    let shared = state.0.clone();
    let existing = shared.lock().unwrap().session.take();
    set_status(&app, &shared, "loading", None);
    thread::spawn(move || {
        if let Some(s) = existing {
            let _ = s.stop();
        }
        set_status(&app, &shared, "ready", None);
    });
}

#[tauri::command]
fn set_monitor(state: State<EngineState>, on: bool) {
    let mut s = state.0.lock().unwrap();
    s.monitor = on;
    if let Some(session) = s.session.as_ref() {
        session.set_monitor(on);
    }
}

#[tauri::command]
fn engine_set_params(app: AppHandle, state: State<EngineState>, params: UiParams) {
    let shared = state.0.clone();
    let restart = {
        let mut s = shared.lock().unwrap();
        let live = s.session.is_some();
        let block_changed = s.params.as_ref().is_some_and(|p| p.block_ms != params.block_ms);
        if live && block_changed {
            Some((s.voice.clone(), s.choice.clone()))
        } else {
            if let Some(session) = s.session.as_ref() {
                session.set_params(Params { pitch: params.pitch, index_rate: params.index_rate, protect: params.protect, silence_db: params.gate_db, denoise_audio: params.denoise, vad: params.vad, envelope: params.envelope, ..Params::default() });
            }
            s.params = Some(params.clone());
            None
        }
    };
    if let Some((voice, choice)) = restart {
        // block size is part of the processor layout: rebuild the session
        let existing = shared.lock().unwrap().session.take();
        set_status(&app, &shared, "loading", None);
        let (app2, shared2) = (app.clone(), shared.clone());
        thread::spawn(move || {
            if let Some(s) = existing {
                let _ = s.stop();
            }
            spawn_start(app2, shared2, voice, params, choice);
        });
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    // Ctrl+Alt+V from any window (the game included): the UI decides start vs stop
                    if matches!(event.state, ShortcutState::Pressed) {
                        let _ = app.emit("hotkey-toggle", ());
                    }
                })
                .build(),
        )
        .manage(EngineState(Arc::new(Mutex::new(Shared::default()))))
        .invoke_handler(tauri::generate_handler![engine_status, list_audio_devices, select_devices, engine_start, engine_stop, engine_set_params, set_monitor])
        .setup(|app| {
            let hotkey = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::KeyV);
            if let Err(e) = app.global_shortcut().register(hotkey) {
                eprintln!("[mimic] global hotkey Ctrl+Alt+V unavailable: {e}");
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A headset with separate chat and game endpoints next to VB-Cable, with Windows' default
    /// output moved onto the 16-channel cable.
    fn devices(output: &str, default_output: &str) -> UiDevices {
        UiDevices {
            inputs: vec!["CABLE Output (VB-Audio Virtual Cable)".into(), "麦克风 (Chat-USB Headset)".into()],
            outputs: vec![
                "CABLE In 16ch (VB-Audio Virtual Cable)".into(),
                "扬声器 (Chat-USB Headset)".into(),
                "XG27ACS (NVIDIA High Definition Audio)".into(),
                "CABLE Input (VB-Audio Virtual Cable)".into(),
                "扬声器 (Game-USB Headset)".into(),
            ],
            input: "麦克风 (Chat-USB Headset)".into(),
            output: output.into(),
            default_output: default_output.into(),
            unusable_outputs: vec![],
        }
    }

    #[test]
    fn skips_a_headset_that_will_not_open() {
        let mut d = devices("CABLE Input (VB-Audio Virtual Cable)", "CABLE In 16ch (VB-Audio Virtual Cable)");
        d.unusable_outputs = vec!["扬声器 (Chat-USB Headset)".into()];
        assert_eq!(pick_monitor(&d).as_deref(), Some("XG27ACS (NVIDIA High Definition Audio)"));
    }

    #[test]
    fn no_monitor_when_out_is_already_a_real_device() {
        assert_eq!(pick_monitor(&devices("扬声器 (Chat-USB Headset)", "CABLE In 16ch (VB-Audio Virtual Cable)")), None);
    }

    #[test]
    fn never_monitors_onto_another_cable() {
        // default output is the 16ch cable: fall back to the headset the microphone belongs to
        let d = devices("CABLE Input (VB-Audio Virtual Cable)", "CABLE In 16ch (VB-Audio Virtual Cable)");
        assert_eq!(pick_monitor(&d).as_deref(), Some("扬声器 (Chat-USB Headset)"));
    }

    #[test]
    fn a_real_default_output_wins() {
        let d = devices("CABLE Input (VB-Audio Virtual Cable)", "扬声器 (Game-USB Headset)");
        assert_eq!(pick_monitor(&d).as_deref(), Some("扬声器 (Game-USB Headset)"));
    }

    #[test]
    fn nothing_real_left() {
        let mut d = devices("CABLE Input (VB-Audio Virtual Cable)", "CABLE In 16ch (VB-Audio Virtual Cable)");
        d.outputs.retain(|o| o.contains("CABLE"));
        assert_eq!(pick_monitor(&d), None);
    }
}
