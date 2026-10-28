import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { AudioDevices, Engine, EngineState, Telemetry, VoiceParams } from "./types";

interface WireTelemetry {
  t: number;
  inputLevel: number;
  outputLevel: number;
  f0In: number;
  f0Out: number;
  latencyMs: number;
  rtf: number;
  gpu: number;
  underruns: number;
  gateDb: number;
  gateOpen: boolean;
  vad: number;
  waveIn: number[];
  waveOut: number[];
}

interface WireState {
  status: EngineState;
  error: string | null;
}

/** Engine backed by the Rust bridge in src-tauri (events: engine-state, telemetry). */
export function createTauriEngine(): Engine {
  const listeners = new Set<(t: Telemetry) => void>();
  const stateListeners = new Set<(s: EngineState) => void>();
  let state: EngineState = "loading";
  let lastError: string | null = null;
  let params: VoiceParams | null = null;

  const setState = (s: EngineState) => {
    state = s;
    stateListeners.forEach((l) => l(s));
  };

  void listen<WireState>("engine-state", (e) => {
    lastError = e.payload.error;
    if (lastError) console.error("engine:", lastError);
    setState(e.payload.status);
  });
  void listen<WireTelemetry>("telemetry", (e) => {
    const p = e.payload;
    const t: Telemetry = {
      t: p.t,
      inputLevel: p.inputLevel,
      outputLevel: p.outputLevel,
      f0In: p.f0In,
      f0Out: p.f0Out,
      latencyMs: p.latencyMs,
      rtf: p.rtf,
      gpu: p.gpu,
      underruns: p.underruns,
      gateDb: p.gateDb,
      gateOpen: p.gateOpen,
      vad: p.vad,
      waveIn: Float32Array.from(p.waveIn),
      waveOut: Float32Array.from(p.waveOut),
    };
    listeners.forEach((l) => l(t));
  });
  void invoke<WireState>("engine_status").then((s) => {
    lastError = s.error;
    setState(s.status);
  });

  return {
    subscribe(l) {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    state: () => state,
    onState(l) {
      stateListeners.add(l);
      return () => stateListeners.delete(l);
    },
    lastError: () => lastError,
    async start(voiceId) {
      if (!params) throw new Error("params not set");
      await invoke("engine_start", { voice: voiceId, params });
    },
    async stop() {
      await invoke("engine_stop");
    },
    setParams(p) {
      params = p;
      void invoke("engine_set_params", { params: p });
    },
    async devices(): Promise<AudioDevices> {
      return invoke<AudioDevices>("list_audio_devices");
    },
    async selectDevices(input, output) {
      await invoke("select_devices", { choice: { input, output } });
    },
    async setMonitor(on) {
      await invoke("set_monitor", { on });
    },
  };
}
