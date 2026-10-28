import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { engine, engineIsReal, useEngineState, useTelemetry, useTelemetryRef } from "./engine/useEngine";
import type { AudioDevices, VoiceParams } from "./engine/types";
import { hexToRgb, VOICES } from "./data/voices";
import { TitleBar } from "./components/TitleBar";
import { Roster } from "./components/Roster";
import { VoiceCore } from "./components/VoiceCore";
import { AgentStage } from "./components/AgentStage";
import { WaveRibbon } from "./components/Scopes";
import { ActivateButton, EngineNotice, TuningPanel } from "./components/Controls";
import { SignalChain } from "./components/SignalChain";
import { TelemetryBar } from "./components/TelemetryBar";
import { AnimatedNumber, BootSplash, DecodeText } from "./components/Hud";

function PitchReadout({ live }: { live: boolean }) {
  const t = useTelemetry(10);
  const hz = (f: number | undefined) => (f && f > 0 ? Math.round(f) : null);
  return (
    <div className="flex items-end gap-3 text-right">
      <div>
        <div className="label">你的音高</div>
        <div className="font-num text-[44px] leading-none text-you" style={{ textShadow: "0 0 18px rgb(79 216 255 / 0.4)" }}>
          <AnimatedNumber value={hz(t?.f0In)} />
          <span className="ml-1 text-base text-dim">HZ</span>
        </div>
      </div>
      <svg width="30" height="16" viewBox="0 0 30 16" className="mb-3 text-dim">
        <path d="M0 8 H26 M19 1 L27 8 L19 15" fill="none" stroke="currentColor" strokeWidth="1.6" />
      </svg>
      <div>
        <div className="label">输出音高</div>
        <div className="font-num accent-text text-[44px] leading-none">
          <AnimatedNumber value={live ? hz(t?.f0Out) : null} />
          <span className="ml-1 text-base text-dim">HZ</span>
        </div>
      </div>
    </div>
  );
}

const STORE_KEY = "mimic.settings.v1";
interface Stored {
  /** settings layout version, see RESET */
  schema?: number;
  params?: Partial<VoiceParams>;
  input?: string;
  output?: string;
  monitor?: boolean;
}
function loadStored(): Stored {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) ?? "{}") as Stored;
  } catch {
    return {};
  }
}
/** Knobs whose meaning changed: a value stored before that schema would be misread, so it falls
 *  back to the voice default. */
const RESET: [number, keyof VoiceParams][] = [
  // v2: pitch 0 now means "centred on the voice", no longer "no shift"
  [2, "pitch"],
  // v3: the denoise switch now covers the converted audio only; the gate always uses a denoised copy
  [3, "denoise"],
];
const SCHEMA = Math.max(1, ...RESET.map(([v]) => v));
function storedParams(): Partial<VoiceParams> {
  const s = loadStored();
  const p = { ...s.params };
  for (const [version, key] of RESET) if ((s.schema ?? 1) < version) delete p[key];
  return p;
}
function saveStored(patch: Stored) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({ ...loadStored(), ...patch }));
  } catch {
    /* storage unavailable: settings just don't persist */
  }
}

function SectionTitle({ en, zh }: { en: string; zh: string }) {
  return (
    <div className="mb-3 flex items-center gap-2">
      <span className="h-3.5 w-[3px]" style={{ background: "var(--accent)", boxShadow: "0 0 8px var(--accent)" }} />
      <span className="font-cn text-[15px] text-text">{zh}</span>
      <span className="font-num text-[14px] tracking-[0.25em] text-dim">{en}</span>
      <span className="h-px flex-1 bg-line" />
    </div>
  );
}

export default function App() {
  const [voiceId, setVoiceId] = useState(VOICES[0].id);
  const voice = VOICES.find((v) => v.id === voiceId)!;
  // stored settings may predate newer knobs: fill the gaps from the voice defaults
  const [params, setParams] = useState<VoiceParams>(() => ({ ...voice.defaults, ...storedParams() }));
  const [monitor, setMonitor] = useState<boolean>(() => loadStored().monitor ?? false);
  const [devices, setDevices] = useState<AudioDevices | null>(null);
  const state = useEngineState();
  const telemetry = useTelemetryRef();
  const live = state === "live";

  const chooseDevices = useCallback((input: string, output: string) => {
    setDevices((d) => (d ? { ...d, input, output } : d));
    saveStored({ input, output });
    void engine.selectDevices?.(input, output);
  }, []);
  useEffect(() => {
    engine
      .devices()
      .then((d) => {
        // restore the last choice when those devices are still present
        const s = loadStored();
        const input = s.input && d.inputs.includes(s.input) ? s.input : d.input;
        const opens = (o: string) => d.outputs.includes(o) && !(d.unusableOutputs ?? []).includes(o);
        const output = s.output && opens(s.output) ? s.output : d.output;
        setDevices({ ...d, input, output });
        if (input !== d.input || output !== d.output) void engine.selectDevices?.(input, output);
      })
      .catch((e) => console.error("devices:", e));
  }, []);
  useEffect(() => {
    engine.setParams(params);
    saveStored({ params, schema: SCHEMA });
  }, [params]);
  useEffect(() => {
    saveStored({ monitor });
    void engine.setMonitor?.(monitor);
  }, [monitor]);
  useEffect(() => {
    const root = document.documentElement.style;
    root.setProperty("--accent", voice.accent);
    root.setProperty("--accent-rgb", hexToRgb(voice.accent));
  }, [voice]);

  const toggle = useCallback(() => {
    if (state === "live") void engine.stop();
    // after a failed start the next click is a retry
    else if (state === "ready" || state === "error") void engine.start(voiceId);
  }, [state, voiceId]);

  // Ctrl+Alt+V: a global shortcut registered by the Rust side in the app, a window key in the browser
  const toggleRef = useRef(toggle);
  toggleRef.current = toggle;
  useEffect(() => {
    if (engineIsReal) {
      let dispose: (() => void) | undefined;
      void listen("hotkey-toggle", () => toggleRef.current()).then((un) => {
        dispose = un;
      });
      return () => dispose?.();
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.altKey && e.key.toLowerCase() === "v") {
        e.preventDefault();
        toggleRef.current();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="app-backdrop relative flex h-full flex-col">
      <div className="grid-overlay pointer-events-none absolute inset-0" />
      <div className="scanlines absolute inset-0" />
      <TitleBar state={state} />

      <main className="relative z-10 grid min-h-0 flex-1 grid-cols-[272px_1fr_292px] gap-6 px-5 pb-4">
        <aside className="flex min-h-0 flex-col">
          <SectionTitle zh="角色" en="AGENTS" />
          <Roster voices={VOICES} selected={voiceId} onSelect={setVoiceId} />
          <div className="mt-auto font-mono text-[10.5px] leading-relaxed text-dim">
            RVC v2 · contentvec · 40 kHz<br />
            ONNX Runtime · 本机 GPU
          </div>
        </aside>

        <section className="relative flex min-h-0 flex-col">
          <div className="relative min-h-0 flex-1">
            {voice.portrait ? (
              <AgentStage
                telemetry={telemetry}
                active={live}
                accent={voice.accent}
                tint={voice.tint}
                portrait={voice.portrait}
                background={voice.background}
              />
            ) : (
              <VoiceCore telemetry={telemetry} active={live} accent={voice.accent} />
            )}
            <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between">
              <div className="relative">
                <div
                  className="outline-text pointer-events-none absolute -top-3 -left-1 font-num text-[92px] leading-none tracking-[0.06em] select-none"
                  aria-hidden
                >
                  {voice.codename}
                </div>
                <div className="relative label">当前角色 · VOICE PROFILE</div>
                <h1 className="relative mt-1 font-cn text-[58px] leading-none accent-text" style={{ transform: "skewX(-6deg)" }}>
                  <DecodeText key={voice.id} text={voice.name} />
                </h1>
                <div className="relative mt-2 flex items-center gap-2 font-num text-[15px] tracking-[0.22em] text-muted">
                  <span style={{ color: "var(--accent)" }}>{voice.codename}</span>
                  {voice.region && (
                    <>
                      <span className="text-dim">/</span>
                      <span className="font-cn tracking-normal">{voice.region}</span>
                    </>
                  )}
                  {voice.tag && (
                    <>
                      <span className="text-dim">/</span>
                      <span>{voice.tag}</span>
                    </>
                  )}
                </div>
              </div>
              <PitchReadout live={live} />
            </div>
          </div>
          <div className="h-14">
            <WaveRibbon telemetry={telemetry} active={live} />
          </div>
          <div className="relative mt-3 flex justify-center">
            <EngineNotice state={state} message={engine.lastError?.() ?? null} />
            <ActivateButton state={state} onToggle={toggle} />
          </div>
        </section>

        <aside className="flex min-h-0 flex-col gap-5">
          <div className="panel p-4">
            <SectionTitle zh="信号链路" en="SIGNAL" />
            <SignalChain devices={devices} live={live || state === "loading"} onDevices={chooseDevices} monitor={monitor} onMonitor={setMonitor} />
          </div>
          <div className="panel min-h-0 flex-1 overflow-y-auto p-4">
            <SectionTitle zh="调校" en="TUNING" />
            <TuningPanel params={params} onChange={setParams} />
          </div>
        </aside>
      </main>

      <footer className="relative z-10 px-5 pb-5">
        <TelemetryBar telemetry={telemetry} targetF0={voice.targetF0} live={live} />
      </footer>
      <BootSplash />
    </div>
  );
}
