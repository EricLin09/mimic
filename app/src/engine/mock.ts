import { VOICES } from "../data/voices";
import type { AudioDevices, Engine, EngineState, Telemetry, VoiceParams } from "./types";

/**
 * Stand-in engine for UI work: synthesises speech-like envelopes, intonation and
 * telemetry at ~60 Hz so every widget can be built and tuned without audio hardware.
 */
export function createMockEngine(): Engine {
  const listeners = new Set<(t: Telemetry) => void>();
  const stateListeners = new Set<(s: EngineState) => void>();
  let state: EngineState = "loading";
  let params: VoiceParams = { ...VOICES[0].defaults };

  const setState = (s: EngineState) => {
    state = s;
    stateListeners.forEach((l) => l(s));
  };
  setTimeout(() => setState("ready"), 900);

  // speech model: phrases made of syllables, natural voice around 126 Hz
  let phraseEnd = 0;
  let nextPhrase = 600;
  let syllablePhase = 0;
  let level = 0;
  let f0 = 0;
  let wavePhase = 0;
  let latency = 160;
  const start = performance.now();
  const N = 256;

  const tick = () => {
    const now = performance.now() - start;
    if (now > nextPhrase && now > phraseEnd) {
      phraseEnd = now + 900 + Math.random() * 2200;
      nextPhrase = phraseEnd + 300 + Math.random() * 1400;
    }
    const speaking = now < phraseEnd;
    syllablePhase += 0.21 + Math.random() * 0.04;
    const syllable = Math.max(0, Math.sin(syllablePhase)) ** 0.6;
    const target = speaking ? 0.35 + 0.5 * syllable : 0;
    level += (target - level) * 0.25;

    const intonation = 126 + 18 * Math.sin(now / 700) + 9 * Math.sin(now / 173);
    const voiced = speaking && syllable > 0.25;
    f0 = voiced ? intonation : 0;

    const live = state === "live";
    const ratio = 2 ** (params.pitch / 12);
    latency += ((live ? params.blockMs + 42 + Math.random() * 14 : 0) - latency) * 0.08;

    const waveIn = new Float32Array(N);
    const waveOut = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const ph = wavePhase + (i / N) * Math.PI * 2 * 6;
      const v = voiced ? Math.sin(ph) * 0.6 + Math.sin(ph * 2) * 0.25 + Math.sin(ph * 3) * 0.12 : 0;
      const noise = (Math.random() - 0.5) * (speaking ? 0.25 : 0.02);
      waveIn[i] = (v + noise) * level;
      const php = ph * ratio;
      const vo = voiced ? Math.sin(php) * 0.7 + Math.sin(php * 2) * 0.35 + Math.sin(php * 4) * 0.1 : 0;
      waveOut[i] = live ? (vo + noise * 0.4) * level * 0.95 : 0;
    }
    wavePhase += 0.9;

    const t: Telemetry = {
      t: now,
      inputLevel: level,
      outputLevel: live ? level * 0.92 : 0,
      f0In: f0,
      f0Out: live && voiced ? f0 * ratio : 0,
      latencyMs: latency,
      rtf: live ? 0.28 + Math.random() * 0.06 + params.blockMs * 0.0002 : 0,
      gpu: live ? 0.18 + level * 0.12 + Math.random() * 0.03 : 0.02,
      waveIn,
      waveOut,
    };
    listeners.forEach((l) => l(t));
    raf = requestAnimationFrame(tick);
  };
  let raf = requestAnimationFrame(tick);
  void raf;

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
    async start() {
      setState("loading");
      await new Promise((r) => setTimeout(r, 650));
      setState("live");
    },
    async stop() {
      setState("ready");
    },
    setParams(p) {
      params = p;
    },
    async devices(): Promise<AudioDevices> {
      return {
        inputs: ["麦克风 (USB Headset)", "麦克风 (Realtek HD Audio)"],
        outputs: ["CABLE Input (VB-Audio Virtual Cable)", "扬声器 (USB Headset)"],
        input: "麦克风 (USB Headset)",
        // dev builds stay off the virtual cable so Applio keeps working
        output: "扬声器 (USB Headset)",
      };
    },
  };
}
