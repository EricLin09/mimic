/** Contract between the UI and the voice engine (mock now, Rust over Tauri IPC later). */

export type EngineState = "loading" | "ready" | "live" | "error";

export interface VoiceParams {
  /** semitones from the voice's pitch centre (0 = centred on the voice when it has f0 stats) */
  pitch: number;
  /** 0..1, how strongly features are pulled toward the voice's index */
  indexRate: number;
  /** 0..0.5, keeps unvoiced consonants from the input (0.5 = off) */
  protect: number;
  /** processing block in milliseconds; the main latency knob */
  blockMs: number;
  /** input gate in dBFS: quieter blocks are not converted at all */
  gateDb: number;
  /** RNNoise also cleans the audio that gets converted (the gate always judges a denoised copy) */
  denoise: boolean;
  /** voice-probability threshold 0..1 the gate needs on top of the level; 0 = level only */
  vad: number;
  /** a loud block with a steady pitch opens the gate even when the voice detector refuses it (held vowels, glides) */
  gatePitch: boolean;
  /** 0..1: follow the speaker's loudness contour (not their absolute level); 0 = the model's own dynamics */
  envelope: number;
}

export interface Telemetry {
  /** engine clock in ms */
  t: number;
  /** 0..1 perceptual levels */
  inputLevel: number;
  outputLevel: number;
  /** Hz, 0 when unvoiced */
  f0In: number;
  f0Out: number;
  /** end-to-end estimate in ms */
  latencyMs: number;
  /** inference time / block time */
  rtf: number;
  /** This process contribution to its busiest GPU engine, 0..1; negative when unavailable. */
  gpu: number;
  /** output buffer underruns since start (real engine only) */
  underruns?: number;
  /** block RMS above 150 Hz after denoising, dBFS (real engine only) */
  gateDb?: number;
  gateOpen?: boolean;
  /** highest Silero VAD chunk probability 0..1 in the block */
  vad?: number;
  /** short oscilloscope snapshots, -1..1 */
  waveIn: Float32Array;
  waveOut: Float32Array;
}

export interface AudioDevices {
  inputs: string[];
  outputs: string[];
  input: string;
  output: string;
  /** system default output (the monitor's first choice when it is a real device) */
  defaultOutput?: string;
  /** listed outputs that would not open right now (busy, format refused) */
  unusableOutputs?: string[];
}

export interface VoiceProfile {
  id: string;
  name: string;
  codename: string;
  region: string;
  /** hex colour used for --accent */
  accent: string;
  /** median voiced f0 of the voice, Hz */
  targetF0: number;
  /** speaker-ID margin from the acceptance test (rank 1 = recognised) */
  matchRank?: number;
  matchMargin?: number;
  /** short free-text tag shown after the region */
  tag?: string;
  defaults: VoiceParams;
  /** dark palette colour for the backdrop */
  tint: string;
  /** transparent full-body render, loaded from the voice folder (never bundled) */
  portrait?: string;
  /** emblem art drawn behind the portrait */
  background?: string;
}

export interface Engine {
  subscribe(listener: (t: Telemetry) => void): () => void;
  state(): EngineState;
  onState(listener: (s: EngineState) => void): () => void;
  /** message of the last failure, when state is "error" */
  lastError?(): string | null;
  start(voiceId: string): Promise<void>;
  stop(): Promise<void>;
  setParams(p: VoiceParams): void;
  devices(): Promise<AudioDevices>;
  /** remember the devices for the next start */
  selectDevices?(input: string, output: string): Promise<void>;
  /** hear the converted voice on the default output while OUT feeds a virtual mic */
  setMonitor?(on: boolean): Promise<void>;
}
