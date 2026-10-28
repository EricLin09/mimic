import type { VoiceParams, VoiceProfile } from "../engine/types";

/** What the app starts from; a voice's profile.json may override any of it. */
export const DEFAULT_PARAMS: VoiceParams = {
  pitch: 0, indexRate: 0.75, protect: 0.33, blockMs: 240, gateDb: -48, denoise: false, vad: 0.5, envelope: 0.5,
};

/** models/voices/<id>/profile.json: how the UI presents a voice. Every field is optional. */
interface ProfileFile {
  name?: string;
  codename?: string;
  region?: string;
  tag?: string;
  accent?: string;
  tint?: string;
  targetF0?: number;
  matchRank?: number;
  matchMargin?: number;
  /** file names inside the voice folder */
  portrait?: string;
  background?: string;
  defaults?: Partial<VoiceParams>;
}

// Voices live outside the source tree (models/voices, never committed). They are collected at build
// time: every folder with a synth.json is a voice, its profile.json (if any) says how to show it.
// A checkout without voices gets the neutral placeholder below.
const synths = import.meta.glob<number>("../../../models/voices/*/synth.json", { eager: true, import: "sample_rate" });
const profiles = import.meta.glob<ProfileFile>("../../../models/voices/*/profile.json", { eager: true, import: "default" });
const folder = (path: string) => path.split("/").slice(-2, -1)[0];

function fromFile(id: string, p: ProfileFile): VoiceProfile {
  return {
    id,
    name: p.name ?? id,
    codename: p.codename ?? id.toUpperCase(),
    region: p.region ?? "",
    tag: p.tag,
    accent: p.accent ?? "#4fd8ff",
    tint: p.tint ?? "#20435b",
    targetF0: p.targetF0 ?? 120,
    matchRank: p.matchRank,
    matchMargin: p.matchMargin,
    defaults: { ...DEFAULT_PARAMS, ...p.defaults },
    portrait: p.portrait ? `/voices/${id}/${p.portrait}` : undefined,
    background: p.background ? `/voices/${id}/${p.background}` : undefined,
  };
}

const byId = new Map(Object.entries(profiles).map(([path, p]) => [folder(path), p]));
const installed = Object.keys(synths).map(folder).sort();

export const VOICES: VoiceProfile[] = installed.length
  ? installed.map((id) => fromFile(id, byId.get(id) ?? {}))
  : [fromFile("default", { name: "示例声线", codename: "VOICE", region: "导出的声线放进 models/voices/" })];

export function hexToRgb(hex: string): string {
  const n = parseInt(hex.slice(1), 16);
  return `${(n >> 16) & 255} ${(n >> 8) & 255} ${n & 255}`;
}
