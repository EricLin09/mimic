import { motion } from "motion/react";
import type { VoiceProfile } from "../engine/types";
import { hexToRgb } from "../data/voices";
import { Icon } from "./Icons";

function Portrait({ voice, size }: { voice: VoiceProfile; size: number }) {
  const rgb = hexToRgb(voice.accent);
  if (voice.portrait) {
    // head-and-shoulders crop of the full-body render
    return (
      <div
        style={{
          width: size,
          height: size,
          backgroundColor: "#0d0f14",
          backgroundImage: `radial-gradient(circle at 50% 30%, rgb(${rgb} / 0.35), transparent 70%), url(${voice.portrait})`,
          backgroundSize: "100% 100%, 480%",
          backgroundPosition: "center, 48.6% 12%",
          backgroundRepeat: "no-repeat",
        }}
      />
    );
  }
  // generated emblem until the user supplies a portrait
  return (
    <div
      className="relative flex items-center justify-center overflow-hidden font-display font-bold"
      style={{
        width: size,
        height: size,
        background: `radial-gradient(circle at 30% 20%, rgb(${rgb} / 0.55), rgb(${rgb} / 0.08) 60%), #0d0f14`,
        color: `rgb(${rgb})`,
        fontSize: size * 0.34,
      }}
    >
      <div
        className="absolute inset-0 opacity-40"
        style={{ backgroundImage: `repeating-linear-gradient(135deg, rgb(${rgb} / 0.35) 0 1px, transparent 1px 7px)` }}
      />
      <span className="relative">{voice.codename.slice(0, 2)}</span>
    </div>
  );
}

export function Roster({ voices, selected, onSelect }: {
  voices: VoiceProfile[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="space-y-3">
      {voices.map((v) => {
        const on = v.id === selected;
        return (
          <motion.button
            key={v.id}
            onClick={() => onSelect(v.id)}
            whileHover={{ x: 3 }}
            className="panel relative flex w-full cursor-pointer items-center gap-3 p-3 text-left"
            style={on ? { boxShadow: `inset 3px 0 0 ${v.accent}` } : undefined}
          >
            <div style={{ clipPath: "polygon(0 0, 80% 0, 100% 20%, 100% 100%, 0 100%)" }}>
              <Portrait voice={v} size={54} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="font-cn text-[19px] leading-none text-text">{v.name}</span>
                {on && <span className="h-1.5 w-1.5 rounded-full" style={{ background: v.accent, boxShadow: `0 0 8px ${v.accent}` }} />}
              </div>
              <div className="label mt-0.5">{[v.codename, v.region].filter(Boolean).join(" · ")}</div>
              {v.matchRank !== undefined && (
                <div className="mt-2 flex items-center gap-2 text-[11px] text-dim">
                  <span className="font-num text-[15px] leading-none" style={{ color: v.accent }}>#{v.matchRank}</span>
                  {v.matchMargin !== undefined && <span>认人测试 · 领先 +{v.matchMargin.toFixed(3)}</span>}
                </div>
              )}
            </div>
          </motion.button>
        );
      })}
      <button className="flex h-[78px] w-full cursor-pointer items-center justify-center gap-2 border border-dashed border-line-strong font-cn text-[15px] text-dim transition hover:border-white/30 hover:text-muted">
        <Icon name="plus" size={16} /> 训练新角色
      </button>
    </div>
  );
}
