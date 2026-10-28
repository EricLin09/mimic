import type { ReactNode, RefObject } from "react";
import { useTelemetry } from "../engine/useEngine";
import type { Telemetry } from "../engine/types";
import { LevelMeter, PitchTrace, Sparkline } from "./Scopes";
import { AnimatedNumber } from "./Hud";
import { Icon, type IconName } from "./Icons";

function Label({ icon, text }: { icon: IconName; text: string }) {
  return (
    <div className="label flex items-center gap-1.5">
      <Icon name={icon} size={12} />
      {text}
    </div>
  );
}

function Stat({ icon, label, value, decimals = 0, unit, children }: {
  icon: IconName;
  label: string;
  value: number | null;
  decimals?: number;
  unit?: string;
  children?: ReactNode;
}) {
  return (
    <div className="min-w-0">
      <Label icon={icon} text={label} />
      <div className="mt-0.5 flex items-baseline gap-1">
        <span className="font-num text-[34px] leading-none text-text">
          <AnimatedNumber value={value} decimals={decimals} />
        </span>
        {unit && <span className="font-num text-sm tracking-wider text-dim">{unit}</span>}
      </div>
      {children}
    </div>
  );
}

export function TelemetryBar({ telemetry, targetF0, live }: {
  telemetry: RefObject<Telemetry | null>;
  targetF0: number;
  live: boolean;
}) {
  const t = useTelemetry(8);
  const gpu = live && t && Number.isFinite(t.gpu) && t.gpu >= 0
    ? Math.round(Math.min(t.gpu, 1) * 100)
    : null;
  return (
    <div className="panel grid grid-cols-[150px_96px_96px_1fr_1fr_1.7fr] items-center gap-6 px-5 py-3">
      <Stat icon="timer" label="延迟" value={live && t ? Math.round(t.latencyMs) : null} unit="MS">
        <div className="mt-1 h-4">
          <Sparkline telemetry={telemetry} pick={(x) => x.latencyMs} max={320} />
        </div>
      </Stat>
      <Stat icon="gauge" label="实时率" value={live && t ? t.rtf : null} decimals={2} />
      <Stat icon="chip" label="GPU" value={gpu} unit="%">
        <div className="mt-1 flex items-baseline gap-1.5 text-[10px] text-dim">
          <span>欠运行</span>
          <span className="font-num tabular-nums text-muted">
            {live && t ? (t.underruns ?? 0) : "—"}
          </span>
        </div>
      </Stat>
      <div>
        <Label icon="mic" text="输入 · 你" />
        <div className="mt-2.5 h-2">
          <LevelMeter telemetry={telemetry} pick={(x) => x.inputLevel} rgb="79 216 255" />
        </div>
      </div>
      <div>
        <Label icon="speaker" text="输出 · 角色" />
        <div className="mt-2.5 h-2">
          <LevelMeter telemetry={telemetry} pick={(x) => x.outputLevel} />
        </div>
      </div>
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex w-10 shrink-0 flex-col items-center gap-1 text-muted">
          <Icon name="pitch" size={16} />
          <span className="label">音高</span>
        </div>
        <div className="h-[52px] min-w-0 flex-1">
          <PitchTrace telemetry={telemetry} targetF0={targetF0} />
        </div>
      </div>
    </div>
  );
}
