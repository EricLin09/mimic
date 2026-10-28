import { motion } from "motion/react";
import type { AudioDevices } from "../engine/types";
import { Icon, type IconName } from "./Icons";

interface Node {
  icon: IconName;
  tag: string;
  title: string;
  detail: string;
  accent?: boolean;
  /** device picker: current value, options, handler; options that will not open are greyed */
  pick?: { value: string; options: string[]; onChange: (v: string) => void; unusable?: string[]; label?: (o: string) => string };
}

function Picker({ value, options, onChange, disabled, unusable = [], label = (o) => o }: {
  value: string; options: string[]; onChange: (v: string) => void; disabled: boolean; unusable?: string[]; label?: (o: string) => string;
}) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      title={value}
      className="mt-0.5 w-full cursor-pointer truncate border border-line-strong bg-ink-2 px-1.5 py-0.5 text-[11px] text-muted outline-none hover:border-white/30 focus:border-[var(--accent)] disabled:cursor-not-allowed disabled:opacity-60"
    >
      {options.map((o) => (
        <option key={o} value={o} disabled={unusable.includes(o)}>
          {label(o)}{unusable.includes(o) ? " · 现在打不开" : ""}
        </option>
      ))}
    </select>
  );
}

const isVirtual = (n: string) => /cable|vb-audio|voicemeeter/i.test(n);

/** Where the monitor plays; the same rule as `pick_monitor` on the Rust side. Only needed while
 *  OUT is a virtual cable, never onto another cable: the default output when it is real, else the
 *  output of the same headset as the microphone, else the first real output. */
function pickMonitor(d: AudioDevices): string | null {
  if (!isVirtual(d.output)) return null;
  const bad = d.unusableOutputs ?? [];
  const real = d.outputs.filter((o) => !isVirtual(o) && o !== d.output && !bad.includes(o));
  if (d.defaultOutput && real.includes(d.defaultOutput)) return d.defaultOutput;
  const headset = /\(([^)]+)\)/.exec(d.input)?.[1];
  return (headset ? real.find((o) => o.includes(headset)) : undefined) ?? real[0] ?? null;
}

/** Mic -> Mimic -> output -> apps, with light flowing along the links while live. */
export function SignalChain({ devices, live, onDevices, monitor = false, onMonitor }: {
  devices: AudioDevices | null;
  live: boolean;
  onDevices?: (input: string, output: string) => void;
  /** play the converted voice on the default output too */
  monitor?: boolean;
  onMonitor?: (on: boolean) => void;
}) {
  const monitorDevice = devices ? pickMonitor(devices) : null;
  const outIsReal = devices ? !isVirtual(devices.output) : true;
  const toCable = devices?.output.startsWith("CABLE") ?? false;
  const nodes: Node[] = [
    {
      icon: "mic", tag: "IN", title: "麦克风", detail: devices?.input ?? "—",
      pick: devices && onDevices ? { value: devices.input, options: devices.inputs, onChange: (v) => onDevices(v, devices.output) } : undefined,
    },
    { icon: "core", tag: "CORE", title: "MIMIC 引擎", detail: "ONNX Runtime · DirectML", accent: true },
    {
      icon: "speaker", tag: "OUT", title: toCable ? "虚拟麦克风" : "耳机试听", detail: devices?.output ?? "—",
      pick: devices && onDevices
        ? {
            value: devices.output, options: devices.outputs, onChange: (v) => onDevices(devices.input, v),
            unusable: devices.unusableOutputs,
            label: (o) => (o.startsWith("CABLE Input") ? `${o} · 游戏/微信用` : o),
          }
        : undefined,
    },
    {
      icon: "target", tag: "APP", title: toCable ? "游戏 · 语音软件" : "只有你听得到",
      detail: toCable ? "游戏和微信的麦克风选 CABLE Output" : "给游戏/微信用：OUT 选 CABLE Input",
    },
  ];
  return (
    <div className="relative">
      {nodes.map((n, i) => (
        <div key={n.tag} className="relative">
          <div className="flex items-start gap-3 py-1.5">
            <div
              className="relative mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center"
              style={{
                clipPath: "polygon(0 0, 75% 0, 100% 25%, 100% 100%, 25% 100%, 0 75%)",
                background: n.accent ? "rgb(var(--accent-rgb) / 0.9)" : "rgb(255 255 255 / 0.06)",
                color: n.accent ? "#0b0b0b" : live ? "var(--accent)" : "var(--color-muted)",
                boxShadow: n.accent && live ? "0 0 18px rgb(var(--accent-rgb) / 0.6)" : undefined,
              }}
            >
              <Icon name={n.icon} size={18} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-2">
                <span className="font-cn text-[15px] text-text">{n.title}</span>
                <span className="font-num text-[13px] tracking-[0.2em] text-dim">{n.tag}</span>
              </div>
              {n.pick ? (
                <Picker value={n.pick.value} options={n.pick.options} onChange={n.pick.onChange} disabled={live}
                  unusable={n.pick.unusable} label={n.pick.label} />
              ) : (
                <div className="truncate text-[11.5px] text-muted">{n.detail}</div>
              )}
            </div>
          </div>
          {i < nodes.length - 1 && (
            <div className="relative ml-[17.5px] h-3 w-px bg-line-strong">
              {live && (
                <motion.span
                  className="absolute -left-[2px] h-1.5 w-[5px]"
                  style={{ background: "var(--accent)", boxShadow: "0 0 8px var(--accent)" }}
                  initial={{ top: -4, opacity: 0 }}
                  animate={{ top: 12, opacity: [0, 1, 0] }}
                  transition={{ duration: 0.7, repeat: Infinity, delay: i * 0.23, ease: "linear" }}
                />
              )}
            </div>
          )}
        </div>
      ))}
      {onMonitor && (
        <button
          type="button"
          onClick={() => onMonitor(!monitor)}
          aria-pressed={monitor}
          disabled={!monitorDevice}
          className="mt-2 flex w-full items-center gap-2 border border-line-strong px-2 py-1.5 text-left hover:border-white/30 disabled:cursor-not-allowed disabled:opacity-50"
          style={{ background: monitor && monitorDevice ? "rgb(var(--accent-rgb) / 0.12)" : "transparent" }}
        >
          <Icon name="speaker" size={14} className={monitor && monitorDevice ? "accent-text" : "text-muted"} />
          <span className="shrink-0 font-cn text-[13px] text-text">返听</span>
          <span className="truncate text-[11.5px] text-muted" title={monitorDevice ? `耳机同步听到变声后的自己 · ${monitorDevice}` : undefined}>
            {monitorDevice ? "耳机同步听自己" : outIsReal ? "OUT 已是耳机" : "没有可监听的耳机"}
          </span>
          <span className={`ml-auto shrink-0 font-num text-[13px] tracking-[0.2em] ${monitor && monitorDevice ? "accent-text" : "text-dim"}`}>
            {monitor && monitorDevice ? "ON" : "OFF"}
          </span>
        </button>
      )}
    </div>
  );
}
