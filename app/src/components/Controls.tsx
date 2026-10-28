import { useTelemetry } from "../engine/useEngine";
import { AnimatePresence, motion } from "motion/react";
import type { EngineState, VoiceParams } from "../engine/types";
import { Icon, type IconName } from "./Icons";

export function ActivateButton({ state, onToggle }: { state: EngineState; onToggle: () => void }) {
  const live = state === "live";
  const busy = state === "loading";
  return (
    <div className="relative flex flex-col items-center gap-3">
      {/* pulse rings behind the button while live */}
      <AnimatePresence>
        {live &&
          [0, 1, 2].map((i) => (
            <motion.span
              key={i}
              className="pointer-events-none absolute top-1/2 left-1/2 h-14 w-72 -translate-x-1/2 -translate-y-[60%] border"
              style={{ borderColor: "rgb(var(--accent-rgb) / 0.5)", clipPath: "polygon(6% 0, 100% 0, 94% 100%, 0 100%)" }}
              initial={{ opacity: 0.6, scale: 1 }}
              animate={{ opacity: 0, scale: 1.45 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 2.4, repeat: Infinity, delay: i * 0.8, ease: "easeOut" }}
            />
          ))}
      </AnimatePresence>
      <motion.button
        onClick={onToggle}
        disabled={busy}
        whileHover={{ scale: 1.03 }}
        whileTap={{ scale: 0.97 }}
        className="relative flex h-14 w-72 cursor-pointer items-center justify-center gap-3 overflow-hidden font-cn text-[22px] tracking-[0.18em] disabled:cursor-wait"
        style={{
          clipPath: "polygon(6% 0, 100% 0, 94% 100%, 0 100%)",
          background: live
            ? "linear-gradient(100deg, rgb(var(--accent-rgb)) 0%, rgb(var(--accent-rgb) / 0.75) 100%)"
            : "rgb(var(--accent-rgb) / 0.08)",
          color: live ? "#0b0b0b" : "var(--accent)",
          boxShadow: live ? "0 0 40px rgb(var(--accent-rgb) / 0.45)" : "none",
        }}
      >
        {!live && (
          <span className="absolute inset-0 border" style={{ borderColor: "rgb(var(--accent-rgb) / 0.6)", clipPath: "inherit" }} />
        )}
        {(busy || live) && (
          <motion.span
            className="absolute inset-y-0 w-1/3"
            style={{ background: `linear-gradient(90deg, transparent, ${live ? "rgb(255 255 255 / 0.35)" : "rgb(var(--accent-rgb) / 0.45)"}, transparent)` }}
            initial={{ x: "-120%" }}
            animate={{ x: "320%" }}
            transition={{ duration: live ? 2.2 : 0.9, repeat: Infinity, ease: "easeInOut", repeatDelay: live ? 1.2 : 0 }}
          />
        )}
        <Icon name="power" size={20} className="relative" strokeWidth={2.2} />
        <span className="relative">{busy ? "接入中" : live ? "变声中" : "启动变声"}</span>
        {live && <span className="relative font-num text-xl tracking-[0.2em]">LIVE</span>}
      </motion.button>
      <div className="label flex items-center gap-2">
        {["Ctrl", "Alt", "V"].map((k) => (
          <kbd
            key={k}
            className="border border-line-strong bg-white/5 px-1.5 py-0.5 font-mono text-[10px] text-text normal-case tracking-normal"
            style={{ boxShadow: "inset 0 -2px 0 rgb(255 255 255 / 0.08)" }}
          >
            {k}
          </kbd>
        ))}
        <span>随时切换</span>
      </div>
    </div>
  );
}

/** Engine errors arrive as English technical text; say what matters in a few words. */
function friendlyError(e: string): string {
  const dev = /"([^"]+)"/.exec(e)?.[1];
  const on = dev ? `：${dev}` : "";
  // WASAPI codes: 0x8889000A in use, 0x88890008 format refused, 0x88890004 unplugged
  const why = e.includes("-2004287478") ? "（被别的程序占着）" : e.includes("-2004287480") ? "（不支持 48 kHz）" : e.includes("-2004287484") ? "（已断开）" : "";
  if (e.startsWith("monitor skipped")) return `返听设备打不开${why}，已跳过${on}`;
  if (e.includes("opening output")) return `输出设备打不开${why}${on}。给游戏/微信用请选 CABLE Input`;
  if (e.includes("opening input")) return `麦克风打不开${why}${on}`;
  if (/no (input|output) device matches|no default/.test(e)) return "找不到选中的音频设备，可能已断开";
  if (e.includes("did not start within")) return "音频设备 15 秒内没有启动";
  return e;
}

/** One line under the start button: why the last start failed, or what was skipped while live. */
export function EngineNotice({ state, message }: { state: EngineState; message: string | null }) {
  if (!message || (state !== "error" && state !== "live")) return null;
  const failed = state === "error";
  return (
    <div
      className="absolute bottom-full left-1/2 z-10 mb-3 w-max max-w-[560px] -translate-x-1/2 border px-3 py-1.5 text-center font-cn text-[13px] leading-snug"
      style={{
        color: failed ? "#ff8a8a" : "var(--color-muted)",
        background: "rgb(8 8 10 / 0.88)",
        borderColor: failed ? "rgb(255 107 107 / 0.45)" : "rgb(255 255 255 / 0.12)",
      }}
      title={message}
    >
      {failed ? `启动失败：${friendlyError(message)} · 再点一次重试` : friendlyError(message)}
    </div>
  );
}

interface SliderProps {
  icon: IconName;
  label: string;
  /** one short line under the slider */
  hint?: string;
  /** the full explanation, shown on hover */
  title?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
}

export function TacSlider({ icon, label, hint, title, value, min, max, step, format, onChange }: SliderProps) {
  const fill = ((value - min) / (max - min)) * 100;
  return (
    <label className="block" title={title}>
      <div className="flex items-center gap-2">
        <Icon name={icon} size={14} className="text-muted" />
        <span className="font-cn text-[15px] text-text">{label}</span>
        <span className="ml-auto font-num text-[22px] leading-none accent-text">{format(value)}</span>
      </div>
      <input
        type="range"
        className="tac -my-0.5"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ ["--fill" as string]: `${fill}%` }}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <div className="text-[11.5px] leading-snug text-muted">{hint}</div>}
    </label>
  );
}

export function TacToggle({ icon, label, hint, title, value, onChange }: {
  icon: IconName; label: string; hint?: string; title?: string; value: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <button type="button" onClick={() => onChange(!value)} aria-pressed={value} title={title} className="block w-full text-left">
      <div className="flex items-center gap-2">
        <Icon name={icon} size={14} className="text-muted" />
        <span className="font-cn text-[15px] text-text">{label}</span>
        <span className={`ml-auto font-num text-[15px] tracking-[0.2em] ${value ? "accent-text" : "text-dim"}`}>{value ? "ON" : "OFF"}</span>
        <span className="relative h-4 w-8 border border-line-strong" style={{ background: value ? "rgb(var(--accent-rgb) / 0.25)" : "rgb(255 255 255 / 0.04)" }}>
          <span
            className="absolute top-0.5 h-2.5 w-3 transition-all"
            style={{ left: value ? 16 : 2, background: value ? "var(--accent)" : "var(--color-muted)", boxShadow: value ? "0 0 8px var(--accent)" : undefined }}
          />
        </span>
      </div>
      {hint && <div className="text-[11.5px] leading-snug text-muted">{hint}</div>}
    </button>
  );
}

/** Live input level (after denoising) against the gate threshold, so the slider can be set by eye. */
function GateMeter({ threshold, min, max, source = "level" }: { threshold: number; min: number; max: number; source?: "level" | "vad" }) {
  const t = useTelemetry(15);
  const db = source === "vad" ? (t?.vad ?? 0) : (t?.gateDb ?? -90);
  const open = t?.gateOpen ?? false;
  const pct = (v: number) => Math.max(0, Math.min(100, ((v - min) / (max - min)) * 100));
  return (
    <div className="relative mt-1 h-1.5 w-full bg-white/5">
      <div
        className="absolute inset-y-0 left-0 transition-[width] duration-75"
        style={{ width: `${pct(db)}%`, background: open ? "var(--accent)" : "var(--color-muted)", opacity: open ? 0.9 : 0.45 }}
      />
      <div className="absolute -top-0.5 h-2.5 w-px bg-white" style={{ left: `${pct(threshold)}%` }} />
    </div>
  );
}

export function TuningPanel({ params, onChange }: { params: VoiceParams; onChange: (p: VoiceParams) => void }) {
  const set = (k: "pitch" | "indexRate" | "protect" | "envelope" | "blockMs" | "gateDb" | "vad") => (v: number) => onChange({ ...params, [k]: v } as VoiceParams);
  // block + 60 ms lookahead + ~50 ms inference + ~30 ms cushion + device buffers
  const latency = Math.round(params.blockMs + 160);
  return (
    <div className="space-y-2.5">
      <TacSlider icon="pitch" label="音高" hint="0 = 对齐角色的音高中心" title="引擎按角色原声的音高中位数（声线目录里的 f0_stats.json）对齐你的声音，0 就是对齐后的位置；往左更低沉，往右更高"
        value={params.pitch} min={-12} max={12} step={1} format={(v) => `${v > 0 ? "+" : ""}${v} ST`} onChange={set("pitch")} />
      <TacSlider icon="index" label="音色贴合" hint="越高越像角色，差别细微"
        title="检索混合：把你的发音特征按这个比例换成训练集里最接近的角色特征。越高越贴角色、越少漏出你本人的音色底子，过高咬字发糊。作用细微，长句对比才听得出"
        value={params.indexRate} min={0} max={1} step={0.05} format={(v) => v.toFixed(2)} onChange={set("indexRate")} />
      <TacSlider icon="shield" label="辅音保护" hint="越低齿音越清楚 · 0.5 = 关"
        title="只作用于不带音高的清辅音帧（s、sh、f、h、ch）：越低这些帧越多保留你的原始特征，齿音越清楚；越高越像角色但齿音易发毛。元音不受影响，所以整体差别很小"
        value={params.protect} min={0} max={0.5} step={0.01} format={(v) => v.toFixed(2)} onChange={set("protect")} />
      <TacSlider icon="speaker" label="音量跟随" hint="响度起伏跟着你说话的轻重走 · 0 = 关"
        title="只跟起伏，不改整体音量：每 10 ms 把变声后的响度往你本人说话的轻重方向拉，最多 ±6 dB。0 = 完全保持模型自己的响度变化"
        value={params.envelope} min={0} max={1} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={set("envelope")} />
      <TacSlider icon="blocks" label="延迟档位" hint={`预计延迟 ≈ ${latency} ms · 改动会重启引擎`} title="240 音质最佳，160 已与 Applio 持平"
        value={params.blockMs} min={120} max={300} step={20} format={(v) => `${v} MS`} onChange={set("blockMs")} />
      <div>
        <TacSlider icon="gauge" label="输入门限" hint="响度低于线不转换"
          title="第一道：块响度（150 Hz 以上、降噪后）低于门限就不转换。说话时条在线右边，安静和旁人远处说话时在线左边。回差 6 dB，松开后保持 0.4 s"
          value={params.gateDb} min={-70} max={-20} step={1} format={(v) => `${v} dB`} onChange={set("gateDb")} />
        <GateMeter threshold={params.gateDb} min={-70} max={-20} />
      </div>
      <div>
        <TacSlider icon="target" label="人声门限" hint="不像人声不转换 · 0 = 关"
          title="第二道：Silero VAD 每 32 ms 给一个「是人声」的概率，一块里有一段超过这条线门才开，开门后低 15 个点才算没人声。摩擦耳机、手机放桌、键盘这类响但不是人声的声音挡在这里。条是当前块的最高概率"
          value={params.vad} min={0} max={1} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={set("vad")} />
        <GateMeter threshold={params.vad} min={0} max={1} source="vad" />
      </div>
      <TacToggle icon="wave" label="降噪进引擎" title="开了会先用 RNNoise 去噪再变声：键盘贴麦、风扇很吵时更干净，但安静时也会改写你的声音细节，所以默认关。门限判定始终用去噪后的信号，不受这个开关影响"
        value={params.denoise} onChange={(v) => onChange({ ...params, denoise: v })} />
    </div>
  );
}
