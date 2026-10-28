import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { LogoMark, Wordmark } from "./Brand";

const LATIN_POOL = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789#%&*/";
const CJK_POOL = "声纹解码变换音色频谱共鸣相位滤波采样合成";

/** Scrambles into `text`, left to right, whenever it changes. */
export function DecodeText({ text, duration = 650, className }: { text: string; duration?: number; className?: string }) {
  const [shown, setShown] = useState(text);
  useEffect(() => {
    const pool = /[一-鿿]/.test(text) ? CJK_POOL : LATIN_POOL;
    const chars = [...text];
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / duration);
      setShown(chars.map((ch, i) => (i < p * chars.length || ch === " " ? ch : pool[(Math.random() * pool.length) | 0])).join(""));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [text, duration]);
  return <span className={className}>{shown}</span>;
}

/** Eases toward the latest value; only animates while it differs. */
export function AnimatedNumber({ value, decimals = 0 }: { value: number | null; decimals?: number }) {
  const [shown, setShown] = useState(value ?? 0);
  const current = useRef(value ?? 0);
  useEffect(() => {
    if (value === null) return;
    let raf = 0;
    const eps = 0.5 * 10 ** -decimals;
    const step = () => {
      current.current += (value - current.current) * 0.22;
      if (Math.abs(value - current.current) < eps) current.current = value;
      setShown(current.current);
      if (current.current !== value) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value, decimals]);
  return <>{value === null ? "—" : shown.toFixed(decimals)}</>;
}

const BOOT_LINES = [
  ["声学特征", "contentvec.onnx"],
  ["音高网络", "rmvpe.onnx"],
  ["角色声线", "synth.onnx"],
  ["音色索引", "retrieval.onnx"],
  ["音频路由", "WASAPI · 虚拟麦克风"],
];

/** Start-up sequence: the mask assembles, then the engine check list types out. Click to skip. */
export function BootSplash() {
  const [open, setOpen] = useState(true);
  const [lines, setLines] = useState(0);
  useEffect(() => {
    const timers = BOOT_LINES.map((_, i) => setTimeout(() => setLines(i + 1), 450 + i * 190));
    timers.push(setTimeout(() => setOpen(false), 450 + BOOT_LINES.length * 190 + 450));
    return () => timers.forEach(clearTimeout);
  }, []);
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="absolute inset-0 z-50 flex cursor-pointer items-center justify-center bg-ink"
          exit={{ opacity: 0, scale: 1.04, filter: "blur(8px)" }}
          transition={{ duration: 0.45 }}
          onClick={() => setOpen(false)}
        >
          <div className="grid-overlay absolute inset-0" />
          <div className="relative flex w-[420px] flex-col items-center">
            <motion.div initial={{ scale: 0.6, opacity: 0, rotate: -8 }} animate={{ scale: 1, opacity: 1, rotate: 0 }} transition={{ type: "spring", stiffness: 160, damping: 14 }}>
              <LogoMark size={96} />
            </motion.div>
            <motion.div className="mt-5" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
              <Wordmark height={26} />
            </motion.div>
            <div className="label mt-2 tracking-[0.5em]">VOICE MIMICRY ENGINE</div>
            <div className="mt-8 w-full space-y-1.5 font-mono text-[11px]">
              {BOOT_LINES.slice(0, lines).map(([k, v]) => (
                <motion.div key={k} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} className="flex items-center gap-2 text-muted">
                  <span className="text-dim">&gt;</span>
                  <span className="text-text">{k}</span>
                  <span className="truncate text-dim">{v}</span>
                  <span className="flex-1 border-b border-dotted border-line-strong" />
                  <span className="text-ok">OK</span>
                </motion.div>
              ))}
            </div>
            <div className="mt-5 h-[3px] w-full overflow-hidden bg-white/5">
              <motion.div
                className="h-full"
                style={{ background: "linear-gradient(90deg, #4fd8ff, var(--accent))" }}
                initial={{ width: "0%" }}
                animate={{ width: `${(lines / BOOT_LINES.length) * 100}%` }}
                transition={{ duration: 0.25 }}
              />
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/** Three bars that bounce while live. */
export function LiveBars({ live }: { live: boolean }) {
  return (
    <span className="flex h-3 items-end gap-[2px]">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-[3px] bg-current"
          style={{
            height: live ? undefined : 3,
            animation: live ? `eq 0.${7 + i}s ease-in-out ${i * 0.12}s infinite alternate` : undefined,
          }}
        />
      ))}
    </span>
  );
}
