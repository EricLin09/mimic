import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import type { Telemetry } from "../engine/types";

const YOU = "79 216 255";

/** Shared canvas loop: sizes the canvas to its box (DPR aware) and calls draw every frame. */
function useCanvasLoop(draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    const loop = () => {
      const dpr = window.devicePixelRatio || 1;
      const { width, height } = canvas.getBoundingClientRect();
      const W = Math.round(width * dpr), H = Math.round(height * dpr);
      if (canvas.width !== W || canvas.height !== H) {
        canvas.width = W;
        canvas.height = H;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      drawRef.current(ctx, width, height);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);
  return ref;
}

function cssRgb(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "242 193 78";
}

/** Glowing dual oscilloscope: your voice (cyan) and the converted voice (accent). */
export function WaveRibbon({ telemetry, active }: { telemetry: RefObject<Telemetry | null>; active: boolean }) {
  const smoothIn = useRef<Float32Array | null>(null);
  const smoothOut = useRef<Float32Array | null>(null);
  const ref = useCanvasLoop((ctx, w, h) => {
    const t = telemetry.current;
    if (!t) return;
    const ease = (prev: Float32Array | null, next: Float32Array) => {
      if (!prev || prev.length !== next.length) return Float32Array.from(next);
      for (let i = 0; i < next.length; i++) prev[i] += (next[i] - prev[i]) * 0.35;
      return prev;
    };
    smoothIn.current = ease(smoothIn.current, t.waveIn);
    smoothOut.current = ease(smoothOut.current, t.waveOut);
    const line = (data: Float32Array, rgb: string, alpha: number, gain: number, width: number) => {
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const x = (i / (data.length - 1)) * w;
        const taper = Math.sin((i / (data.length - 1)) * Math.PI);
        const y = h / 2 + data[i] * gain * taper * (h * 0.42);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `rgb(${rgb} / ${alpha})`;
      ctx.lineWidth = width;
      ctx.shadowColor = `rgb(${rgb} / 0.9)`;
      ctx.shadowBlur = 12;
      ctx.stroke();
      ctx.shadowBlur = 0;
    };
    ctx.strokeStyle = "rgb(255 255 255 / 0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
    line(smoothIn.current, YOU, active ? 0.45 : 0.95, 1.4, active ? 1 : 1.6);
    if (active) line(smoothOut.current, cssRgb("--accent-rgb"), 0.95, 1.5, 1.8);
  });
  return <canvas ref={ref} className="h-full w-full" />;
}

const F_MIN = 70, F_MAX = 240;
const yOf = (f: number, h: number) => h - (Math.log(f / F_MIN) / Math.log(F_MAX / F_MIN)) * h;

/** Scrolling pitch history: input (cyan), output (accent), target band of the voice. */
export function PitchTrace({ telemetry, targetF0 }: { telemetry: RefObject<Telemetry | null>; targetF0: number }) {
  const hist = useRef<{ t: number; i: number; o: number }[]>([]);
  const ref = useCanvasLoop((ctx, w, h) => {
    const t = telemetry.current;
    if (!t) return;
    const H = hist.current;
    if (!H.length || t.t - H[H.length - 1].t > 15) H.push({ t: t.t, i: t.f0In, o: t.f0Out });
    const span = 6000;
    while (H.length && t.t - H[0].t > span) H.shift();
    const accent = cssRgb("--accent-rgb");

    // target band
    const top = yOf(targetF0 * 1.12, h), bottom = yOf(targetF0 / 1.12, h);
    ctx.fillStyle = `rgb(${accent} / 0.07)`;
    ctx.fillRect(0, top, w, bottom - top);
    ctx.strokeStyle = `rgb(${accent} / 0.25)`;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(0, yOf(targetF0, h));
    ctx.lineTo(w, yOf(targetF0, h));
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = "9px 'JetBrains Mono'";
    ctx.fillStyle = "rgb(255 255 255 / 0.28)";
    for (const f of [80, 120, 160, 220]) {
      const y = yOf(f, h);
      ctx.fillRect(w - 26, y, 4, 1);
      ctx.fillText(String(f), w - 20, y + 3);
    }

    const series = (key: "i" | "o", rgb: string, width: number) => {
      ctx.beginPath();
      let pen = false;
      for (const p of H) {
        const f = p[key];
        const x = ((p.t - (t.t - span)) / span) * (w - 30);
        if (f <= 0) {
          pen = false;
          continue;
        }
        const y = yOf(f, h);
        if (pen) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
        pen = true;
      }
      ctx.strokeStyle = `rgb(${rgb} / 0.95)`;
      ctx.lineWidth = width;
      ctx.shadowColor = `rgb(${rgb} / 0.8)`;
      ctx.shadowBlur = 8;
      ctx.stroke();
      ctx.shadowBlur = 0;
    };
    series("i", YOU, 1.2);
    series("o", accent, 2);
  });
  return <canvas ref={ref} className="h-full w-full" />;
}

/** Tiny history line for a numeric telemetry field. */
export function Sparkline({ telemetry, pick, max }: {
  telemetry: RefObject<Telemetry | null>;
  pick: (t: Telemetry) => number;
  max: number;
}) {
  const hist = useRef<number[]>([]);
  const last = useRef(0);
  const ref = useCanvasLoop((ctx, w, h) => {
    const t = telemetry.current;
    if (!t) return;
    if (t.t - last.current > 60) {
      last.current = t.t;
      hist.current.push(pick(t));
      if (hist.current.length > 80) hist.current.shift();
    }
    const H = hist.current;
    const accent = cssRgb("--accent-rgb");
    ctx.beginPath();
    H.forEach((v, i) => {
      const x = (i / 79) * w;
      const y = h - Math.min(1, v / max) * (h - 2) - 1;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = `rgb(${accent} / 0.9)`;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.lineTo(((H.length - 1) / 79) * w, h);
    ctx.lineTo(0, h);
    ctx.fillStyle = `rgb(${accent} / 0.08)`;
    ctx.fill();
  });
  return <canvas ref={ref} className="h-full w-full" />;
}

/** Segmented level meter with peak hold. */
export function LevelMeter({ telemetry, pick, rgb }: {
  telemetry: RefObject<Telemetry | null>;
  pick: (t: Telemetry) => number;
  rgb?: string;
}) {
  const peak = useRef(0);
  const ref = useCanvasLoop((ctx, w, h) => {
    const t = telemetry.current;
    const v = t ? Math.min(1, pick(t) * 1.25) : 0;
    peak.current = Math.max(v, peak.current - 0.008);
    const color = rgb ?? cssRgb("--accent-rgb");
    const n = 24, gap = 2, seg = (w - gap * (n - 1)) / n;
    for (let i = 0; i < n; i++) {
      const on = i / n < v;
      const hot = i >= n - 4;
      ctx.fillStyle = on ? (hot ? "rgb(255 70 85 / 0.95)" : `rgb(${color} / 0.9)`) : "rgb(255 255 255 / 0.06)";
      ctx.fillRect(i * (seg + gap), 0, seg, h);
    }
    const px = Math.min(n - 1, Math.floor(peak.current * n)) * (seg + gap);
    ctx.fillStyle = "rgb(255 255 255 / 0.85)";
    ctx.fillRect(px, 0, 2, h);
  });
  return <canvas ref={ref} className="h-full w-full" />;
}
