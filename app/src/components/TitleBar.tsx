import type { EngineState } from "../engine/types";
import { LogoMark, Wordmark } from "./Brand";
import { LiveBars } from "./Hud";

const STATUS: Record<EngineState, { text: string; color: string }> = {
  loading: { text: "初始化", color: "var(--color-muted)" },
  ready: { text: "待命", color: "var(--color-ok)" },
  live: { text: "变声中", color: "var(--color-alert)" },
  error: { text: "故障", color: "var(--color-alert)" },
};

async function windowAction(action: "minimize" | "close") {
  try {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    const w = getCurrentWindow();
    await (action === "minimize" ? w.minimize() : w.close());
  } catch {
    /* running in a plain browser preview */
  }
}

export function TitleBar({ state }: { state: EngineState }) {
  const s = STATUS[state];
  return (
    <header data-tauri-drag-region className="relative z-10 flex h-12 shrink-0 items-center gap-4 px-5">
      <div data-tauri-drag-region className="flex items-center gap-3">
        <LogoMark size={24} />
        <Wordmark height={13} />
        <span className="label border border-line-strong px-1.5 py-px text-[9px]">DEV 0.1</span>
      </div>
      <div data-tauri-drag-region className="flex-1" />
      <div
        className="flex items-center gap-2 border px-2.5 py-1 font-cn text-[13px]"
        style={{ color: s.color, borderColor: "currentColor", clipPath: "polygon(8px 0, 100% 0, calc(100% - 8px) 100%, 0 100%)" }}
      >
        <LiveBars live={state === "live"} />
        <span>{s.text}</span>
      </div>
      <div className="flex">
        {(["minimize", "close"] as const).map((a) => (
          <button
            key={a}
            onClick={() => windowAction(a)}
            className={`flex h-8 w-10 cursor-pointer items-center justify-center text-muted transition hover:text-text ${a === "close" ? "hover:bg-alert/80" : "hover:bg-white/8"}`}
            aria-label={a === "close" ? "关闭" : "最小化"}
          >
            <svg width="10" height="10" viewBox="0 0 10 10" stroke="currentColor" strokeWidth="1.2">
              {a === "close" ? <path d="M1 1 L9 9 M9 1 L1 9" /> : <path d="M1 5 H9" />}
            </svg>
          </button>
        ))}
      </div>
    </header>
  );
}
