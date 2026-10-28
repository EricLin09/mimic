import { useEffect, useRef, useState } from "react";
import { createMockEngine } from "./mock";
import { createTauriEngine } from "./tauri";
import type { EngineState, Telemetry } from "./types";

// inside the Tauri window the Rust engine answers; in a plain browser the mock does
const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
export const engine = inTauri ? createTauriEngine() : createMockEngine();
export const engineIsReal = inTauri;

export function useEngineState(): EngineState {
  const [s, setS] = useState<EngineState>(engine.state());
  useEffect(() => engine.onState(setS), []);
  return s;
}

/** Latest telemetry in a ref (for canvas/WebGL loops) without re-rendering React at 60 Hz. */
export function useTelemetryRef() {
  const ref = useRef<Telemetry | null>(null);
  useEffect(
    () =>
      engine.subscribe((t) => {
        ref.current = t;
      }),
    [],
  );
  return ref;
}

/** Throttled telemetry for text readouts. */
export function useTelemetry(hz = 12): Telemetry | null {
  const [t, setT] = useState<Telemetry | null>(null);
  useEffect(() => {
    let last = 0;
    return engine.subscribe((x) => {
      if (x.t - last >= 1000 / hz) {
        last = x.t;
        setT(x);
      }
    });
  }, [hz]);
  return t;
}
