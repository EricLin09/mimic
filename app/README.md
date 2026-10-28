# Mimic app (Tauri 2 + React)

The showpiece UI over the Rust engine in `../engine`. The pipeline, the ground rules and the
model files are described in [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

```bash
npm install
npm run tauri dev      # Rust bridge + Vite dev server
npm run dev            # UI only, in a browser, against the mock engine
npm run tauri build    # installer under src-tauri/target/release/bundle
```

- `src/engine/` is the contract with the engine: `tauri.ts` talks to `src-tauri/src/lib.rs` over
  IPC, `mock.ts` fakes telemetry so every widget can be tuned without audio hardware.
- Voice assets (portraits, backgrounds) are read from `../models/voices/<id>/` and never bundled
  with the source; `vite.config.ts` serves them in dev and copies the small files next to the page
  on build.
- `scripts/render-icon.mjs` regenerates the app icon and `docs/brand/*.svg` from the same mask
  paths as `src/components/Brand.tsx` (keep the two in sync).
