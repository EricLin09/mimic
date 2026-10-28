import { defineConfig, searchForWorkspaceRoot, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
const host = process.env.TAURI_DEV_HOST;

// Voice data (models, portraits) stays in ../models and never enters the app source.
// In dev it is served from /voices; the Tauri build will read it through the asset protocol.
const VOICES_DIR = path.resolve(import.meta.dirname, "../models/voices");
const TYPES: Record<string, string> = { ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp", ".json": "application/json" };

function serveVoices(): Plugin {
  return {
    name: "mimic-serve-voices",
    // production bundle: copy the small per-voice assets next to the page (never the .onnx)
    closeBundle() {
      if (!fs.existsSync(VOICES_DIR)) return;
      const out = path.resolve(import.meta.dirname, "dist/voices");
      for (const voice of fs.readdirSync(VOICES_DIR, { withFileTypes: true })) {
        if (!voice.isDirectory()) continue;
        const src = path.join(VOICES_DIR, voice.name);
        for (const file of fs.readdirSync(src)) {
          if (!TYPES[path.extname(file).toLowerCase()]) continue;
          fs.mkdirSync(path.join(out, voice.name), { recursive: true });
          fs.copyFileSync(path.join(src, file), path.join(out, voice.name, file));
        }
      }
    },
    configureServer(server) {
      server.middlewares.use("/voices", (req, res, next) => {
        const rel = decodeURIComponent((req.url ?? "").split("?")[0]);
        const file = path.resolve(VOICES_DIR, "." + rel);
        const type = TYPES[path.extname(file).toLowerCase()];
        if (!file.startsWith(VOICES_DIR) || !type || !fs.existsSync(file)) return next();
        res.setHeader("Content-Type", type);
        fs.createReadStream(file).pipe(res);
      });
    },
  };
}

// https://vite.dev/config/
export default defineConfig(() => ({
  plugins: [react(), tailwindcss(), serveVoices()],

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
    // voices are imported from ../models/voices, outside the project root
    fs: { allow: [searchForWorkspaceRoot(process.cwd()), VOICES_DIR] },
  },
}));
