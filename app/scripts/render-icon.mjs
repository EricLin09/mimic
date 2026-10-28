// Renders the Mimic app icon (split mask on a chamfered dark tile) to a 1024px PNG for
// `tauri icon`, and saves the vector sources to docs/brand.
import { Resvg } from "@resvg/resvg-js";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const app = path.resolve(here, "..");
const brand = path.resolve(app, "../docs/brand");

// keep in sync with src/components/Brand.tsx
const MASK = "M30 12 H98 L116 32 V70 L84 116 H44 L12 70 V32 Z";
const SEAM = "M66 0 L60 22 L68 36 L61 50 L67 62 L58 80 L68 98 L61 128";
const LEFT = `${SEAM} H0 V0 Z`;
const RIGHT = `${SEAM} H128 V0 Z`;
const EYE_L = "M30 50 L54 45 L57 55 L33 60 Z";
const EYE_R = "M98 50 L74 45 L71 55 L95 60 Z";
const CYAN = "#4fd8ff", GOLD = "#f2c14e";

const mark = (id) => `
  <defs>
    <clipPath id="${id}m"><path d="${MASK}"/></clipPath>
    <linearGradient id="${id}l" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${CYAN}"/><stop offset="1" stop-color="${CYAN}" stop-opacity="0.55"/></linearGradient>
    <linearGradient id="${id}r" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${GOLD}"/><stop offset="1" stop-color="${GOLD}" stop-opacity="0.6"/></linearGradient>
  </defs>
  <g clip-path="url(#${id}m)">
    <path d="${LEFT}" fill="url(#${id}l)"/>
    <path d="${RIGHT}" fill="url(#${id}r)"/>
    <path d="${SEAM}" fill="none" stroke="#06070a" stroke-width="4"/>
    <path d="${EYE_L}" fill="#06070a"/>
    <path d="${EYE_R}" fill="#06070a"/>
  </g>
  <path d="${MASK}" fill="none" stroke="#ffffff" stroke-opacity="0.38" stroke-width="2"/>`;

const TILE = "M16 2 H112 L126 16 V112 L112 126 H16 L2 112 V16 Z";
const icon = `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 128 128">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#141821"/><stop offset="1" stop-color="#06070a"/></linearGradient>
    <radialGradient id="glow" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="${GOLD}" stop-opacity="0.28"/><stop offset="0.6" stop-color="${CYAN}" stop-opacity="0.06"/><stop offset="1" stop-color="${CYAN}" stop-opacity="0"/></radialGradient>
  </defs>
  <path d="${TILE}" fill="url(#bg)"/>
  <circle cx="64" cy="64" r="58" fill="url(#glow)"/>
  <path d="${TILE}" fill="none" stroke="${GOLD}" stroke-opacity="0.45" stroke-width="1.4"/>
  <path d="M2 30 V16 L16 2 H30" fill="none" stroke="${GOLD}" stroke-width="2.4"/>
  <path d="M126 98 V112 L112 126 H98" fill="none" stroke="${GOLD}" stroke-width="2.4"/>
  <g transform="translate(64 64) scale(0.74) translate(-64 -64)">${mark("i")}</g>
</svg>`;

const logo = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 128 128">${mark("g")}</svg>`;

fs.mkdirSync(brand, { recursive: true });
fs.writeFileSync(path.join(brand, "mimic-icon.svg"), icon);
fs.writeFileSync(path.join(brand, "mimic-mark.svg"), logo);
const png = new Resvg(icon, { fitTo: { mode: "width", value: 1024 } }).render().asPng();
fs.writeFileSync(path.join(app, "src-tauri/icons/app-icon.png"), png);
fs.writeFileSync(path.join(brand, "mimic-icon-1024.png"), png);
console.log("icon written", png.length, "bytes");
