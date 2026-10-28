import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import type { Telemetry } from "../engine/types";
import { dialPos, fullscreenProgram, glslNoise, hexToVec3, trackCanvasSize } from "./webgl";

// The agent "materialises": idle = cyan hologram (scanlines, chroma split, band glitches),
// live = full colour sweeping up from the feet, with a voice-driven aura from the
// silhouette (mip-blurred alpha), rising embers, the emblem backdrop and a projector
// base whose front arc doubles as the pitch dial.
const FRAG = `#version 300 es
precision highp float;
uniform vec2 uRes;
uniform float uAspect;
uniform float uTime;
uniform float uLevel;
uniform float uProgress;
uniform float uFlash;
uniform float uVoiced;
uniform float uPitch;
uniform vec3 uIdle;
uniform vec3 uLive;
uniform vec3 uTint;
uniform sampler2D uAgent;
uniform vec4 uAgentRect;
uniform vec4 uAgentUV;
uniform sampler2D uBg;
uniform vec4 uBgRect;
uniform float uHasBg;
out vec4 outColor;
${glslNoise(4)}

vec4 agentAt(vec2 p, float lod) {
  vec2 a = (p - uAgentRect.xy) / uAgentRect.zw;
  vec2 uv = vec2(uAgentUV.x + a.x * uAgentUV.z, uAgentUV.y + (1.0 - a.y) * uAgentUV.w);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return vec4(0.0);
  return textureLod(uAgent, uv, lod);
}

void main() {
  vec2 p = gl_FragCoord.xy / uRes;
  vec2 q = vec2((p.x - (uAgentRect.x + uAgentRect.z * 0.5)) * uAspect, p.y);
  float t = uTime;
  float act = smoothstep(0.0, 1.0, uProgress);
  vec3 col = mix(uIdle, uLive, act);
  vec3 back = vec3(0.0);
  vec3 front = vec3(0.0);

  // emblem backdrop with a slow scan band
  if (uHasBg > 0.5) {
    vec2 b = (p - uBgRect.xy) / uBgRect.zw;
    if (b.x >= 0.0 && b.x <= 1.0 && b.y >= 0.0 && b.y <= 1.0) {
      float bgA = texture(uBg, vec2(b.x, 1.0 - b.y)).a;
      float sweep = exp(-abs(fract(b.y * 0.6 - t * 0.05) - 0.5) * 16.0);
      vec3 tint = mix(uTint * 0.55 + uIdle * 0.04, uTint * 0.35 + uLive * 0.12, act);
      back += tint * bgA * (0.5 + 0.6 * sweep + 0.5 * uLevel * act);
    }
  }

  // projector base under the feet, its front arc is the pitch dial
  float feet = uAgentRect.y + 0.004;
  vec2 eb = vec2(q.x / 0.27, (q.y - feet) / 0.04);
  float er = length(eb);
  back += col * (exp(-abs(er - 1.0) * 26.0) * (0.45 + 0.9 * uLevel) + smoothstep(1.0, 0.0, er) * 0.22 * (0.4 + uLevel));
  float coneW = 0.16 + (q.y - feet) * 0.22;
  back += uIdle * smoothstep(coneW, 0.0, abs(q.x)) * smoothstep(feet, feet + 0.06, q.y) * smoothstep(1.0, 0.25, q.y) * 0.06 * (1.0 - act);
  float ang = atan(eb.y, eb.x);
  float onArc = step(-2.8, ang) * step(ang, -0.34);
  float along = (ang + 2.8) / 2.46;
  float arcBand = exp(-abs(er - 1.2) * 34.0) * onArc;
  back += col * arcBand * (0.12 + 0.5 * step(0.72, fract(along * 18.0))) * (0.4 + 0.6 * act);
  float na = mix(-2.8, -0.34, clamp(uPitch, 0.0, 1.0));
  front += (col * 0.6 + 0.5) * exp(-length(eb - vec2(cos(na), sin(na)) * 1.2) * 7.0) * uVoiced * (0.5 + act);

  // silhouette aura from mip-blurred alpha
  float a0 = agentAt(p, 0.0).a;
  float b1 = agentAt(p, 3.0).a;
  float b2 = agentAt(p, 4.5).a;
  float b3 = agentAt(p, 6.0).a;
  float halo = clamp(b1 * 0.55 + b2 * 0.4 + b3 * 0.35 - a0, 0.0, 1.0);
  float flow = fbm(vec2(p.x * 7.0, p.y * 5.0 - t * 0.9));
  back += col * halo * mix(0.2, 0.4 + 1.7 * uLevel, act) * (0.55 + 0.9 * flow);

  // figure: band glitches + chroma split, hologram vs materialised
  float band = floor(p.y * 64.0);
  float gt = floor(t * 12.0);
  float slipAmt = mix(0.01, 0.0015, act) + uFlash * 0.03;
  float slip = hash(vec2(band, gt)) > 0.94 ? (hash(vec2(band, gt + 7.0)) - 0.5) * slipAmt * 2.0 : 0.0;
  vec2 pg = p + vec2(slip, 0.0);
  float ca = mix(0.0032, 0.0005, act) + uFlash * 0.007;
  vec4 c0 = agentAt(pg, 0.0);
  vec4 agent = vec4(agentAt(pg + vec2(ca, 0.0), 0.0).r, c0.g, agentAt(pg - vec2(ca, 0.0), 0.0).b, c0.a);
  vec3 straight = agent.a > 0.002 ? agent.rgb / agent.a : vec3(0.0);
  float lum = dot(straight, vec3(0.299, 0.587, 0.114));
  vec2 px = 2.0 / uRes;
  float gx = agentAt(pg + vec2(px.x, 0.0), 1.0).a - agentAt(pg - vec2(px.x, 0.0), 1.0).a;
  float gy = agentAt(pg + vec2(0.0, px.y), 1.0).a - agentAt(pg - vec2(0.0, px.y), 1.0).a;
  float rim = clamp(length(vec2(gx, gy)) * 1.5, 0.0, 1.0);

  float scan = 0.7 + 0.3 * sin(gl_FragCoord.y * 1.35 + t * 6.0);
  float flick = 0.9 + 0.1 * noise(vec2(t * 18.0, 0.0));
  vec3 holo = uIdle * (0.16 + 1.05 * lum) * scan * flick + uIdle * rim * 0.9;
  vec3 real = straight * (0.92 + 0.12 * uLevel) + uLive * rim * (0.3 + 0.9 * uLevel);

  float ly = (p.y - uAgentRect.y) / uAgentRect.w;
  float edge = mix(-0.08, 1.12, uProgress) + 0.035 * (noise(vec2(p.x * 38.0, t * 2.0)) - 0.5);
  float solid = smoothstep(edge + 0.012, edge - 0.012, ly);
  vec3 figure = mix(holo, real, solid);
  float figA = mix(agent.a * 0.8, agent.a, solid);
  float during = smoothstep(0.0, 0.04, uProgress) * smoothstep(1.0, 0.96, uProgress);
  front += uLive * exp(-abs(ly - edge) * 60.0) * during * clamp(b2 * 2.5, 0.0, 1.0) * 2.4;

  // embers rising near the body
  vec2 gp = vec2(p.x * 90.0, p.y * 45.0 - t * (1.2 + 3.0 * uLevel));
  vec2 id = floor(gp);
  float rnd = hash(id);
  vec2 ctr = id + vec2(0.5) + (vec2(hash(id + 3.1), hash(id + 7.7)) - 0.5) * 0.7;
  float ember = exp(-length((gp - ctr) / vec2(90.0, 45.0) * uRes) * 0.5) * step(0.86, rnd) * (0.5 + 0.5 * sin(t * 7.0 + rnd * 40.0));
  front += col * ember * clamp(b3 * 3.5, 0.0, 1.0) * mix(0.5, 1.3 + 2.0 * uLevel, act);

  // activation shockwave
  vec2 c = uAgentRect.xy + uAgentRect.zw * vec2(0.5, 0.55);
  float r = length((p - c) * vec2(uAspect, 1.0));
  front += uLive * exp(-abs(r - (1.0 - uFlash) * 0.95) * 38.0) * uFlash * 1.6;

  vec3 backL = vec3(1.0) - exp(-back * 1.3);
  vec3 frontL = vec3(1.0) - exp(-front * 1.3);
  vec3 rgb = backL * (1.0 - figA) + figure * figA + frontL;
  float alpha = max(figA, max(max(backL.r, max(backL.g, backL.b)), max(frontL.r, max(frontL.g, frontL.b))));
  outColor = vec4(min(rgb, vec3(1.0)), clamp(alpha, 0.0, 1.0));
}`;

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

/** Normalised bounding box of the non-transparent pixels (u0, v0, du, dv; v from the top). */
function alphaBox(img: HTMLImageElement): [number, number, number, number] {
  const w = 256, h = Math.round((img.height / img.width) * 256);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(img, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  let x0 = w, y0 = h, x1 = 0, y1 = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (d[(y * w + x) * 4 + 3] > 10) {
        x0 = Math.min(x0, x); x1 = Math.max(x1, x);
        y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      }
  return x1 < x0 ? [0, 0, 1, 1] : [x0 / w, y0 / h, (x1 - x0 + 1) / w, (y1 - y0 + 1) / h];
}

function texture(gl: WebGL2RenderingContext, img: HTMLImageElement, unit: number) {
  const tex = gl.createTexture()!;
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  // premultiply so the colour hidden under alpha=0 cannot bleed into filtered edges
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

interface Props {
  telemetry: RefObject<Telemetry | null>;
  active: boolean;
  accent: string;
  tint: string;
  portrait: string;
  background?: string;
}

export function AgentStage({ telemetry, active, accent, tint, portrait, background }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const live = useRef({ active, accent, tint });
  live.current = { active, accent, tint };

  useEffect(() => {
    const canvas = canvasRef.current!;
    const gl = canvas.getContext("webgl2", { premultipliedAlpha: true, alpha: true, antialias: false });
    if (!gl) return;
    let disposed = false;
    let raf = 0;
    const cleanup: (() => void)[] = [];

    (async () => {
      const [img, bg] = await Promise.all([loadImage(portrait), background ? loadImage(background) : Promise.resolve(null)]);
      if (disposed) return;
      const { prog, dispose } = fullscreenProgram(gl, FRAG);
      const locs = new Map<string, WebGLUniformLocation | null>();
      const u = (n: string) => {
        if (!locs.has(n)) locs.set(n, gl.getUniformLocation(prog, n));
        return locs.get(n)!;
      };

      const agentTex = texture(gl, img, 0);
      gl.uniform1i(u("uAgent"), 0);
      const bgTex = bg ? texture(gl, bg, 1) : null;
      gl.uniform1i(u("uBg"), 1);
      gl.uniform1f(u("uHasBg"), bg ? 1 : 0);
      gl.uniform3f(u("uIdle"), 79 / 255, 216 / 255, 1);
      const box = alphaBox(img);
      gl.uniform4f(u("uAgentUV"), ...box);
      const contentAspect = (box[2] * img.width) / (box[3] * img.height);
      const bgAspect = bg ? bg.width / bg.height : 1;
      cleanup.push(() => {
        gl.deleteTexture(agentTex);
        if (bgTex) gl.deleteTexture(bgTex);
        dispose();
      });
      cleanup.push(trackCanvasSize(gl, canvas));

      let level = 0, progress = 0, flash = 0, pitch = 0.5, voiced = 0, wasActive = false, last = performance.now();
      const t0 = last;
      const frame = (now: number) => {
        const dt = Math.min(0.05, (now - last) / 1000);
        last = now;
        const tel = telemetry.current;
        const { active: on, accent: hex, tint: tintHex } = live.current;
        if (on && !wasActive) flash = 1;
        wasActive = on;
        flash = Math.max(0, flash - dt * 1.1);
        progress = Math.min(1, Math.max(0, progress + (on ? dt : -dt) / 0.9));
        level += ((tel?.inputLevel ?? 0) - level) * 0.18;
        const f0 = on ? tel?.f0Out ?? 0 : tel?.f0In ?? 0;
        if (f0 > 0) pitch += (dialPos(f0) - pitch) * 0.2;
        voiced += ((f0 > 0 ? 1 : 0) - voiced) * 0.15;

        const W = canvas.width, H = canvas.height, aspect = W / H;
        const h = 0.9, y0 = 0.05, w = (h * contentAspect) / aspect;
        const bh = 0.84, bw = (bh * bgAspect) / aspect;
        // a little right of centre so the name block at the top left stays clear of the head
        const cx = 0.5 + Math.min(0.08, Math.max(0, 0.5 - w * 1.6) * 0.25);
        gl.uniform2f(u("uRes"), W, H);
        gl.uniform1f(u("uAspect"), aspect);
        gl.uniform4f(u("uAgentRect"), cx - w / 2, y0, w, h);
        gl.uniform4f(u("uBgRect"), cx - bw / 2, 0.1, bw, bh);
        gl.uniform1f(u("uTime"), (now - t0) / 1000);
        gl.uniform1f(u("uLevel"), Math.min(1, level * 1.3));
        gl.uniform1f(u("uProgress"), progress);
        gl.uniform1f(u("uFlash"), flash);
        gl.uniform1f(u("uPitch"), pitch);
        gl.uniform1f(u("uVoiced"), voiced);
        gl.uniform3f(u("uLive"), ...hexToVec3(hex));
        gl.uniform3f(u("uTint"), ...hexToVec3(tintHex));
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        raf = requestAnimationFrame(frame);
      };
      raf = requestAnimationFrame(frame);
    })().catch((e) => console.error("AgentStage", e));

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      cleanup.forEach((f) => f());
    };
  }, [telemetry, portrait, background]);

  return <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />;
}
