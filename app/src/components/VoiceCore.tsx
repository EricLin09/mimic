import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import type { Telemetry } from "../engine/types";
import { dialPos, fullscreenProgram, glslNoise, hexToVec3, trackCanvasSize } from "./webgl";

// Audio-reactive "voice core": a plasma orb whose rim is pushed by the voice,
// instrument rings, a pitch dial along the bottom arc and orbiting sparks when live.
const FRAG = `#version 300 es
precision highp float;
uniform vec2 uRes;
uniform float uTime;
uniform float uLevel;
uniform float uActive;
uniform float uPitch;
uniform float uVoiced;
uniform vec3 uIdle;
uniform vec3 uLive;
out vec4 outColor;
${glslNoise(5)}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / min(uRes.x, uRes.y);
  float r = length(uv);
  float a = atan(uv.y, uv.x);
  float t = uTime;
  vec3 col = mix(uIdle, uLive, uActive);

  // living boundary
  float wob = fbm(vec2(a * 1.6 + t * 0.35, t * 0.22)) - 0.5;
  float wob2 = fbm(vec2(a * 4.0 - t * 0.9, t * 0.5 + 3.0)) - 0.5;
  float energy = 0.035 + uLevel * 0.17;
  float R = 0.235 + uLevel * 0.05 + wob * energy + wob2 * energy * 0.35;
  float d = r - R;

  // domain-warped plasma inside
  vec2 q = uv * 3.2;
  float w = fbm(q + vec2(t * 0.18, -t * 0.12));
  float plasma = fbm(q * 1.4 + 2.2 * vec2(w, -w) + t * 0.25);
  float inside = smoothstep(0.012, -0.02, d);
  vec3 c = col * inside * (0.1 + 0.9 * plasma * plasma) * (0.45 + 0.7 * uLevel);
  // filaments: thin bright veins through the plasma
  float vein = 1.0 - abs(fbm(q * 2.3 + 1.7 * vec2(-w, w) - t * 0.3) * 2.0 - 1.0);
  c += col * inside * pow(vein, 9.0) * (0.35 + 0.8 * uLevel);
  c += mix(col, vec3(1.0), 0.55) * inside * exp(-r * 13.0) * (0.2 + 0.8 * uLevel);

  // rim and halo
  c += col * exp(-abs(d) * 42.0) * (0.75 + 1.6 * uLevel);
  c += col * 0.022 / (abs(d) + 0.035) * (0.35 + 0.9 * uLevel);

  // counter-rotating instrument rings
  for (int k = 0; k < 3; k++) {
    float fk = float(k);
    float rr = 0.345 + fk * 0.052;
    float ring = exp(-abs(r - rr) * 300.0);
    float dir = mod(fk, 2.0) * 2.0 - 1.0;
    float ticks = step(0.55, fract((a / 6.2831853) * (36.0 + fk * 24.0) + t * (0.03 + 0.02 * fk) * dir));
    float arc = smoothstep(0.0, 0.02, sin(a * 2.0 + t * 0.4 * dir + fk) * 0.5 + 0.48);
    c += col * ring * (0.18 + 0.55 * ticks) * mix(0.35, 1.0, uActive) * mix(1.0, arc, 0.5);
  }

  // pitch dial: theta = 0 at the bottom, scale spans +-2.35 rad
  float theta = atan(uv.x, -uv.y);
  float onScale = step(abs(theta), 2.35);
  float dialR = 0.455;
  float minor = step(0.82, fract(theta / 2.35 * 12.0 + 0.5));
  float major = step(0.9, fract(theta / 2.35 * 3.0 + 0.5));
  float tickBand = exp(-abs(r - dialR) * 160.0) * onScale;
  c += col * tickBand * (0.08 + 0.35 * minor + 0.6 * major) * mix(0.4, 1.0, uActive);
  float tn = mix(-2.35, 2.35, clamp(uPitch, 0.0, 1.0));
  vec2 tip = vec2(sin(tn), -cos(tn)) * dialR;
  float needle = exp(-length(uv - tip) * 110.0) * uVoiced;
  c += (col * 0.6 + vec3(0.6)) * needle * (0.6 + 1.2 * uActive);
  // needle trail along the dial
  float trail = exp(-abs(r - dialR) * 260.0) * smoothstep(0.35, 0.0, abs(theta - tn)) * onScale * uVoiced;
  c += col * trail * 0.9;

  // orbiting sparks when live
  for (int i = 0; i < 16; i++) {
    float fi = float(i);
    float ang = t * (0.35 + 0.045 * fi) * (mod(fi, 2.0) * 2.0 - 1.0) + fi * 2.399;
    float rad = 0.29 + 0.17 * fract(fi * 0.371 + t * 0.06);
    vec2 sp = vec2(cos(ang), sin(ang)) * rad;
    c += col * exp(-length(uv - sp) * (180.0 - 60.0 * uLevel)) * uActive * (0.35 + uLevel);
  }

  c *= smoothstep(0.8, 0.12, r);
  // filmic roll-off instead of hard clipping, so a loud voice glows rather than blows out
  c = vec3(1.0) - exp(-c * 1.35);
  outColor = vec4(c, max(c.r, max(c.g, c.b)));
}`;

interface Props {
  telemetry: RefObject<Telemetry | null>;
  active: boolean;
  accent: string;
}

export function VoiceCore({ telemetry, active, accent }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const live = useRef({ active, accent });
  live.current = { active, accent };

  useEffect(() => {
    const canvas = canvasRef.current!;
    const gl = canvas.getContext("webgl2", { premultipliedAlpha: true, alpha: true, antialias: false });
    if (!gl) return;
    const { prog, dispose } = fullscreenProgram(gl, FRAG);
    const u = (n: string) => gl.getUniformLocation(prog, n);
    const U = {
      res: u("uRes"), time: u("uTime"), level: u("uLevel"), active: u("uActive"),
      pitch: u("uPitch"), voiced: u("uVoiced"), idle: u("uIdle"), live: u("uLive"),
    };
    gl.uniform3f(U.idle, 79 / 255, 216 / 255, 255 / 255);

    const untrack = trackCanvasSize(gl, canvas);

    let level = 0, act = 0, pitch = 0.5, voiced = 0, raf = 0;
    const t0 = performance.now();
    const frame = () => {
      const tel = telemetry.current;
      const { active: on, accent: hex } = live.current;
      level += ((tel?.inputLevel ?? 0) - level) * 0.18;
      act += ((on ? 1 : 0) - act) * 0.06;
      const f0 = on ? tel?.f0Out ?? 0 : tel?.f0In ?? 0;
      if (f0 > 0) pitch += (dialPos(f0) - pitch) * 0.2;
      voiced += ((f0 > 0 ? 1 : 0) - voiced) * 0.15;
      gl.uniform2f(U.res, canvas.width, canvas.height);
      gl.uniform1f(U.time, (performance.now() - t0) / 1000);
      gl.uniform1f(U.level, Math.min(1, level * 1.3));
      gl.uniform1f(U.active, act);
      gl.uniform1f(U.pitch, pitch);
      gl.uniform1f(U.voiced, voiced);
      gl.uniform3f(U.live, ...hexToVec3(hex));
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      untrack();
      dispose();
    };
  }, [telemetry]);

  return <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />;
}
