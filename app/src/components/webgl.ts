/** Shared WebGL2 plumbing for the full-screen shader stages (VoiceCore, AgentStage). */

const VERT = `#version 300 es
in vec2 p;
void main() { gl_Position = vec4(p, 0.0, 1.0); }`;

/** GLSL value noise: hash / noise / fbm with `octaves` layers. Goes after the uniforms. */
export const glslNoise = (octaves: number) => `
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < ${octaves}; i++) { v += a * noise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; }
  return v;
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string) {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || "shader compile failed");
  return s;
}

/** Links `frag` behind the pass-through vertex shader over one full-screen triangle and makes it current. */
export function fullscreenProgram(gl: WebGL2RenderingContext, frag: string) {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, frag));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "p");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  return {
    prog,
    /** frees the GPU objects but keeps the context: StrictMode remounts on the same canvas */
    dispose() {
      gl.deleteBuffer(buf);
      gl.deleteProgram(prog);
    },
  };
}

/** Keeps the drawing buffer at the canvas's CSS size (DPR capped at 1.5) and the viewport with it. */
export function trackCanvasSize(gl: WebGL2RenderingContext, canvas: HTMLCanvasElement) {
  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    gl.viewport(0, 0, canvas.width, canvas.height);
  };
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  resize();
  return () => ro.disconnect();
}

export function hexToVec3(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/** log-scale position of f0 on the pitch dial, 70..240 Hz */
export const dialPos = (f0: number) => Math.log(f0 / 70) / Math.log(240 / 70);
