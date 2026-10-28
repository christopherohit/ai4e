// Small WebGL2 renderer for the atlas: every file is one vertex/index buffer drawn in a few calls, while per-part state
// (explode offset, display mode, colour) lives in two textures indexed by a per-vertex part id. Picking renders part ids
// into an offscreen target. No dependencies.
import type { Manifest, Part, Vec3 } from './atlas';
import { FEMALE_DEPTH, FEMALE_SCALE, FEMALE_WIDTH, STATURE } from './atlas';

export const MODE = { hidden: 0, solid: 1, ghost: 2, focus: 3 } as const;

const TEX_W = 64;

const keys = FEMALE_WIDTH.map(([y, s]) => `vec2(${y.toFixed(3)}, ${s.toFixed(3)})`).join(', ');
const VERTEX = `#version 300 es
precision highp float;
precision highp int;
in vec3 a_pos;
in vec2 a_oct;
in uint a_part;
uniform mat4 u_viewProj;
uniform vec3 u_qMin;
uniform vec3 u_qSize;
uniform float u_female;
uniform highp sampler2D u_state;
uniform lowp sampler2D u_color;
out vec3 v_normal;
out vec3 v_world;
flat out vec4 v_color;
flat out float v_mode;
flat out uint v_part;
const vec2 K[${FEMALE_WIDTH.length}] = vec2[](${keys});
float widthAt(float y) {
  if (y <= K[0].x) return K[0].y;
  for (int i = 1; i < ${FEMALE_WIDTH.length}; i++) {
    if (y <= K[i].x) { float t = (y - K[i - 1].x) / (K[i].x - K[i - 1].x); return mix(K[i - 1].y, K[i].y, t * t * (3.0 - 2.0 * t)); }
  }
  return K[${FEMALE_WIDTH.length - 1}].y;
}
vec3 decodeOct(vec2 e) {
  vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
  if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
  return normalize(n);
}
void main() {
  ivec2 cell = ivec2(int(a_part) % ${TEX_W}, int(a_part) / ${TEX_W});
  vec4 state = texelFetch(u_state, cell, 0);
  v_mode = state.w;
  v_color = texelFetch(u_color, cell, 0);
  v_part = a_part;
  vec3 p = u_qMin + (a_pos * 32767.0 + 32768.0) / 65535.0 * u_qSize;
  vec3 f = vec3(p.x * widthAt(p.y / ${STATURE.toFixed(3)}) * ${FEMALE_SCALE.toFixed(3)}, p.y * ${FEMALE_SCALE.toFixed(3)}, p.z * ${(FEMALE_DEPTH * FEMALE_SCALE).toFixed(4)});
  p = mix(p, f, u_female) + state.xyz;
  v_world = p;
  v_normal = decodeOct(a_oct);
  gl_Position = state.w < 0.5 ? vec4(2.0, 2.0, 2.0, 1.0) : u_viewProj * vec4(p, 1.0);
}`;

const FRAGMENT = `#version 300 es
precision highp float;
precision highp int;
in vec3 v_normal;
in vec3 v_world;
flat in vec4 v_color;
flat in float v_mode;
flat in uint v_part;
uniform vec3 u_eye;
uniform int u_pass;
uniform float u_pulse;
uniform uint u_hover;
out vec4 outColor;
void main() {
  bool ghost = v_mode > 1.5 && v_mode < 2.5;
  if (u_pass == 2) {
    if (ghost) discard;
    uint id = v_part + 1u;
    outColor = vec4(float(id & 255u), float((id >> 8) & 255u), float((id >> 16) & 255u), 255.0) / 255.0;
    return;
  }
  if ((u_pass == 0) == ghost) discard;
  vec3 n = normalize(v_normal);
  vec3 v = normalize(u_eye - v_world);
  if (dot(n, v) < 0.0) n = -n;
  float rim = pow(1.0 - max(dot(n, v), 0.0), 2.2);
  if (ghost) {
    float a = 0.02 + 0.22 * rim;
    outColor = vec4(vec3(0.45, 0.68, 1.0) * (0.5 + rim), a);
    return;
  }
  vec3 key = normalize(vec3(-0.45, 0.8, 0.55));
  vec3 fill = normalize(vec3(0.6, 0.2, -0.4));
  float hemi = 0.5 + 0.5 * n.y;
  float diff = max(dot(n, key), 0.0);
  float spec = pow(max(dot(n, normalize(key + v)), 0.0), 36.0) * 0.28;
  vec3 base = v_color.rgb;
  vec3 c = base * (0.30 + 0.22 * hemi + 0.62 * diff + 0.16 * max(dot(n, fill), 0.0)) + spec;
  c += vec3(0.35, 0.6, 1.0) * rim * 0.22;
  if (v_mode > 2.5) c += base * rim * 0.55 + vec3(1.0, 0.85, 0.6) * u_pulse * 0.10;
  if (u_hover == v_part + 1u) c = mix(c, vec3(1.0, 0.92, 0.7), 0.28);
  outColor = vec4(pow(c, vec3(0.92)), 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) ?? 'shader');
  return shader;
}

// ---- matrices (column-major) ----------------------------------------------------------------------------------------
export function perspective(fovy: number, aspect: number, near: number, far: number) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
export function lookAt(eye: Vec3, target: Vec3, up: Vec3 = [0, 1, 0]) {
  const z = norm(sub(eye, target)), x = norm(cross(up, z)), y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
}
export function multiply(a: Float32Array, b: Float32Array) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    out[c * 4 + r] = s;
  }
  return out;
}
export function project(m: Float32Array, p: Vec3, w: number, h: number) {
  const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
  const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
  const cw = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
  return { x: (x / cw * 0.5 + 0.5) * w, y: (1 - (y / cw * 0.5 + 0.5)) * h, behind: cw <= 0 };
}
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: Vec3): Vec3 => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// ---- camera --------------------------------------------------------------------------------------------------------
export class OrbitCamera {
  yaw = 0.5; pitch = 0.08; distance = 3.4; target: Vec3 = [0, 0.9, 0];
  private goal: { yaw: number; pitch: number; distance: number; target: Vec3 } | null = null;
  fov = 30 * Math.PI / 180;
  eye(): Vec3 {
    const cp = Math.cos(this.pitch);
    return [this.target[0] + this.distance * cp * Math.sin(this.yaw), this.target[1] + this.distance * Math.sin(this.pitch), this.target[2] + this.distance * cp * Math.cos(this.yaw)];
  }
  matrix(aspect: number) {
    const near = Math.max(0.01, this.distance * 0.02);
    return multiply(perspective(this.fov, aspect, near, this.distance * 4 + 4), lookAt(this.eye(), this.target));
  }
  /** Distance that fits a sphere of `radius` in the view for this aspect. */
  fitDistance(radius: number, aspect: number) {
    const half = Math.min(this.fov / 2, Math.atan(Math.tan(this.fov / 2) * aspect));
    return radius / Math.sin(half) * 1.04;
  }
  animateTo(goal: Partial<{ yaw: number; pitch: number; distance: number; target: Vec3 }>) {
    let yaw = goal.yaw ?? this.yaw;
    while (yaw - this.yaw > Math.PI) yaw -= Math.PI * 2;
    while (yaw - this.yaw < -Math.PI) yaw += Math.PI * 2;
    this.goal = { yaw, pitch: goal.pitch ?? this.pitch, distance: goal.distance ?? this.distance, target: goal.target ?? [...this.target] };
  }
  stop() { this.goal = null; }
  /** Advance an animation; returns true while moving. */
  step(dt: number) {
    if (!this.goal) return false;
    const k = 1 - Math.exp(-dt * 7);
    const g = this.goal;
    this.yaw += (g.yaw - this.yaw) * k;
    this.pitch += (g.pitch - this.pitch) * k;
    this.distance += (g.distance - this.distance) * k;
    for (let a = 0; a < 3; a++) this.target[a] += (g.target[a] - this.target[a]) * k;
    const done = Math.abs(g.yaw - this.yaw) < 1e-3 && Math.abs(g.pitch - this.pitch) < 1e-3 && Math.abs(g.distance - this.distance) < 1e-3 * g.distance && Math.hypot(...sub(g.target, this.target)) < 1e-4;
    if (done) this.goal = null;
    return true;
  }
}

// ---- renderer ------------------------------------------------------------------------------------------------------
interface Batch { vao: WebGLVertexArrayObject; count: number }

export class AtlasRenderer {
  readonly gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private uniforms: Record<string, WebGLUniformLocation | null> = {};
  private batches: Batch[] = [];
  private stateData: Float32Array;
  private colorData: Uint8Array;
  private stateTex: WebGLTexture;
  private colorTex: WebGLTexture;
  private pickTarget?: { fb: WebGLFramebuffer; color: WebGLRenderbuffer; depth: WebGLRenderbuffer; w: number; h: number };
  private thumbTarget?: { fb: WebGLFramebuffer; color: WebGLRenderbuffer; depth: WebGLRenderbuffer; w: number; h: number };

  constructor(readonly canvas: HTMLCanvasElement, private manifest: Manifest, partCount: number) {
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: true, premultipliedAlpha: true, powerPreference: 'high-performance' });
    if (!gl) throw new Error('WebGL2 không khả dụng trên trình duyệt này.');
    this.gl = gl;
    const program = gl.createProgram()!;
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
    gl.bindAttribLocation(program, 0, 'a_pos');
    gl.bindAttribLocation(program, 1, 'a_oct');
    gl.bindAttribLocation(program, 2, 'a_part');
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? 'link');
    this.program = program;
    for (const name of ['u_viewProj', 'u_qMin', 'u_qSize', 'u_female', 'u_state', 'u_color', 'u_eye', 'u_pass', 'u_pulse', 'u_hover']) this.uniforms[name] = gl.getUniformLocation(program, name);
    const rows = Math.ceil(partCount / TEX_W);
    this.stateData = new Float32Array(TEX_W * rows * 4);
    this.colorData = new Uint8Array(TEX_W * rows * 4);
    this.stateTex = this.texture(gl.RGBA32F, gl.RGBA, gl.FLOAT, this.stateData, rows);
    this.colorTex = this.texture(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this.colorData, rows);
  }

  private texture(internal: number, format: number, type: number, data: ArrayBufferView, rows: number) {
    const gl = this.gl, tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, TEX_W, rows, 0, format, type, data);
    return tex;
  }

  /** Upload one manifest file: interleave a part id per vertex and rebase the 16-bit indices into one 32-bit buffer. */
  addFile(buffer: ArrayBuffer, parts: Part[]) {
    const gl = this.gl;
    const vertexTotal = parts.reduce((s, p) => s + p.vertexCount, 0);
    const indexTotal = parts.reduce((s, p) => s + p.indexCount, 0);
    const vertices = new Uint8Array(vertexTotal * 8), ids = new Uint16Array(vertexTotal), indices = new Uint32Array(indexTotal);
    const bytes = new Uint8Array(buffer);
    let v = 0, i = 0;
    for (const p of parts) {
      vertices.set(bytes.subarray(p.vertexOffset, p.vertexOffset + p.vertexCount * 8), v * 8);
      ids.fill(p.index, v, v + p.vertexCount);
      const local = new Uint16Array(buffer, p.indexOffset, p.indexCount);
      for (let k = 0; k < local.length; k++) indices[i + k] = local[k] + v;
      v += p.vertexCount;
      i += p.indexCount;
    }
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.SHORT, true, 8, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.BYTE, true, 8, 6);
    const idb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, idb);
    gl.bufferData(gl.ARRAY_BUFFER, ids, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribIPointer(2, 1, gl.UNSIGNED_SHORT, 0, 0);
    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    this.batches.push({ vao, count: indexTotal });
  }

  setPart(index: number, offset: Vec3, mode: number, color: [number, number, number]) {
    const s = index * 4;
    this.stateData[s] = offset[0]; this.stateData[s + 1] = offset[1]; this.stateData[s + 2] = offset[2]; this.stateData[s + 3] = mode;
    this.colorData[s] = color[0]; this.colorData[s + 1] = color[1]; this.colorData[s + 2] = color[2]; this.colorData[s + 3] = 255;
  }
  commit() {
    const gl = this.gl, rows = this.stateData.length / 4 / TEX_W;
    gl.bindTexture(gl.TEXTURE_2D, this.stateTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, rows, gl.RGBA, gl.FLOAT, this.stateData);
    gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, rows, gl.RGBA, gl.UNSIGNED_BYTE, this.colorData);
  }

  private prepare(viewProj: Float32Array, eye: Vec3, female: number, pass: number, pulse: number, hover: number) {
    const gl = this.gl, u = this.uniforms, { min, max } = this.manifest.quantization;
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(u.u_viewProj, false, viewProj);
    gl.uniform3fv(u.u_qMin, min);
    gl.uniform3f(u.u_qSize, max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    gl.uniform1f(u.u_female, female);
    gl.uniform3fv(u.u_eye, eye);
    gl.uniform1i(u.u_pass, pass);
    gl.uniform1f(u.u_pulse, pulse);
    gl.uniform1ui(u.u_hover, hover);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.stateTex);
    gl.uniform1i(u.u_state, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
    gl.uniform1i(u.u_color, 1);
  }

  private drawAll() {
    const gl = this.gl;
    for (const b of this.batches) { gl.bindVertexArray(b.vao); gl.drawElements(gl.TRIANGLES, b.count, gl.UNSIGNED_INT, 0); }
    gl.bindVertexArray(null);
  }

  render(viewProj: Float32Array, eye: Vec3, female: number, opts: { ghosts: boolean; pulse: number; hover: number }, w = this.canvas.width, h = this.canvas.height) {
    const gl = this.gl;
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    gl.depthMask(true);
    this.prepare(viewProj, eye, female, 0, opts.pulse, opts.hover);
    this.drawAll();
    if (opts.ghosts) {
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      this.prepare(viewProj, eye, female, 1, opts.pulse, opts.hover);
      this.drawAll();
      gl.depthMask(true);
      gl.disable(gl.BLEND);
    }
  }

  private target(slot: 'pickTarget' | 'thumbTarget', w: number, h: number) {
    const gl = this.gl;
    let t = this[slot];
    if (!t || t.w !== w || t.h !== h) {
      if (t) { gl.deleteFramebuffer(t.fb); gl.deleteRenderbuffer(t.color); gl.deleteRenderbuffer(t.depth); }
      const fb = gl.createFramebuffer()!, color = gl.createRenderbuffer()!, depth = gl.createRenderbuffer()!;
      gl.bindRenderbuffer(gl.RENDERBUFFER, color);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, w, h);
      gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, color);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
      t = { fb, color, depth, w, h };
      this[slot] = t;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
    return t;
  }

  /** Part index under canvas pixel (x, y) in device pixels, or -1. */
  pick(viewProj: Float32Array, eye: Vec3, female: number, x: number, y: number) {
    const gl = this.gl, w = this.canvas.width, h = this.canvas.height;
    this.target('pickTarget', w, h);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    this.prepare(viewProj, eye, female, 2, 0, 0);
    this.drawAll();
    const px = new Uint8Array(4);
    gl.readPixels(Math.round(x), Math.round(h - 1 - y), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return (px[0] | (px[1] << 8) | (px[2] << 16)) - 1;
  }

  /** Render the current state at w×h offscreen and return it as ImageData (used for the organ cards). */
  snapshot(viewProj: Float32Array, eye: Vec3, female: number, w: number, h: number) {
    const gl = this.gl;
    this.target('thumbTarget', w, h);
    this.render(viewProj, eye, female, { ghosts: false, pulse: 0, hover: 0 }, w, h);
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const image = new ImageData(w, h);
    for (let row = 0; row < h; row++) image.data.set(px.subarray((h - 1 - row) * w * 4, (h - row) * w * 4), row * w * 4);
    return image;
  }
}
