// Drawing kit for the glass labs: a dark "screen" palette, glow strokes, glass panels, 2D plots (axes, contours, heat
// fields) and a small orbit camera for 3D surfaces. Everything works in CSS pixels; the canvas is DPR-scaled once.
export type Point = readonly [number, number];
export type Point3 = readonly [number, number, number];

export const C = {
  bg0: '#060913', bg1: '#0d1530', ink: '#e8eefb', muted: '#8d9ab8', faint: 'rgba(160,178,220,0.14)', glass: 'rgba(255,255,255,0.045)', edge: 'rgba(255,255,255,0.10)',
  cyan: '#4fd6ff', magenta: '#ff5fa8', amber: '#ffb648', lime: '#7af0a0', violet: '#a98bff', red: '#ff6b6b', white: '#ffffff',
} as const;
export type Tone = 'cyan' | 'magenta' | 'amber' | 'lime' | 'violet' | 'red' | 'white';

export const clamp = (value: number, low = 0, high = 1) => Math.min(high, Math.max(low, value));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const ease = (t: number) => { const x = clamp(t); return x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2; };
/** Progress of t inside the window [from, to], eased. */
export const seg = (t: number, from: number, to: number) => ease((t - from) / (to - from));

export function alpha(color: string, a: number) {
  const hex = color.replace('#', '');
  const [r, g, b] = [0, 2, 4].map(i => parseInt(hex.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${clamp(a)})`;
}

const RAMP = [[0, [32, 28, 92]], [0.28, [96, 62, 214]], [0.55, [47, 198, 255]], [0.8, [122, 240, 160]], [1, [255, 209, 102]]] as const;
/** Height / value colour ramp: deep indigo → violet → cyan → lime → amber. */
export function ramp(t: number, a = 1) {
  const x = clamp(t);
  let i = 0;
  while (i < RAMP.length - 2 && x > RAMP[i + 1][0]) i++;
  const [t0, c0] = RAMP[i], [t1, c1] = RAMP[i + 1];
  const k = (x - t0) / (t1 - t0);
  return `rgba(${c0.map((v, j) => Math.round(lerp(v, c1[j], k))).join(',')},${a})`;
}

export interface LineStyle { width?: number; glow?: number; alpha?: number; dash?: number[]; cap?: CanvasLineCap }
export interface TextStyle { size?: number; color?: string; align?: CanvasTextAlign; base?: CanvasTextBaseline; weight?: number; mono?: boolean; alpha?: number; shadow?: boolean }

export class Gfx {
  constructor(readonly ctx: CanvasRenderingContext2D, readonly w: number, readonly h: number) {}

  background() {
    const { ctx, w, h } = this;
    const g = ctx.createRadialGradient(w * 0.5, h * 0.35, 0, w * 0.5, h * 0.5, Math.max(w, h) * 0.8);
    g.addColorStop(0, C.bg1); g.addColorStop(1, C.bg0);
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(160,180,230,0.07)';
    for (let x = 12; x < w; x += 28) for (let y = 12; y < h; y += 28) ctx.fillRect(x, y, 1, 1);
  }

  line(points: readonly Point[], color: string, style: LineStyle = {}) {
    const pts = points.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
    if (pts.length < 2) return;
    const { ctx } = this;
    ctx.save();
    ctx.globalAlpha = style.alpha ?? 1;
    ctx.lineJoin = 'round'; ctx.lineCap = style.cap ?? 'round';
    ctx.setLineDash(style.dash ?? []);
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    if (style.glow) { ctx.shadowColor = color; ctx.shadowBlur = style.glow; }
    ctx.strokeStyle = color; ctx.lineWidth = style.width ?? 2; ctx.stroke();
    ctx.restore();
  }

  circle(x: number, y: number, r: number, fill: string | null, opts: { glow?: number; stroke?: string; width?: number; alpha?: number } = {}) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    const { ctx } = this;
    ctx.save();
    ctx.globalAlpha = opts.alpha ?? 1;
    ctx.beginPath(); ctx.arc(x, y, Math.max(0.1, r), 0, Math.PI * 2);
    if (opts.glow && fill) { ctx.shadowColor = fill; ctx.shadowBlur = opts.glow; }
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    ctx.shadowBlur = 0;
    if (opts.stroke) { ctx.strokeStyle = opts.stroke; ctx.lineWidth = opts.width ?? 1.5; ctx.stroke(); }
    ctx.restore();
  }

  square(x: number, y: number, s: number, fill: string, opts: { glow?: number; stroke?: string; alpha?: number } = {}) {
    const { ctx } = this;
    ctx.save(); ctx.globalAlpha = opts.alpha ?? 1;
    if (opts.glow) { ctx.shadowColor = fill; ctx.shadowBlur = opts.glow; }
    ctx.fillStyle = fill; ctx.beginPath(); ctx.roundRect(x - s, y - s, 2 * s, 2 * s, s * 0.3); ctx.fill();
    ctx.shadowBlur = 0;
    if (opts.stroke) { ctx.strokeStyle = opts.stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
    ctx.restore();
  }

  polygon(points: readonly Point[], fill: string | null, stroke?: string, width = 1.5) {
    if (points.length < 3) return;
    const { ctx } = this;
    ctx.beginPath(); ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.stroke(); }
  }

  arrow(x1: number, y1: number, x2: number, y2: number, color: string, style: LineStyle & { head?: number } = {}) {
    const length = Math.hypot(x2 - x1, y2 - y1);
    if (!Number.isFinite(length) || length < 1) return;
    const head = Math.min(style.head ?? 10, length * 0.5);
    const [ux, uy] = [(x2 - x1) / length, (y2 - y1) / length];
    this.line([[x1, y1], [x2 - ux * head * 0.6, y2 - uy * head * 0.6]], color, style);
    const { ctx } = this;
    ctx.save(); ctx.globalAlpha = style.alpha ?? 1;
    if (style.glow) { ctx.shadowColor = color; ctx.shadowBlur = style.glow; }
    this.polygon([[x2, y2], [x2 - ux * head + uy * head * 0.5, y2 - uy * head - ux * head * 0.5], [x2 - ux * head - uy * head * 0.5, y2 - uy * head + ux * head * 0.5]], color);
    ctx.restore();
  }

  text(value: string, x: number, y: number, style: TextStyle = {}) {
    const { ctx } = this;
    ctx.save();
    ctx.globalAlpha = style.alpha ?? 1;
    ctx.font = `${style.weight ?? 500} ${style.size ?? 13}px ${style.mono ? 'ui-monospace, SFMono-Regular, Menlo, monospace' : 'Inter, ui-sans-serif, system-ui, sans-serif'}`;
    ctx.textAlign = style.align ?? 'left'; ctx.textBaseline = style.base ?? 'middle';
    if (style.shadow !== false) { ctx.shadowColor = 'rgba(0,0,0,0.65)'; ctx.shadowBlur = 6; }
    ctx.fillStyle = style.color ?? C.ink; ctx.fillText(value, x, y);
    ctx.restore();
  }

  measure(value: string, size = 13, weight = 500) {
    this.ctx.font = `${weight} ${size}px Inter, ui-sans-serif, system-ui, sans-serif`;
    return this.ctx.measureText(value).width;
  }

  /** Frosted glass card with an optional title, the frame for every sub-plot. */
  panel(x: number, y: number, w: number, h: number, title?: string, accent: string = C.cyan) {
    const { ctx } = this;
    ctx.save();
    const g = ctx.createLinearGradient(x, y, x, y + h);
    g.addColorStop(0, 'rgba(255,255,255,0.06)'); g.addColorStop(1, 'rgba(255,255,255,0.02)');
    ctx.fillStyle = g; ctx.strokeStyle = C.edge; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(x, y, w, h, 14); ctx.fill(); ctx.stroke();
    ctx.restore();
    if (title) {
      this.circle(x + 16, y + 17, 3.5, accent, { glow: 10 });
      this.text(title, x + 27, y + 17, { size: 12.5, weight: 600, color: C.ink });
    }
  }

  /** A rounded value tag drawn on the canvas (for in-scene annotations). */
  tag(value: string, x: number, y: number, color: string = C.cyan, align: 'left' | 'center' | 'right' = 'center') {
    const width = this.measure(value, 12, 600) + 16;
    const left = align === 'left' ? x : align === 'right' ? x - width : x - width / 2;
    const { ctx } = this;
    ctx.save();
    ctx.fillStyle = 'rgba(8,12,26,0.82)'; ctx.strokeStyle = alpha(color, 0.7); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(left, y - 11, width, 22, 11); ctx.fill(); ctx.stroke();
    ctx.restore();
    this.text(value, left + width / 2, y, { size: 12, weight: 600, color, align: 'center', shadow: false });
  }

  clip(x: number, y: number, w: number, h: number, draw: () => void) {
    const { ctx } = this;
    ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); draw(); ctx.restore();
  }
}

export interface Rect { x: number; y: number; w: number; h: number }
export const inset = (r: Rect, l: number, t = l, rr = l, b = t): Rect => ({ x: r.x + l, y: r.y + t, w: r.w - l - rr, h: r.h - t - b });

/** Split a rectangle into columns (wide screens) or rows (narrow screens) with a gap. */
export function split(r: Rect, weights: number[], vertical: boolean, gap = 14): Rect[] {
  const total = weights.reduce((a, b) => a + b, 0), span = (vertical ? r.h : r.w) - gap * (weights.length - 1);
  let offset = 0;
  return weights.map(weight => {
    const size = span * weight / total;
    const out = vertical ? { x: r.x, y: r.y + offset, w: r.w, h: size } : { x: r.x + offset, y: r.y, w: size, h: r.h };
    offset += size + gap;
    return out;
  });
}

export interface AxisSpec { xTicks?: number[]; yTicks?: number[]; xLabel?: string; yLabel?: string; fx?: (v: number) => string; fy?: (v: number) => string }

/** A 2D data window inside a rect: mapping, axes, curves, scatter, contours and heat fields. */
export class Plot {
  readonly box: Rect;
  constructor(readonly g: Gfx, rect: Rect, readonly xr: Point, readonly yr: Point, pad = { l: 46, r: 14, t: 34, b: 34 }) {
    this.box = { x: rect.x + pad.l, y: rect.y + pad.t, w: Math.max(10, rect.w - pad.l - pad.r), h: Math.max(10, rect.h - pad.t - pad.b) };
  }
  px = (x: number) => this.box.x + (x - this.xr[0]) / (this.xr[1] - this.xr[0]) * this.box.w;
  py = (y: number) => this.box.y + this.box.h - (y - this.yr[0]) / (this.yr[1] - this.yr[0]) * this.box.h;
  p = ([x, y]: Point): Point => [this.px(x), this.py(y)];
  inv = (px: number, py: number): Point => [this.xr[0] + (px - this.box.x) / this.box.w * (this.xr[1] - this.xr[0]), this.yr[0] + (this.box.y + this.box.h - py) / this.box.h * (this.yr[1] - this.yr[0])];
  inside = (px: number, py: number) => px >= this.box.x && px <= this.box.x + this.box.w && py >= this.box.y && py <= this.box.y + this.box.h;

  axes(spec: AxisSpec = {}) {
    const { g, box } = this;
    const fx = spec.fx ?? vn, fy = spec.fy ?? vn;
    for (const t of spec.xTicks ?? []) {
      g.line([[this.px(t), box.y], [this.px(t), box.y + box.h]], C.faint, { width: 1 });
      g.text(fx(t), this.px(t), box.y + box.h + 13, { size: 10.5, color: C.muted, align: 'center', shadow: false });
    }
    for (const t of spec.yTicks ?? []) {
      g.line([[box.x, this.py(t)], [box.x + box.w, this.py(t)]], C.faint, { width: 1 });
      g.text(fy(t), box.x - 7, this.py(t), { size: 10.5, color: C.muted, align: 'right', shadow: false });
    }
    if (spec.xLabel) g.text(spec.xLabel, box.x + box.w, box.y + box.h + 27, { size: 11, color: C.muted, align: 'right', shadow: false });
    if (spec.yLabel) g.text(spec.yLabel, box.x + 6, box.y + 11, { size: 11, color: C.muted, align: 'left' });
  }

  clip(draw: () => void) { this.g.clip(this.box.x, this.box.y, this.box.w, this.box.h, draw); }
  line(points: readonly Point[], color: string, style?: LineStyle) { this.g.line(points.map(this.p), color, style); }
  curve(fn: (x: number) => number, color: string, style?: LineStyle, samples = 160, from = this.xr[0], to = this.xr[1]) {
    this.line(Array.from({ length: samples + 1 }, (_, i) => { const x = from + (to - from) * i / samples; return [x, fn(x)] as Point; }), color, style);
  }

  /** Scalar field as a smooth image: value → colour per cell, upscaled with smoothing. */
  heat(fn: (x: number, y: number) => number, color: (v: number) => [number, number, number, number], res = 90) {
    const cols = res, rows = Math.max(8, Math.round(res * this.box.h / this.box.w));
    const image = new ImageData(cols, rows);
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      const [x, y] = this.inv(this.box.x + (i + 0.5) / cols * this.box.w, this.box.y + (j + 0.5) / rows * this.box.h);
      const [r, g, b, a] = color(fn(x, y));
      const k = (j * cols + i) * 4;
      image.data[k] = r; image.data[k + 1] = g; image.data[k + 2] = b; image.data[k + 3] = a;
    }
    const off = heatCanvas(cols, rows);
    off.getContext('2d')!.putImageData(image, 0, 0);
    const { ctx } = this.g;
    ctx.save(); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, this.box.x, this.box.y, this.box.w, this.box.h); ctx.restore();
  }

  /** Iso-lines of fn at the given levels (marching squares). */
  contours(fn: (x: number, y: number) => number, levels: number[], color: (level: number, index: number) => string, style: LineStyle = {}, res = 70) {
    const grid = sample(fn, this.xr, this.yr, res, Math.max(10, Math.round(res * this.box.h / this.box.w)));
    levels.forEach((level, index) => {
      const segments = march(grid, level);
      const { ctx } = this.g;
      ctx.save();
      ctx.strokeStyle = color(level, index); ctx.lineWidth = style.width ?? 1.2; ctx.globalAlpha = style.alpha ?? 1;
      if (style.glow) { ctx.shadowColor = color(level, index); ctx.shadowBlur = style.glow; }
      ctx.beginPath();
      for (const [a, b] of segments) { ctx.moveTo(this.px(a[0]), this.py(a[1])); ctx.lineTo(this.px(b[0]), this.py(b[1])); }
      ctx.stroke(); ctx.restore();
    });
  }
}

let offscreen: HTMLCanvasElement | null = null;
function heatCanvas(w: number, h: number) {
  offscreen ??= document.createElement('canvas');
  if (offscreen.width !== w || offscreen.height !== h) { offscreen.width = w; offscreen.height = h; }
  return offscreen;
}

export interface Grid { xr: Point; yr: Point; cols: number; rows: number; v: Float64Array }
export function sample(fn: (x: number, y: number) => number, xr: Point, yr: Point, cols: number, rows: number): Grid {
  const v = new Float64Array((cols + 1) * (rows + 1));
  for (let j = 0; j <= rows; j++) for (let i = 0; i <= cols; i++) v[j * (cols + 1) + i] = fn(xr[0] + (xr[1] - xr[0]) * i / cols, yr[0] + (yr[1] - yr[0]) * j / rows);
  return { xr, yr, cols, rows, v };
}
/** Marching squares: line segments of the level set {fn = level} in data coordinates. */
export function march(grid: Grid, level: number): [Point, Point][] {
  const { xr, yr, cols, rows, v } = grid, out: [Point, Point][] = [];
  const at = (i: number, j: number) => v[j * (cols + 1) + i];
  const X = (i: number) => xr[0] + (xr[1] - xr[0]) * i / cols, Y = (j: number) => yr[0] + (yr[1] - yr[0]) * j / rows;
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const corners: [number, number, number][] = [[i, j, at(i, j)], [i + 1, j, at(i + 1, j)], [i + 1, j + 1, at(i + 1, j + 1)], [i, j + 1, at(i, j + 1)]];
    const hits: Point[] = [];
    for (let k = 0; k < 4; k++) {
      const [ai, aj, a] = corners[k], [bi, bj, b] = corners[(k + 1) % 4];
      if ((a > level) !== (b > level) && Number.isFinite(a) && Number.isFinite(b)) {
        const t = (level - a) / (b - a);
        hits.push([X(ai + (bi - ai) * t), Y(aj + (bj - aj) * t)]);
      }
    }
    if (hits.length >= 2) out.push([hits[0], hits[1]]);
    if (hits.length === 4) out.push([hits[2], hits[3]]);
  }
  return out;
}

/** Orbit camera over the box [-1, 1]² × [0, 1]: drag to rotate, auto-drift when idle. */
export class Camera {
  yaw = -0.75; pitch = 0.62; zoom = 1;
  constructor(public rect: Rect) {}
  project([x, y, z]: Point3): [number, number, number] {
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const X = x * cy - y * sy, Y = x * sy + y * cy;
    const depth = Y * cp + (z - 0.35) * sp;
    const up = (z - 0.35) * cp - Y * sp;
    const scale = Math.min(this.rect.w, this.rect.h * 1.25) * 0.36 * this.zoom, persp = 1 / (1 + depth * 0.18);
    return [this.rect.x + this.rect.w / 2 + X * scale * persp, this.rect.y + this.rect.h * 0.56 - up * scale * persp, depth];
  }
  drag(dx: number, dy: number) { this.yaw -= dx * 0.008; this.pitch = clamp(this.pitch + dy * 0.006, 0.12, 1.35); }
}

/** Shaded height-field surface z = fn(u, v) on [-1, 1]², drawn back to front with a ramp colour and soft lighting. */
export function surface(g: Gfx, cam: Camera, fn: (u: number, v: number) => number, n = 38, opacity = 0.92) {
  const z: number[][] = Array.from({ length: n + 1 }, (_, i) => Array.from({ length: n + 1 }, (_, j) => fn(-1 + 2 * i / n, -1 + 2 * j / n)));
  const P = z.map((row, i) => row.map((h, j) => cam.project([-1 + 2 * i / n, -1 + 2 * j / n, h])));
  const quads: { d: number; i: number; j: number }[] = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) quads.push({ d: (P[i][j][2] + P[i + 1][j][2] + P[i][j + 1][2] + P[i + 1][j + 1][2]) / 4, i, j });
  quads.sort((a, b) => b.d - a.d);
  const { ctx } = g;
  ctx.save(); ctx.lineWidth = 0.6;
  for (const { i, j } of quads) {
    const h = (z[i][j] + z[i + 1][j] + z[i][j + 1] + z[i + 1][j + 1]) / 4;
    const du = (z[i + 1][j] - z[i][j]) * n / 2, dv = (z[i][j + 1] - z[i][j]) * n / 2;
    const light = clamp(0.62 + 0.38 * (-du * 0.5 + dv * 0.3 + 1) / Math.hypot(du, dv, 1), 0.35, 1);
    const pts = [P[i][j], P[i + 1][j], P[i + 1][j + 1], P[i][j + 1]];
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (let k = 1; k < 4; k++) ctx.lineTo(pts[k][0], pts[k][1]);
    ctx.closePath();
    ctx.fillStyle = shade(h, light, opacity); ctx.fill();
    ctx.strokeStyle = `rgba(200,225,255,${0.06 + 0.12 * light})`; ctx.stroke();
  }
  ctx.restore();
}
function shade(h: number, light: number, a: number) {
  const m = ramp(h, 1).match(/\d+/g)!.slice(0, 3).map(Number);
  return `rgba(${m.map(v => Math.round(v * light)).join(',')},${a})`;
}

/** Floor plane frame of the 3D box with optional axis names. */
export function floor(g: Gfx, cam: Camera, labels?: [string, string]) {
  const corners: Point3[] = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]];
  g.polygon(corners.map(c => cam.project(c).slice(0, 2) as unknown as Point), 'rgba(120,150,255,0.05)', 'rgba(150,175,255,0.22)', 1);
  if (labels) {
    const a = cam.project([0, -1.18, 0]), b = cam.project([1.18, 0, 0]);
    g.text(labels[0], a[0], a[1], { size: 11.5, color: C.muted, align: 'center' });
    g.text(labels[1], b[0], b[1], { size: 11.5, color: C.muted, align: 'center' });
  }
}

/** Vietnamese number format: comma decimals, dot thousands, true minus sign. */
export function vn(value: number, digits = 2) {
  if (!Number.isFinite(value)) return value > 0 ? '∞' : value < 0 ? '−∞' : '—';
  const fixed = Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value;
  return fixed.toLocaleString('vi-VN', { minimumFractionDigits: digits, maximumFractionDigits: digits }).replace('-', '−');
}
export const vnAuto = (value: number) => vn(value, Math.abs(value) >= 100 ? 0 : Math.abs(value) >= 10 ? 1 : 2);
export const pct = (value: number, digits = 0) => `${vn(value * 100, digits)}%`;

/** Ranges with the same data units per pixel on both axes that still contain both requested ranges. */
export function fitEqual(rect: Rect, pad: { l: number; r: number; t: number; b: number }, xr: Point, yr: Point): [Point, Point] {
  const w = rect.w - pad.l - pad.r, h = rect.h - pad.t - pad.b;
  const unit = Math.max((xr[1] - xr[0]) / w, (yr[1] - yr[0]) / h);
  const cx = (xr[0] + xr[1]) / 2, cy = (yr[0] + yr[1]) / 2;
  return [[cx - unit * w / 2, cx + unit * w / 2], [cy - unit * h / 2, cy + unit * h / 2]];
}
