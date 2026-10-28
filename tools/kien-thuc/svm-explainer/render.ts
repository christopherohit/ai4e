// SVM lab on the Iris flowers: sweep candidate lines, find the widest street and its support vectors, soften the
// margin with C on overlapping species, then lift the plane with an RBF kernel to wrap the middle species.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, Camera, Plot, alpha, clamp, floor, lerp, march, sample, seg, split, surface, vn, type Point, type Rect } from '../../_shared/glass-lab/gfx';
import { categoryChart } from '../../_shared/glass-lab/views';
import { C_GRID, GAMMA_GRID, HARD_C, MIDDLE, OVERLAP, SEPARABLE, corridor, crossValidate, decision, errors, marginWidth, roles, supportCount, toStd, train, weights, type Model, type Problem, type Sample } from './model';

export const explainerKind = 'educational-simulation';

const SPECIES_COLOR = [C.cyan, C.amber, C.magenta];
const SPECIES_NAME = ['setosa', 'versicolor', 'virginica'];
const VIEWS: Record<string, { x: Point; y: Point; labels: [string, string] }> = {
  separable: { x: [4, 7.2], y: [1.9, 4.6], labels: ['chiều dài đài hoa (cm)', 'chiều rộng đài hoa (cm)'] },
  overlap: { x: [2.8, 7.2], y: [0.85, 2.7], labels: ['chiều dài cánh hoa (cm)', 'chiều rộng cánh hoa (cm)'] },
  middle: { x: [0.6, 7.4], y: [-0.1, 2.8], labels: ['chiều dài cánh hoa (cm)', 'chiều rộng cánh hoa (cm)'] },
};
const state = { angle: null as number | null, removed: new Set<number>(), cIndex: null as number | null, gammaIndex: 2, camera: new Camera({ x: 0, y: 0, w: 1, h: 1 }), dragging: false };
let currentT = 0;
let pickPlot: Plot | null = null;

/** Iris lengths are rounded to 0.1 cm, so many flowers coincide; a fixed tiny offset keeps every one visible. */
const jitter = (i: number): Point => [((i * 37) % 11 - 5) * 0.007, ((i * 53) % 13 - 6) * 0.006];

function canvasFor(f: Frame, rect: Rect, p: Problem, title: string) {
  f.g.panel(rect.x, rect.y, rect.w, rect.h, title, C.violet);
  const view = VIEWS[p.name];
  const plot = new Plot(f.g, rect, view.x, view.y, { l: 44, r: 14, t: 40, b: 36 });
  return plot;
}

function axes(plot: Plot, p: Problem) {
  const view = VIEWS[p.name];
  const step = (r: Point) => (r[1] - r[0] > 4 ? 1 : 0.5);
  const ticks = (r: Point) => { const s = step(r), out: number[] = []; for (let v = Math.ceil(r[0] / s) * s; v <= r[1]; v += s) out.push(v); return out; };
  plot.axes({ xTicks: ticks(view.x), yTicks: ticks(view.y), xLabel: view.labels[0], yLabel: view.labels[1], fx: v => vn(v, 1), fy: v => vn(v, 1) });
}

/** Decision field: tinted by the predicted class, brighter where |f| is large, with the street f = ±1 and boundary f = 0. */
function field(plot: Plot, p: Problem, f: (x: Point) => number, street = true) {
  const neg = p.name === 'overlap' ? [255, 182, 72] : [79, 214, 255], pos = p.name === 'separable' || p.name === 'middle' ? [255, 182, 72] : [255, 95, 168];
  plot.heat((a, b) => f(toStd(p, [a, b])), v => { const c = v >= 0 ? pos : neg, k = 0.1 + 0.22 * Math.min(1, Math.abs(v) / 2); return [c[0] * k + 6, c[1] * k + 9, c[2] * k + 20, 255]; }, 90);
  const fn = (a: number, b: number) => f(toStd(p, [a, b]));
  if (street) plot.contours(fn, [-1, 1], () => alpha(C.white, 0.55), { width: 1.4 });
  plot.contours(fn, [0], () => C.white, { width: 2.6, glow: 14 });
}

function points(plot: Plot, p: Problem, options: { dim?: Set<number>; rings?: Set<number>; violate?: Set<number>; time?: number } = {}) {
  const { g } = plot;
  p.data.forEach((s, i) => {
    const [jx, jy] = jitter(i), x = plot.px(s.raw[0] + jx), y = plot.py(s.raw[1] + jy), color = SPECIES_COLOR[s.species];
    const a = options.dim?.has(i) ? 0.22 : 1;
    if (s.y === 1) g.square(x, y, 4.2, color, { glow: 8, alpha: a }); else g.circle(x, y, 4.4, color, { glow: 8, alpha: a });
    if (options.rings?.has(i)) g.circle(x, y, 10 + (options.time ? Math.sin(options.time * 4) * 1.5 : 0), null, { stroke: options.violate?.has(i) ? C.red : C.white, width: 2 });
  });
}

function legend(f: Frame, plot: Plot, p: Problem) {
  const species = [...new Set(p.data.map(s => s.species))];
  let x = plot.box.x + 8;
  for (const sp of species) {
    const positive = p.data.find(s => s.species === sp)!.y === 1;
    if (positive) f.g.square(x + 5, plot.box.y + 12, 4, SPECIES_COLOR[sp]); else f.g.circle(x + 5, plot.box.y + 12, 4.2, SPECIES_COLOR[sp]);
    f.g.text(`${SPECIES_NAME[sp]}${positive ? ' (+1)' : ' (−1)'}`, x + 14, plot.box.y + 12, { size: 11.5, color: C.ink });
    x += f.g.measure(`${SPECIES_NAME[sp]} (+1)`, 11.5) + 30;
  }
}

// ---------- Chapter 1: candidate lines ----------
const ANGLES = Array.from({ length: 73 }, (_, i) => -90 + i * 2.5);
const WIDTHS = ANGLES.map(a => corridor(SEPARABLE.data, a * Math.PI / 180).width);
const BEST = ANGLES[WIDTHS.indexOf(Math.max(...WIDTHS))];
const HARD = train(SEPARABLE.data, HARD_C);
const animatedAngle = (t: number) => (t < 7.5 ? lerp(-90, 90, seg(t, 0.8, 7.5)) : lerp(90, BEST, seg(t, 7.5, 9.5)));
const angleNow = () => state.angle ?? animatedAngle(currentT);

const chapter1: Scene = {
  title: 'Vô số đường phân tách',
  duration: 11,
  beats: () => [
    { at: 0, text: '100 bông hoa thật: setosa (tròn) và versicolor (vuông), đo chiều dài và chiều rộng đài hoa.' },
    { at: 1.5, text: 'Xoay một đường thẳng qua mọi hướng. Với mỗi hướng, đặt nó giữa hành lang trống rộng nhất giữa hai loài.' },
    { at: 4.5, text: 'Phần lớn các hướng cắt qua dữ liệu (đỏ). Chỉ một dải hướng hẹp tách được hai loài, và hành lang rộng hẹp khác nhau.' },
    { at: 8, text: `SVM chọn hướng có hành lang rộng nhất (khoảng ${vn(BEST, 0)}°): điểm mới hơi lệch vẫn rơi đúng phía.` },
  ],
  formula: 'lề = khoảng cách từ đường phân tách tới điểm gần nhất của mỗi loài (đơn vị chuẩn hóa)',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.4, 1] : [1.5, 1], narrow);
    const plot = canvasFor(f, left, SEPARABLE, 'Hành lang giữa hai loài');
    axes(plot, SEPARABLE);
    const angle = angleNow(), c = corridor(SEPARABLE.data, angle * Math.PI / 180);
    const proj = (a: number, b: number) => { const z = toStd(SEPARABLE, [a, b]); return z[0] * c.normal[0] + z[1] * c.normal[1] - c.offset; };
    plot.clip(() => {
      if (c.width > 0) {
        plot.heat((a, b) => proj(a, b), v => (Math.abs(v) < c.width / 2 ? [40, 60, 110, 255] : [8, 12, 26, 255]), 90);
        plot.contours(proj, [-c.width / 2, c.width / 2], () => alpha(C.white, 0.55), { width: 1.4 });
      }
      plot.contours(proj, [0], () => (c.width > 0 ? C.lime : C.red), { width: 2.6, glow: 14 });
    });
    points(plot, SEPARABLE);
    legend(f, plot, SEPARABLE);
    g.tag(c.width > 0 ? `hành lang rộng ${vn(c.width, 3)}` : 'cắt qua dữ liệu', plot.box.x + plot.box.w - 8, plot.box.y + plot.box.h - 14, c.width > 0 ? C.lime : C.red, 'right');
    const reveal = state.angle === null ? clamp((angle + 90) / 180) : 1;
    categoryChart(f, right, 'Độ rộng hành lang theo hướng', ANGLES.map(a => (a % 45 === 0 ? `${a}°` : '')), [{ label: 'độ rộng (âm = không tách được)', values: WIDTHS.map(w => Math.max(-0.4, w)), color: C.lime }], [-0.4, 0.5], Math.round((angle + 90) / 2.5), t < 7.5 && state.angle === null ? reveal : 1, v => vn(v, 1));
  },
  metrics() { const c = corridor(SEPARABLE.data, angleNow() * Math.PI / 180); return [{ label: 'Hướng pháp tuyến', value: `${vn(angleNow(), 0)}°`, tone: 'violet' }, { label: 'Độ rộng hành lang', value: c.width > 0 ? vn(c.width, 3) : 'không tách được', tone: c.width > 0 ? 'lime' : 'red' }, { label: 'Rộng nhất', value: vn(Math.max(...WIDTHS), 3), tone: 'amber' }]; },
  table: () => ({ headers: ['Góc', 'Độ rộng hành lang'], rows: ANGLES.filter((_, i) => i % 2 === 0).map(a => [`${vn(a, 1)}°`, WIDTHS[ANGLES.indexOf(a)] > 0 ? vn(WIDTHS[ANGLES.indexOf(a)], 3) : 'không tách được']) }),
};

// ---------- Chapter 2: maximum margin ----------
const hardModel = () => { const kept = SEPARABLE.data.filter((_, i) => !state.removed.has(i)); return kept.some(s => s.y === 1) && kept.some(s => s.y === -1) ? { model: train(kept, HARD_C), kept } : null; };

const chapter2: Scene = {
  title: 'Lề cực đại và support vector',
  duration: 11,
  idle: true,
  beats: () => [
    { at: 0, text: 'SVM giải bài toán: mọi điểm nằm ngoài dải y·f(x) ≥ 1, và dải đó rộng nhất có thể.' },
    { at: 3, text: `Lời giải chạm đúng ${supportCount(HARD)} bông hoa trên mép dải (vòng trắng). Đó là các support vector.` },
    { at: 6, text: `Biểu đồ bên phải: trong ${SEPARABLE.data.length} hệ số αᵢ, chỉ ${supportCount(HARD)} khác 0. Các điểm khác không ảnh hưởng gì tới đường biên.` },
    { at: 8.5, text: 'Nhấp vào một bông hoa để bỏ nó: bỏ điểm thường thì đường biên đứng yên, bỏ support vector thì dải rộng ra.' },
  ],
  formula: () => { const m = hardModel(); const w = m ? weights(m.model) : [0, 0]; return `min ½‖w‖² sao cho yᵢ(w·xᵢ + b) ≥ 1\nw = Σ αᵢ yᵢ xᵢ = (${vn(w[0], 3)}; ${vn(w[1], 3)}),  lề = 2/‖w‖ = ${m ? vn(marginWidth(m.model), 3) : '—'}`; },
  render(f) {
    const { g, t, narrow, time } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.4, 1] : [1.5, 1], narrow);
    const plot = canvasFor(f, left, SEPARABLE, 'Dải lề rộng nhất (đơn vị chuẩn hóa)');
    pickPlot = plot;
    const fit = hardModel();
    if (fit) {
      const grow = seg(t, 0.3, 2.6);
      plot.clip(() => field(plot, SEPARABLE, x => decision(fit.model, x) / Math.max(grow, 0.05), true));
    }
    axes(plot, SEPARABLE);
    const rings = new Set<number>(), keptIndex = SEPARABLE.data.map((_, i) => i).filter(i => !state.removed.has(i));
    if (fit && t > 3) roles(fit.model).forEach((role, k) => { if (role !== 'free') rings.add(keptIndex[k]); });
    const dim = new Set(state.removed);
    if (t > 6 && t < 8.5 && fit) SEPARABLE.data.forEach((_, i) => { if (!rings.has(i)) dim.add(i); });
    points(plot, SEPARABLE, { dim, rings, time });
    legend(f, plot, SEPARABLE);
    // Right: all α_i as bars, sorted by flower index.
    g.panel(right.x, right.y, right.w, right.h, 'Hệ số αᵢ của từng bông hoa', C.amber);
    if (!fit) return;
    const alphaOf = new Map(keptIndex.map((original, k) => [original, fit.model.alpha[k]]));
    const max = Math.max(...fit.model.alpha, 1e-9), box = { x: right.x + 16, y: right.y + 50, w: right.w - 32, h: right.h - 86 };
    const bw = box.w / SEPARABLE.data.length;
    SEPARABLE.data.forEach((s, i) => {
      const a = alphaOf.get(i) ?? 0, h = box.h * a / max * seg(t, 5.5, 7.5);
      if (state.removed.has(i)) { g.text('×', box.x + i * bw + bw / 2, box.y + box.h - 6, { size: 9, color: C.red, align: 'center' }); return; }
      g.ctx.save(); g.ctx.fillStyle = a > 1e-6 ? SPECIES_COLOR[s.species] : alpha(C.white, 0.15); if (a > 1e-6) { g.ctx.shadowColor = SPECIES_COLOR[s.species]; g.ctx.shadowBlur = 12; }
      g.ctx.fillRect(box.x + i * bw + bw * 0.15, box.y + box.h - Math.max(1.5, h), bw * 0.7, Math.max(1.5, h)); g.ctx.restore();
    });
    g.line([[box.x, box.y + box.h], [box.x + box.w, box.y + box.h]], C.faint, { width: 1 });
    g.text(`${supportCount(fit.model)} / ${fit.kept.length} hệ số khác 0`, box.x, box.y + box.h + 20, { size: 12, color: C.ink, weight: 650 });
    g.text('trục ngang: 100 bông hoa', box.x + box.w, box.y + box.h + 20, { size: 11, color: C.muted, align: 'right' });
  },
  metrics() { const fit = hardModel(); return fit ? [{ label: 'Độ rộng lề', value: vn(marginWidth(fit.model), 3), tone: 'lime' }, { label: 'Support vector', value: `${supportCount(fit.model)} / ${fit.kept.length}`, tone: 'amber' }, { label: 'Đã bỏ', value: String(state.removed.size), tone: 'red' }] : []; },
  pointer(e) {
    if (e.type !== 'click' || !pickPlot) return false;
    let best = -1, distance = 12;
    SEPARABLE.data.forEach((s, i) => { const [jx, jy] = jitter(i); const d = Math.hypot(pickPlot!.px(s.raw[0] + jx) - e.x, pickPlot!.py(s.raw[1] + jy) - e.y); if (d < distance) { distance = d; best = i; } });
    if (best < 0) return false;
    if (state.removed.has(best)) state.removed.delete(best); else state.removed.add(best);
    return true;
  },
  table: () => { const fit = hardModel(); if (!fit) return { headers: ['—'], rows: [] }; const r = roles(fit.model); return { headers: ['Bông hoa', 'Loài', 'Đài dài (cm)', 'Đài rộng (cm)', 'αᵢ', 'y·f(x)'], rows: fit.kept.map((s, k) => ({ s, k })).filter(({ k }) => r[k] !== 'free').map(({ s, k }) => [`#${SEPARABLE.data.indexOf(s) + 1}`, SPECIES_NAME[s.species], vn(s.raw[0], 1), vn(s.raw[1], 1), vn(fit.model.alpha[k], 3), vn(s.y * decision(fit.model, s.x), 3)]), caption: 'Chỉ liệt kê support vector (αᵢ > 0).' }; },
};

// ---------- Chapter 3: soft margin ----------
const SOFT = C_GRID.map(c => train(OVERLAP.data, c));
const cIndexNow = () => state.cIndex ?? Math.min(C_GRID.length - 1, Math.floor(seg(currentT, 1, 10) * C_GRID.length));

const chapter3: Scene = {
  title: 'Lề mềm và tham số C',
  duration: 11.5,
  idle: true,
  beats: () => [
    { at: 0, text: 'Versicolor (tròn) và virginica (vuông) chồng lấn ở vài bông: không có dải nào sạch hoàn toàn.' },
    { at: 2, text: 'Lề mềm cho phép điểm lấn vào dải hoặc sang nhầm phía, mỗi vi phạm bị phạt theo hinge loss nhân C.' },
    { at: 5, text: `C nhỏ: dải rộng, rất nhiều điểm thành support vector (${supportCount(SOFT[0])} điểm ở C = 0,01).` },
    { at: 8, text: `C lớn: dải co lại để ôm sát các điểm khó (${supportCount(SOFT.at(-1)!)} support vector ở C = 100). Chọn C bằng cross-validation.` },
  ],
  formula: () => `min ½‖w‖² + C Σ max(0, 1 − yᵢ(w·xᵢ + b)),  C = ${vn(C_GRID[cIndexNow()], 2)}`,
  render(f) {
    const { t, narrow, time } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.4, 1] : [1.5, 1], narrow);
    const i = cIndexNow(), model = SOFT[i];
    const plot = canvasFor(f, left, OVERLAP, `Lề mềm · C = ${vn(C_GRID[i], 2)}`);
    plot.clip(() => field(plot, OVERLAP, x => decision(model, x)));
    axes(plot, OVERLAP);
    const r = roles(model), rings = new Set<number>(), violate = new Set<number>();
    r.forEach((role, k) => { if (role !== 'free') rings.add(k); if (role === 'bound') violate.add(k); });
    points(plot, OVERLAP, { rings, violate, time });
    legend(f, plot, OVERLAP);
    categoryChart(f, right, 'Theo C: support vector và điểm sai', C_GRID.map(c => vn(c, c < 1 ? 2 : 0)), [
      { label: 'support vector', values: SOFT.map(supportCount), color: C.amber },
      { label: 'vi phạm lề', values: SOFT.map(violations), color: C.red },
      { label: 'phân loại sai', values: SOFT.map(m => errors(m)), color: C.cyan },
    ], [0, 80], i, state.cIndex === null ? seg(t, 1, 10) + 0.12 : 1, v => vn(v, 0), 'C');
  },
  metrics() { const m = SOFT[cIndexNow()]; return [{ label: 'C', value: vn(C_GRID[cIndexNow()], 2), tone: 'violet' }, { label: 'Độ rộng lề', value: vn(marginWidth(m), 3), tone: 'lime' }, { label: 'Support vector', value: String(supportCount(m)), tone: 'amber' }, { label: 'Đúng', value: `${vn((1 - errors(m) / m.data.length) * 100, 0)}%`, tone: 'cyan' }]; },
  table: () => ({ headers: ['C', 'Độ rộng lề', 'Support vector', 'Vi phạm lề', 'Sai', 'Độ chính xác'], rows: SOFT.map((m, i) => [vn(C_GRID[i], 2), vn(marginWidth(m), 3), supportCount(m), violations(m), errors(m), `${vn((1 - errors(m) / m.data.length) * 100, 0)}%`]) }),
};
function violations(m: Model) { return m.data.filter(p => p.y * decision(m, p.x) < 1 - 1e-3).length; }

// ---------- Chapter 4: kernel ----------
const LINEAR = train(MIDDLE.data, 1);
const RBF = new Map<number, Model>();
const rbf = (i: number) => { if (!RBF.has(i)) RBF.set(i, train(MIDDLE.data, 1, { type: 'rbf', gamma: GAMMA_GRID[i] })); return RBF.get(i)!; };
let cv: number[] | null = null;
const cvScores = () => (cv ??= GAMMA_GRID.map(gamma => crossValidate(MIDDLE.data, 1, { type: 'rbf', gamma })));
const VIEW = VIEWS.middle;
const toUV = (s: Sample): [number, number] => [-1 + 2 * (s.raw[0] - VIEW.x[0]) / (VIEW.x[1] - VIEW.x[0]), -1 + 2 * (s.raw[1] - VIEW.y[0]) / (VIEW.y[1] - VIEW.y[0])];
const fromUV = (u: number, v: number): Point => [VIEW.x[0] + (u + 1) / 2 * (VIEW.x[1] - VIEW.x[0]), VIEW.y[0] + (v + 1) / 2 * (VIEW.y[1] - VIEW.y[0])];
const lift = (value: number) => 0.5 + 0.42 * Math.tanh(value / 1.4);

const chapter4: Scene = {
  title: 'Kernel: uốn cong ranh giới',
  duration: 14,
  idle: true,
  beats: () => [
    { at: 0, text: 'Versicolor nằm kẹp giữa setosa và virginica. Một đường thẳng không thể tách loài ở giữa: chỉ đúng 66,7%, bằng đoán “không phải versicolor” cho mọi bông.' },
    { at: 4, text: 'Kernel RBF đặt một “ngọn đồi” quanh mỗi support vector. Cộng lại, f(x) thành một mặt cong nhô lên ở vùng versicolor.' },
    { at: 8, text: 'Cắt mặt cong ở độ cao 0 (đường trắng) cho ranh giới đóng kín quanh versicolor: đúng 96,7%.' },
    { at: 11, text: 'γ lớn làm mỗi ngọn đồi hẹp lại: độ chính xác huấn luyện tăng nhưng cross-validation giảm. Đó là overfitting.' },
  ],
  formula: () => `K(x, x′) = exp(−γ‖x − x′‖²),  γ = ${vn(GAMMA_GRID[state.gammaIndex], 2)}\nf(x) = Σ αᵢ yᵢ K(xᵢ, x) + b`,
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, rightAll] = split(f.safe, narrow ? [1.1, 1.3] : [1.15, 1], narrow);
    const [right, bottom] = split(rightAll, [1.7, 1], true, 12);
    const model = rbf(state.gammaIndex), useRbf = t >= 4;
    // Left: the RBF decision function as a surface; the white loop is f = 0.
    g.panel(left.x, left.y, left.w, left.h, 'f(x) của kernel RBF như một mặt cong', C.violet);
    const cam = state.camera;
    cam.rect = { x: left.x + 8, y: left.y + 30, w: left.w - 16, h: left.h - 36 };
    if (!state.dragging) cam.yaw += t > 9 ? 0.0025 : 0;
    const rise = seg(t, 4, 7.5);
    floor(g, cam, ['dài cánh', 'rộng cánh']);
    const fn = (u: number, v: number) => decision(model, toStd(MIDDLE, fromUV(u, v)));
    surface(g, cam, (u, v) => lerp(0.02, lift(fn(u, v)), rise), narrow ? 26 : 34, 0.85);
    if (rise > 0.6) {
      const segments = march(sample(fn, [-1, 1], [-1, 1], 50, 50), 0);
      g.ctx.save(); g.ctx.strokeStyle = C.white; g.ctx.lineWidth = 2.4; g.ctx.shadowColor = C.white; g.ctx.shadowBlur = 10; g.ctx.globalAlpha = seg(t, 7.5, 8.5); g.ctx.beginPath();
      for (const [a, b] of segments) { const p = cam.project([a[0], a[1], lift(0)]), q = cam.project([b[0], b[1], lift(0)]); g.ctx.moveTo(p[0], p[1]); g.ctx.lineTo(q[0], q[1]); }
      g.ctx.stroke(); g.ctx.restore();
    }
    MIDDLE.data.forEach(s => {
      const [u, v] = toUV(s), z = lerp(0.02, lift(decision(model, s.x)), rise), p = cam.project([u, v, z + 0.01]);
      g.circle(p[0], p[1], 3.2, SPECIES_COLOR[s.species], { glow: 6 });
    });
    // Right top: decision regions, linear first then RBF.
    const plot = canvasFor(f, right, MIDDLE, useRbf ? `Kernel RBF · γ = ${vn(GAMMA_GRID[state.gammaIndex], 2)}` : 'Kernel tuyến tính');
    plot.clip(() => field(plot, MIDDLE, x => decision(useRbf ? model : LINEAR, x), false));
    axes(plot, MIDDLE);
    const rings = new Set<number>();
    roles(useRbf ? model : LINEAR).forEach((role, k) => { if (role !== 'free' && useRbf) rings.add(k); });
    points(plot, MIDDLE, { rings });
    const acc = 1 - errors(useRbf ? model : LINEAR) / MIDDLE.data.length;
    g.tag(`đúng ${vn(acc * 100, 1)}% trên 150 bông`, plot.box.x + plot.box.w - 8, plot.box.y + 14, useRbf ? C.lime : C.red, 'right');
    const train_ = GAMMA_GRID.map((_, i) => 1 - errors(rbf(i)) / MIDDLE.data.length);
    categoryChart(f, bottom, 'Theo γ: huấn luyện và cross-validation 5 phần', GAMMA_GRID.map(v => vn(v, v < 1 ? 2 : 0)), [
      { label: 'huấn luyện', values: train_, color: C.cyan, format: v => `${vn(v * 100, 1)}%` },
      { label: 'cross-validation', values: cvScores(), color: C.lime, format: v => `${vn(v * 100, 1)}%` },
    ], [0.6, 1], state.gammaIndex, 1, v => `${vn(v * 100, 0)}%`, 'γ');
  },
  metrics() { const m = rbf(state.gammaIndex); return [{ label: 'Tuyến tính', value: `${vn((1 - errors(LINEAR) / 150) * 100, 1)}%`, tone: 'red' }, { label: 'RBF (huấn luyện)', value: `${vn((1 - errors(m) / 150) * 100, 1)}%`, tone: 'cyan' }, { label: 'RBF (cross-validation)', value: `${vn(cvScores()[state.gammaIndex] * 100, 1)}%`, tone: 'lime' }, { label: 'Support vector', value: String(supportCount(m)), tone: 'amber' }]; },
  pointer(e) {
    if (e.type === 'down') state.dragging = true;
    if (e.type === 'up' || e.type === 'leave') state.dragging = false;
    if (e.type === 'move' && e.dragging) { state.camera.drag(e.dx, e.dy); return true; }
    return false;
  },
  table: () => ({ headers: ['γ', 'Huấn luyện', 'Cross-validation 5 phần', 'Support vector'], rows: GAMMA_GRID.map((gamma, i) => [vn(gamma, 2), `${vn((1 - errors(rbf(i)) / 150) * 100, 1)}%`, `${vn(cvScores()[i] * 100, 1)}%`, supportCount(rbf(i))]) }),
  enter() { state.camera.yaw = -0.5; state.camera.pitch = 0.75; },
};

export const initSvmLab = () => glassLab('svm', [chapter1, chapter2, chapter3, chapter4], lab => {
  lab.range('angle', value => `${value}°`, value => { state.angle = value; });
  lab.range('c', value => vn(C_GRID[value], 2), value => { state.cIndex = value; });
  lab.range('gamma', value => vn(GAMMA_GRID[value], 2), value => { state.gammaIndex = value; });
  lab.on('[data-reset-points]', 'click', () => state.removed.clear());
  lab.on('[data-autoplay]', 'click', () => { state.angle = null; state.cIndex = null; lab.replay(); });
});
