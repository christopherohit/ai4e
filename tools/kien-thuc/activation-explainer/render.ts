// Activation lab: why a neural network needs a non-linearity. Affine maps only slide, turn and shear the plane (the
// grid stays straight), stacking them collapses to one affine map, ReLU folds the plane, a trained tanh layer lifts the
// two moons into a 3-D hidden space where one plane separates them, and the decision boundaries compare networks.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, Camera, Plot, alpha, fitEqual, lerp, seg, split, vn, type Point, type Point3, type Rect } from '../../_shared/glass-lab/gfx';
import { categoryChart, lossChart } from '../../_shared/glass-lab/views';
import { IDENTITY, LABELS, LAYER1, LAYER2, LINEAR, MERGED, MODELS, NETS, POINTS, STEPS, XOR, accuracy, apply, collapse, hidden, linearLogit, logit, mix, relu, scoreOf, testAccuracyOf, xorHidden, xorOut, type Affine, type ModelKey, type Vec } from './model';

export const explainerKind = 'educational-simulation';

const CLASS = [C.cyan, C.amber];
const state = { model: null as ModelKey | null, epoch: null as number | null, fold: null as number | null, camera: new Camera({ x: 0, y: 0, w: 1, h: 1 }), dragging: false };
let currentT = 0;

const PAD = { l: 18, r: 18, t: 40, b: 18 };
function plane(f: Frame, rect: Rect, title: string, accent: string = C.violet, xr: Point = [-2, 2], yr: Point = [-1.6, 1.6]) {
  f.g.panel(rect.x, rect.y, rect.w, rect.h, title, accent);
  const [x, y] = fitEqual(rect, PAD, xr, yr);
  return new Plot(f.g, rect, x, y, PAD);
}

/** Draw the image of a square grid under `map`; straight lines stay straight only if `map` is affine. */
function grid(plot: Plot, map: (p: Vec) => Vec, extent = 2.4, step = 0.4, strength = 1) {
  plot.clip(() => {
    for (let k = -extent; k <= extent + 1e-9; k += step) {
      const major = Math.abs(k) < 1e-9;
      const color = major ? alpha(C.white, 0.5 * strength) : alpha(C.violet, 0.28 * strength);
      const vertical: Point[] = [], horizontal: Point[] = [];
      for (let s = -extent; s <= extent + 1e-9; s += step / 6) { vertical.push(map([k, s])); horizontal.push(map([s, k])); }
      plot.line(vertical, color, { width: major ? 1.6 : 1 });
      plot.line(horizontal, color, { width: major ? 1.6 : 1 });
    }
  });
}

function dots(plot: Plot, map: (p: Vec) => Vec, opts: { r?: number; dim?: number } = {}) {
  plot.clip(() => POINTS.forEach((p, i) => { const [x, y] = map(p); plot.g.circle(plot.px(x), plot.py(y), opts.r ?? 2.8, CLASS[LABELS[i]], { glow: 6, alpha: opts.dim ?? 1 }); }));
}

function field(plot: Plot, score: (p: Vec) => number, res = 90) {
  plot.clip(() => {
    plot.heat((x, y) => score([x, y]), v => { const c = v > 0 ? [255, 182, 72] : [79, 214, 255], k = 0.08 + 0.2 * Math.min(1, Math.abs(v) / 4); return [c[0] * k + 6, c[1] * k + 9, c[2] * k + 20, 255]; }, res);
    plot.contours((x, y) => score([x, y]), [0], () => C.white, { width: 2.4, glow: 12 });
  });
}

const fmtM = ({ A, b }: Affine) => `[[${vn(A[0][0], 2)}; ${vn(A[0][1], 2)}], [${vn(A[1][0], 2)}; ${vn(A[1][1], 2)}]] · x + (${vn(b[0], 2)}; ${vn(b[1], 2)})`;

// ---------- Chapter 1: affine maps keep lines straight ----------
const stepNow = (t: number): Affine => {
  if (t < 1.5) return IDENTITY;
  if (t < 4) return mix(IDENTITY, STEPS[0], seg(t, 1.5, 4));
  if (t < 6.5) return mix(STEPS[0], STEPS[1], seg(t, 4, 6.5));
  return mix(STEPS[1], STEPS[2], seg(t, 6.5, 9));
};
const chapter1: Scene = {
  title: 'Affine chỉ kéo, xoay và trượt',
  duration: 13,
  beats: () => [
    { at: 0, text: 'Hai “mặt trăng” lồng vào nhau: 400 điểm, hai lớp. Lưới vuông bên dưới là chính mặt phẳng.' },
    { at: 1.5, text: 'Một lớp tuyến tính tính W·x + b. Hình học của nó: xoay, co giãn, trượt và dịch cả mặt phẳng.' },
    { at: 5, text: 'Dù biến đổi thế nào, mọi đường lưới vẫn thẳng và song song. Không có chỗ nào bị uốn cong hay gập lại.' },
    { at: 9.5, text: `Vì vậy ranh giới tốt nhất của một mô hình tuyến tính luôn là một đường thẳng: chỉ đúng ${vn(LINEAR.testAccuracy * 100, 1)}% trên tập kiểm tra.` },
  ],
  formula: () => `h = W·x + b  →  ${fmtM(stepNow(currentT))}`,
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.2, 1] : [1.25, 1], narrow);
    const M = stepNow(t), map = (p: Vec) => apply(M, p);
    const plot = plane(f, left, 'Mặt phẳng sau phép biến đổi affine');
    grid(plot, map);
    dots(plot, map);
    f.g.tag('đường lưới vẫn thẳng', plot.box.x + plot.box.w - 8, plot.box.y + plot.box.h - 14, C.violet, 'right');
    const fit = plane(f, right, 'Ranh giới tốt nhất: một đường thẳng', C.amber);
    const reveal = seg(t, 9.5, 11);
    if (reveal > 0) fit.clip(() => { f.g.ctx.globalAlpha = reveal; field(fit, linearLogit, 80); f.g.ctx.globalAlpha = 1; });
    grid(fit, p => p, 2.4, 0.4, 0.5);
    dots(fit, p => p);
    if (reveal > 0) f.g.tag(`đúng ${vn(LINEAR.testAccuracy * 100, 1)}% (kiểm tra)`, fit.box.x + fit.box.w - 8, fit.box.y + 14, C.red, 'right');
  },
  metrics: () => [{ label: 'Lớp', value: 'tuyến tính', tone: 'violet' }, { label: 'Đường thẳng còn thẳng', value: 'luôn luôn', tone: 'lime' }, { label: 'Logistic (kiểm tra)', value: `${vn(LINEAR.testAccuracy * 100, 1)}%`, tone: 'red' }],
  table: () => ({ headers: ['Bước', 'Ma trận W', 'Độ dịch b'], rows: STEPS.map((s, i) => [String(i + 1), `[[${vn(s.A[0][0], 3)}; ${vn(s.A[0][1], 3)}], [${vn(s.A[1][0], 3)}; ${vn(s.A[1][1], 3)}]]`, `(${vn(s.b[0], 3)}; ${vn(s.b[1], 3)})`]) }),
};

// ---------- Chapter 2: stacking affine layers ----------
const chapter2: Scene = {
  title: 'Chồng nhiều lớp affine vẫn là một lớp',
  duration: 12,
  beats: () => [
    { at: 0, text: 'Thử xếp hai lớp tuyến tính liên tiếp, như một mạng “sâu” không có hàm kích hoạt.' },
    { at: 2, text: 'Lớp 1 xoay và co mặt phẳng. Lớp 2 trượt nó thêm một lần nữa.' },
    { at: 6.5, text: 'Ô cuối cùng làm điều tương tự chỉ bằng một lớp: W = W₂W₁, b = W₂b₁ + b₂. Hai hình trùng khít.' },
    { at: 9.5, text: 'Thêm bao nhiêu lớp tuyến tính cũng vậy: độ sâu không mua được gì nếu thiếu phi tuyến.' },
  ],
  formula: () => `W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)\nW₂W₁ = [[${vn(MERGED.A[0][0], 3)}; ${vn(MERGED.A[0][1], 3)}], [${vn(MERGED.A[1][0], 3)}; ${vn(MERGED.A[1][1], 3)}]],  W₂b₁ + b₂ = (${vn(MERGED.b[0], 3)}; ${vn(MERGED.b[1], 3)})`,
  render(f) {
    const { t, narrow } = f; currentT = t;
    const rows = split(f.safe, [1, 1], true, 12);
    const [a, b] = split(rows[0], [1, 1], false, 12), [c, d] = split(rows[1], [1, 1], false, 12);
    const p1 = seg(t, 2, 4), p2 = seg(t, 4, 6), p3 = seg(t, 6.5, 9);
    const L1 = mix(IDENTITY, LAYER1, p1), L2 = mix(IDENTITY, LAYER2, p2);
    const panels: [Rect, string, (p: Vec) => Vec, string][] = [
      [a, 'Đầu vào x', p => p, C.cyan],
      [b, 'Sau lớp 1: W₁x + b₁', p => apply(L1, p), C.violet],
      [c, 'Sau lớp 2: W₂(W₁x + b₁) + b₂', p => apply(L2, apply(LAYER1, p)), C.violet],
      [d, 'Một lớp duy nhất: (W₂W₁)x + (W₂b₁ + b₂)', p => apply(mix(IDENTITY, MERGED, p3), p), C.lime],
    ];
    panels.forEach(([rect, title, map, accent], i) => {
      const plot = plane(f, rect, title, accent, [-2.4, 2.4], [-1.8, 1.8]);
      grid(plot, i === 2 && p2 === 0 ? p => apply(LAYER1, p) : map, 2.4, 0.4, 0.9);
      dots(plot, i === 2 && p2 === 0 ? p => apply(LAYER1, p) : map, { r: narrow ? 1.8 : 2.2 });
    });
    if (p3 >= 1) f.g.tag('trùng khít với ô bên trái', d.x + d.w - 10, d.y + d.h - 14, C.lime, 'right');
  },
  metrics: () => [{ label: 'Số lớp tuyến tính', value: '2', tone: 'violet' }, { label: 'Tương đương', value: '1 lớp', tone: 'lime' }],
};

// ---------- Chapter 3: ReLU folds the plane; XOR ----------
const R2 = NETS.relu2;
const affineR2: Affine = { A: [[R2.W1[0][0], R2.W1[0][1]], [R2.W1[1][0], R2.W1[1][1]]], b: [R2.b1[0], R2.b1[1]] };
const foldNow = () => state.fold ?? seg(currentT, 5, 8);
const affNow = () => (state.fold !== null ? 1 : seg(currentT, 1.5, 4.5));
const chapter3: Scene = {
  title: 'ReLU gập mặt phẳng',
  duration: 15,
  idle: false,
  beats: () => [
    { at: 0, text: 'Thêm phi tuyến đơn giản nhất: ReLU(z) = max(0, z), áp lên từng tọa độ sau lớp affine.' },
    { at: 1.5, text: 'Bước affine của 2 nơron ReLU (đã huấn luyện) xoay và kéo hai mặt trăng ra.' },
    { at: 5, text: 'ReLU ép mọi giá trị âm về 0: nửa mặt phẳng bị gập lên trục. Đường lưới gãy khúc, điều mà affine không làm được.' },
    { at: 9, text: 'Bên phải là XOR, bài toán kinh điển không đường thẳng nào tách được. Hai nơron ReLU đưa (1,0) và (0,1) về cùng một điểm.' },
    { at: 12, text: 'Trong không gian ẩn, một đường thẳng đã đủ tách XOR. Phi tuyến làm cho “đường thẳng ở lớp sau” trở nên mạnh.' },
  ],
  formula: () => `h = ReLU(W·x + b),  ReLU(z) = max(0, z)\nXOR: h₁ = ReLU(x₁ + x₂), h₂ = ReLU(x₁ + x₂ − 1), y = h₁ − 2h₂`,
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.2, 1] : [1.3, 1], narrow);
    const a = affNow(), k = foldNow();
    const map = (p: Vec): Vec => { const z = apply(mix(IDENTITY, affineR2, a), p); return [lerp(z[0], relu(z[0]), k), lerp(z[1], relu(z[1]), k)]; };
    const plot = plane(f, left, `2 nơron ReLU: affine ${Math.round(a * 100)}%, gập ${Math.round(k * 100)}%`, C.violet, [-2.5, 4.5], [-2.5, 4.5]);
    grid(plot, map, 2.4, 0.3);
    dots(plot, map);
    if (k > 0.98) {
      const w: Vec = [R2.W2[0][0], R2.W2[0][1]], b = R2.b2[0];
      plot.clip(() => plot.contours((x, y) => w[0] * x + w[1] * y + b, [0], () => C.white, { width: 2.2, glow: 10 }));
      g.tag(`đường thẳng ở lớp sau · ${vn(R2.testAccuracy * 100, 1)}%`, plot.box.x + plot.box.w - 8, plot.box.y + plot.box.h - 14, C.amber, 'right');
    }
    // XOR
    const morph = seg(t, 9.5, 12);
    const xp = plane(f, right, morph > 0 ? 'XOR: từ đầu vào sang không gian ẩn' : 'XOR: không đường thẳng nào tách được', C.amber, [-0.5, 2.4], [-0.6, 1.6]);
    grid(xp, p => p, 3, 0.5, 0.5);
    if (t > 12) xp.clip(() => xp.contours((x, y) => xorOut([x, y]) - 0.5, [0], () => C.white, { width: 2.4, glow: 12 }));
    XOR.forEach(({ x, y }) => {
      const h = xorHidden(x), p: Vec = [lerp(x[0], h[0], morph), lerp(x[1], h[1], morph)];
      g.circle(xp.px(p[0]), xp.py(p[1]), 9, CLASS[y], { glow: 14 });
      g.text(`(${x[0]},${x[1]})`, xp.px(p[0]) + (x[0] === 0 && x[1] === 1 ? -14 : 14), xp.py(p[1]) - 14, { size: 11.5, color: C.ink, align: x[0] === 0 && x[1] === 1 ? 'right' : 'left' });
    });
    if (morph >= 1) g.tag('(1,0) và (0,1) trùng nhau', xp.box.x + xp.box.w - 8, xp.box.y + xp.box.h - 14, C.lime, 'right');
  },
  metrics: () => [{ label: '2 nơron ReLU (kiểm tra)', value: `${vn(R2.testAccuracy * 100, 1)}%`, tone: 'amber' }, { label: 'XOR sau ReLU', value: '4/4 đúng', tone: 'lime' }],
  table: () => ({ headers: ['x₁', 'x₂', 'XOR', 'h₁', 'h₂', 'y = h₁ − 2h₂'], rows: XOR.map(({ x, y }) => { const h = xorHidden(x); return [x[0], x[1], y, vn(h[0], 0), vn(h[1], 0), vn(xorOut(h), 0)]; }) }),
};

// ---------- Chapter 4: 3-D hidden space of a trained tanh layer ----------
const T3 = NETS.tanh3;
const SNAPS = T3.snapshots!;
const snapNow = () => state.epoch ?? Math.min(SNAPS.length - 1, Math.floor(seg(currentT, 2, 11) * SNAPS.length));
const toCube = (h: number[]): Point3 => [h[0], h[1], (h[2] + 1) * 0.45];
function planePolygon(w: number[], b: number): Point3[] {
  // Intersect w·h + b = 0 with the 12 edges of the cube [−1, 1]³ and sort the points around their centre.
  const corners: number[][] = [];
  for (let i = 0; i < 8; i++) corners.push([i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1]);
  const pts: number[][] = [];
  for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
    const diff = corners[i].reduce((s, v, k) => s + (v !== corners[j][k] ? 1 : 0), 0);
    if (diff !== 1) continue;
    const fi = corners[i].reduce((s, v, k) => s + v * w[k], b), fj = corners[j].reduce((s, v, k) => s + v * w[k], b);
    if (fi * fj > 0 || fi === fj) continue;
    const u = fi / (fi - fj);
    pts.push(corners[i].map((v, k) => v + (corners[j][k] - v) * u));
  }
  if (pts.length < 3) return [];
  const c = [0, 1, 2].map(k => pts.reduce((s, p) => s + p[k], 0) / pts.length);
  const n = w.map(v => v / Math.hypot(...w));
  const ref = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = [n[1] * ref[2] - n[2] * ref[1], n[2] * ref[0] - n[0] * ref[2], n[0] * ref[1] - n[1] * ref[0]];
  const e2 = [n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2], n[0] * e1[1] - n[1] * e1[0]];
  return pts.map(p => ({ p, a: Math.atan2((p[0] - c[0]) * e2[0] + (p[1] - c[1]) * e2[1] + (p[2] - c[2]) * e2[2], (p[0] - c[0]) * e1[0] + (p[1] - c[1]) * e1[1] + (p[2] - c[2]) * e1[2]) })).sort((x, y) => x.a - y.a).map(({ p }) => toCube(p));
}
const chapter4: Scene = {
  title: 'Không gian ẩn: nơi dữ liệu thành tách được',
  duration: 14,
  idle: true,
  beats: () => [
    { at: 0, text: 'Một lớp ẩn 3 nơron tanh biến mỗi điểm (x₁, x₂) thành một điểm (h₁, h₂, h₃) trong khối lập phương [−1, 1]³.' },
    { at: 2, text: 'Theo dõi quá trình huấn luyện: lúc đầu hai lớp trộn lẫn, dần dần tanh uốn và kéo chúng về hai phía.' },
    { at: 8, text: 'Lớp đầu ra chỉ là một mặt phẳng (tấm trắng) trong không gian ẩn. Khi huấn luyện xong, mặt phẳng ấy tách gọn hai lớp.' },
    { at: 11.5, text: `Kết quả: ${vn(T3.testAccuracy * 100, 1)}% trên 2 000 điểm kiểm tra. Kéo để xoay khối lập phương.` },
  ],
  formula: () => { const s = SNAPS[snapNow()]; return `h = tanh(W₁x + b₁) ∈ [−1, 1]³,  ŷ = σ(w₂·h + b₂)\nepoch ${s.epoch}: loss = ${vn(s.loss, 4)}`; },
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.5, 1] : [1.5, 1], narrow);
    const s = SNAPS[snapNow()], net = { ...s, act: 'tanh' as const };
    g.panel(left.x, left.y, left.w, left.h, `Không gian ẩn 3 chiều · epoch ${s.epoch}`, C.violet);
    const cam = state.camera;
    cam.rect = { x: left.x + 8, y: left.y + 30, w: left.w - 16, h: left.h - 36 };
    if (!state.dragging) cam.yaw += 0.003;
    // cube wireframe
    for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
      const a = [i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1], b = [j & 1 ? 1 : -1, j & 2 ? 1 : -1, j & 4 ? 1 : -1];
      if (a.reduce((n, v, k) => n + (v !== b[k] ? 1 : 0), 0) !== 1) continue;
      const p = cam.project(toCube(a)), q = cam.project(toCube(b));
      g.line([[p[0], p[1]], [q[0], q[1]]], alpha(C.white, 0.18), { width: 1 });
    }
    const poly = planePolygon(s.W2[0], s.b2[0]);
    const pts = POINTS.map((p, i) => ({ q: cam.project(toCube(hidden(net, p))), c: CLASS[LABELS[i]] })).sort((a, b) => b.q[2] - a.q[2]);
    if (poly.length && t > 8) g.polygon(poly.map(p => { const q = cam.project(p); return [q[0], q[1]] as Point; }), alpha(C.white, 0.12 * seg(t, 8, 9)), alpha(C.white, 0.6 * seg(t, 8, 9)), 1.4);
    for (const { q, c } of pts) g.circle(q[0], q[1], 2.6, c, { glow: 5 });
    ['h₁', 'h₂', 'h₃'].forEach((label, k) => { const e = [0, 0, 0]; e[k] = 1.18; const q = cam.project(toCube(k === 2 ? [0, 0, 1.25] : e)); g.text(label, q[0], q[1], { size: 12, color: C.muted, align: 'center' }); });
    const [top, bottom] = split(right, [1.2, 1], true, 12);
    const input = plane(f, top, 'Ranh giới tương ứng ở đầu vào', C.amber, [-1.8, 1.8], [-1.1, 1.1]);
    field(input, p => logit(net, p), 70);
    dots(input, p => p, { r: 2 });
    lossChart(f, bottom, [{ values: SNAPS.map(x => x.loss), color: C.amber, x: SNAPS.map(x => Math.log10(x.epoch)) }], Math.log10(s.epoch), 0.8, 0, 'Loss theo epoch (trục log)', 'log₁₀ epoch', Math.log10(SNAPS.at(-1)!.epoch), v => vn(v, 1));
  },
  metrics: () => { const s = SNAPS[snapNow()], net = { ...s, act: 'tanh' as const }; return [{ label: 'Epoch', value: String(s.epoch), tone: 'violet' }, { label: 'Loss', value: vn(s.loss, 3), tone: 'amber' }, { label: 'Đúng (huấn luyện)', value: `${vn(accuracy(p => logit(net, p)) * 100, 1)}%`, tone: 'lime' }]; },
  pointer(e) {
    if (e.type === 'down') state.dragging = true;
    if (e.type === 'up' || e.type === 'leave') state.dragging = false;
    if (e.type === 'move' && e.dragging) { state.camera.drag(e.dx, e.dy); return true; }
    return false;
  },
  enter() { state.camera.yaw = 0.6; state.camera.pitch = 0.5; },
  table: () => ({ headers: ['Epoch', 'Loss'], rows: SNAPS.map(s => [s.epoch, vn(s.loss, 4)]) }),
};

// ---------- Chapter 5: decision boundaries ----------
const modelNow = (): ModelKey => state.model ?? MODELS[Math.min(MODELS.length - 1, Math.floor(seg(currentT, 0.5, 12) * MODELS.length))].key;
const chapter5: Scene = {
  title: 'So sánh ranh giới quyết định',
  duration: 13,
  idle: false,
  beats: () => [
    { at: 0, text: `Hồi quy logistic: một đường thẳng, ${vn(LINEAR.testAccuracy * 100, 1)}%.` },
    { at: 2.8, text: `8 nơron nhưng không có hàm kích hoạt: gộp lại đúng thành một đường thẳng, ${vn(NETS.none8.testAccuracy * 100, 1)}%.` },
    { at: 5.2, text: `2 nơron ReLU: ranh giới gãy khúc, ${vn(NETS.relu2.testAccuracy * 100, 1)}%. 8 nơron ReLU: nhiều nếp gập, ${vn(NETS.relu8.testAccuracy * 100, 1)}%.` },
    { at: 10, text: `3 nơron tanh uốn mềm hơn: ${vn(NETS.tanh3.testAccuracy * 100, 1)}%. Chọn mô hình ở bảng điều khiển để so sánh.` },
  ],
  formula: () => { const k = modelNow(); if (k === 'none8') { const c = collapse(NETS.none8); return `không kích hoạt ⇒ ŷ = σ(${vn(c.w[0], 2)}·x₁ + ${vn(c.w[1], 2)}·x₂ + ${vn(c.b, 2)}): vẫn là một đường thẳng`; } return k === 'linear' ? 'ŷ = σ(w·x + b)' : `ŷ = σ(w₂·${k.startsWith('relu') ? 'ReLU' : 'tanh'}(W₁x + b₁) + b₂)`; },
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.4, 1] : [1.5, 1], narrow);
    const key = modelNow(), meta = MODELS.find(m => m.key === key)!;
    const plot = plane(f, left, `${meta.label} · ${meta.note}`, C.violet, [-1.9, 1.9], [-1.2, 1.2]);
    field(plot, scoreOf(key));
    dots(plot, p => p);
    f.g.tag(`đúng ${vn(testAccuracyOf(key) * 100, 1)}% (kiểm tra)`, plot.box.x + plot.box.w - 8, plot.box.y + 14, testAccuracyOf(key) > 0.95 ? C.lime : C.red, 'right');
    categoryChart(f, right, 'Độ chính xác trên 2 000 điểm kiểm tra', MODELS.map(m => m.key === 'linear' ? 'logistic' : m.key === 'none8' ? '8 tuyến tính' : m.key === 'relu2' ? '2 ReLU' : m.key === 'relu8' ? '8 ReLU' : '3 tanh'), [
      { label: 'kiểm tra', values: MODELS.map(m => testAccuracyOf(m.key)), color: C.lime, format: v => `${vn(v * 100, 1)}%` },
    ], [0.8, 1], MODELS.findIndex(m => m.key === key), 1, v => `${vn(v * 100, 0)}%`);
  },
  metrics: () => { const key = modelNow(); return [{ label: 'Mô hình', value: MODELS.find(m => m.key === key)!.label, tone: 'violet' }, { label: 'Kiểm tra', value: `${vn(testAccuracyOf(key) * 100, 1)}%`, tone: testAccuracyOf(key) > 0.95 ? 'lime' : 'red' }]; },
  table: () => ({ headers: ['Mô hình', 'Huấn luyện', 'Kiểm tra (2 000 điểm)'], rows: MODELS.map(m => [m.label, `${vn((m.key === 'linear' ? LINEAR.trainAccuracy : NETS[m.key].trainAccuracy) * 100, 1)}%`, `${vn(testAccuracyOf(m.key) * 100, 1)}%`]) }),
};

export const initActivationLab = () => glassLab('activation', [chapter1, chapter2, chapter3, chapter4, chapter5], lab => {
  lab.range('fold', value => `${value}%`, value => { state.fold = value / 100; });
  lab.range('epoch', value => String(SNAPS[value].epoch), value => { state.epoch = value; });
  lab.on('input[name="model"]', 'change', element => { state.model = (element as HTMLInputElement).value as ModelKey; lab.finish(); });
  lab.on('[data-autoplay]', 'click', () => { state.model = null; state.epoch = null; state.fold = null; lab.replay(); });
});
