// Gradient descent lab: four animated chapters on Auto MPG. Fit a line to 392 real cars, roll down the 3D loss
// surface, race four learning rates, then fix the narrow valley by standardizing the feature.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, Camera, Plot, alpha, clamp, floor, inset, march, sample, seg, split, surface, vn, type Point, type Rect } from '../../_shared/glass-lab/gfx';
import type { Theta } from '../../_shared/glass-lab/regression';
import { at, heightOf, lossChart, lossMap, trail, type Series } from '../../_shared/glass-lab/views';
import { CARS, LR_PRESETS, START_RAW, START_STD, VIEW_RAW, VIEW_STD, WEIGHT_MEAN, WEIGHT_SD, raw, std, stdToRaw } from './model';

export const explainerKind = 'educational-simulation';

const ORIGIN: Record<string, string> = { usa: C.cyan, europe: C.violet, japan: C.lime };
const ORIGIN_NAME: Record<string, string> = { usa: 'Mỹ', europe: 'châu Âu', japan: 'Nhật' };
const PRESET_COLOR = [C.cyan, C.lime, C.amber, C.magenta];
const state = { hover: -1, lr: 0.09 as number, camera: new Camera({ x: 0, y: 0, w: 1, h: 1 }), dragging: false };

const zRaw = heightOf(raw, 8, 2400), zStd = heightOf(std, 8, 2400);
const mapRaw = (f: Frame, rect: Rect, title: string, labels: [string, string] = ['w (mpg / nghìn lb)', 'b (mpg)']) => lossMap(f, rect, raw, VIEW_RAW, zRaw, title, labels);
const chart = (f: Frame, rect: Rect, series: Series[], k: number) => lossChart(f, rect, series, k, 100, raw.minLoss);

// ---------- Chapter 1: real data ----------
/** Timeline position of the chapter being drawn; metrics are read right after render. */
let currentT = 0;
const PATH1 = std.descend(START_STD, 0.1, 40);
const LOSS1 = PATH1.map(p => std.loss(p));
let scatter: Plot | null = null;

const chapter1: Scene = {
  title: 'Dữ liệu thật: 392 chiếc xe',
  duration: 13,
  beats: () => [
    { at: 0, text: '392 mẫu xe đời 1970–1982: xe càng nặng, mỗi gallon xăng càng đi được ít dặm.' },
    { at: 2.6, text: 'Bắt đầu bằng một đường thẳng đoán bừa. Vạch hồng là sai số của từng xe.' },
    { at: 4.5, text: 'Mỗi bước, gradient descent xoay và dịch đường thẳng theo hướng làm MSE giảm nhanh nhất.' },
    { at: 9.5, text: `Sau 40 bước: mpg ≈ ${vn(stdToRaw(PATH1[40])[1], 1)} − ${vn(-stdToRaw(PATH1[40])[0], 2)} × (nghìn pound), đúng nghiệm bình phương tối thiểu.` },
  ],
  formula: 'MSE(w, b) = (1/392) Σ (w·xᵢ + b − mpgᵢ)²\nθ ← θ − η ∇MSE(θ),  η = 0,1 (x đã chuẩn hóa, lý do ở chương 4)',
  render(f) {
    const { g, t, narrow } = f;
    currentT = t;
    const [left, right] = split(f.safe, narrow ? [3, 2] : [1.7, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, 'Khối lượng và mức tiêu thụ nhiên liệu', C.cyan);
    const plot = new Plot(g, left, [1500, 5250], [6, 48], { l: 46, r: 16, t: 40, b: 36 });
    scatter = plot;
    plot.axes({ xTicks: [2000, 3000, 4000, 5000], yTicks: [10, 20, 30, 40], xLabel: 'khối lượng (pound)', yLabel: 'mpg (dặm / gallon)', fx: v => vn(v, 0), fy: v => vn(v, 0) });
    const k = seg(t, 4.5, 10.5) * 40;
    const theta = stdToRaw(at(PATH1, k));
    const line = (weight: number) => theta[0] * weight / 1000 + theta[1];
    const showLine = seg(t, 2.6, 3.6);
    plot.clip(() => {
      if (showLine > 0) CARS.forEach(car => plot.line([[car.weight, car.mpg], [car.weight, line(car.weight)]], alpha(C.magenta, 0.28 * showLine), { width: 1 }));
      CARS.forEach((car, i) => {
        const appear = seg(t, i / CARS.length * 2.2, i / CARS.length * 2.2 + 0.4);
        if (appear <= 0) return;
        g.circle(plot.px(car.weight), plot.py(car.mpg), (state.hover === i ? 6 : 3.1) * appear, ORIGIN[car.origin], { glow: state.hover === i ? 16 : 6, alpha: 0.9 });
      });
      if (showLine > 0) plot.line([[1500, line(1500)], [1500 + 3750 * showLine, line(1500 + 3750 * showLine)]], C.amber, { width: 3, glow: 18 });
    });
    legend(f, plot);
    if (state.hover >= 0) {
      const car = CARS[state.hover];
      const x = plot.px(car.weight), y = plot.py(car.mpg);
      g.tag(`${car.name} · ${car.year} · ${vn(car.weight, 0)} lb · ${vn(car.mpg, 1)} mpg`, clamp(x, plot.box.x + 150, plot.box.x + plot.box.w - 150), y - 18, ORIGIN[car.origin]);
    }
    chart(f, right, [{ values: LOSS1, color: C.amber }], k);
  },
  metrics() {
    const k = Math.round(seg(currentT, 4.5, 10.5) * 40), theta = stdToRaw(PATH1[k]);
    return [{ label: 'Bước', value: String(k), tone: 'violet' }, { label: 'MSE', value: vn(std.loss(PATH1[k]), 2), tone: 'amber' }, { label: 'Độ dốc w', value: `${vn(theta[0], 2)} mpg/nghìn lb`, tone: 'cyan' }];
  },
  pointer(e) {
    if (!scatter || e.type !== 'move') { if (e.type === 'leave') { state.hover = -1; return true; } return false; }
    let best = -1, distance = 14;
    CARS.forEach((car, i) => { const d = Math.hypot(scatter!.px(car.weight) - e.x, scatter!.py(car.mpg) - e.y); if (d < distance) { distance = d; best = i; } });
    if (best !== state.hover) { state.hover = best; return true; }
    return false;
  },
  table: () => ({ headers: ['Bước', 'w (mpg / nghìn lb)', 'b (mpg)', 'MSE'], rows: PATH1.filter((_, k) => k % 4 === 0 || k < 5).map((p, i, list) => { const k = PATH1.indexOf(list[i]); const r = stdToRaw(p); return [k, vn(r[0], 3), vn(r[1], 2), vn(LOSS1[k], 3)]; }) }),
};

function legend(f: Frame, plot: Plot) {
  let x = plot.box.x + plot.box.w - 8;
  for (const key of ['japan', 'europe', 'usa']) {
    const label = ORIGIN_NAME[key], width = f.g.measure(label, 11) + 18;
    x -= width;
    f.g.circle(x + 5, plot.box.y + 12, 4, ORIGIN[key], { glow: 8 });
    f.g.text(label, x + 13, plot.box.y + 12, { size: 11, color: C.muted });
  }
}

// ---------- Chapter 2: the 3D surface ----------
const PATH2 = raw.descend(START_RAW, 0.09, 230);
const toUV = ([w, b]: Theta): [number, number] => [-1 + 2 * (w - VIEW_RAW.w[0]) / (VIEW_RAW.w[1] - VIEW_RAW.w[0]), -1 + 2 * (b - VIEW_RAW.b[0]) / (VIEW_RAW.b[1] - VIEW_RAW.b[0])];
const fromUV = (u: number, v: number): Theta => [VIEW_RAW.w[0] + (u + 1) / 2 * (VIEW_RAW.w[1] - VIEW_RAW.w[0]), VIEW_RAW.b[0] + (v + 1) / 2 * (VIEW_RAW.b[1] - VIEW_RAW.b[0])];
const FLOOR_LINES = [0.08, 0.18, 0.3, 0.44, 0.6, 0.76].map(level => march(sample((u, v) => zRaw(fromUV(u, v)), [-1, 1], [-1, 1], 60, 60), level));

const chapter2: Scene = {
  title: 'Mặt loss 3D',
  duration: 15,
  idle: true,
  beats: () => [
    { at: 0, text: 'Mỗi điểm trên mặt phẳng là một đường thẳng (w, b); độ cao là MSE của đường đó trên 392 xe.' },
    { at: 3, text: 'Đáy thung lũng là đường thẳng tốt nhất. Viên bi bắt đầu từ đường đoán bừa ở chương 1.' },
    { at: 5.5, text: 'Gradient chỉ hướng dốc nhất. Bi lăn ngược gradient, vuông góc với đường đồng mức, nên đi chữ chi qua thung lũng hẹp.' },
    { at: 11, text: `Thung lũng dài gấp ${vn(Math.sqrt(raw.condition), 0)} lần bề ngang: bi tới đáy rất nhanh theo chiều ngang nhưng phải bò rất lâu dọc theo đáy.` },
  ],
  formula: () => `∇MSE = (2/n) Σ (w·xᵢ + b − yᵢ) · (xᵢ, 1)\nη = 0,09,  độ cong λ_max = ${vn(raw.curvatures[0], 1)}, λ_min = ${vn(raw.curvatures[1], 3)}`,
  enter() { state.camera.yaw = -0.85; state.camera.pitch = 0.6; },
  render(f) {
    const { g, t, narrow } = f;
    currentT = t;
    const [left, right] = split(f.safe, narrow ? [3, 2] : [1.55, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, 'MSE(w, b) với khối lượng tính bằng nghìn pound', C.violet);
    const cam = state.camera;
    cam.rect = inset(left, 10, 34, 10, 10);
    if (!state.dragging && f.progress >= 1) cam.yaw += 0.0025;
    floor(g, cam, ['w', 'b']);
    FLOOR_LINES.forEach((segments, index) => {
      const { ctx } = g;
      ctx.save(); ctx.strokeStyle = alpha(C.cyan, 0.16 + index * 0.02); ctx.lineWidth = 1; ctx.beginPath();
      for (const [a, b] of segments) { const p = cam.project([a[0], a[1], 0]), q = cam.project([b[0], b[1], 0]); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); }
      ctx.stroke(); ctx.restore();
    });
    surface(g, cam, (u, v) => zRaw(fromUV(u, v)) * 0.95, narrow ? 30 : 40, 0.88);
    const k = seg(t, 3, 14) * (PATH2.length - 1);
    const shown = [...PATH2.slice(0, Math.floor(k) + 1), at(PATH2, k)];
    const pts3 = shown.map(p => { const [u, v] = toUV(p); return cam.project([u, v, zRaw(p) * 0.95 + 0.012]); });
    g.line(shown.map(p => { const [u, v] = toUV(p); const q = cam.project([u, v, 0]); return [q[0], q[1]] as Point; }), alpha(C.amber, 0.45), { width: 1.5, dash: [3, 3] });
    g.line(pts3.map(p => [p[0], p[1]] as Point), C.amber, { width: 2.4, glow: 14 });
    const ball = pts3.at(-1)!, [bu, bv] = toUV(at(PATH2, k)), shadow = cam.project([bu, bv, 0]);
    g.line([[shadow[0], shadow[1]], [ball[0], ball[1]]], alpha(C.white, 0.3), { width: 1, dash: [2, 3] });
    g.circle(ball[0], ball[1], 8, C.amber, { glow: 26 });
    g.circle(ball[0] - 2, ball[1] - 2, 3, C.white);
    const opt = toUV(raw.optimum), o = cam.project([opt[0], opt[1], 0.012]);
    g.circle(o[0], o[1], 5, null, { stroke: C.white, width: 2 });
    const plot = mapRaw(f, right, 'Nhìn từ trên xuống');
    trail(plot, PATH2, k, C.amber, { dots: false });
    const p = at(PATH2, k), [gw, gb] = raw.gradient(p), norm = Math.hypot(gw, gb);
    if (norm > 1e-6 && t > 5.5) {
      const [x, y] = plot.p(p), len = 46;
      const dx = -gw / norm, dy = -gb / norm;
      const sx = plot.box.w / (VIEW_RAW.w[1] - VIEW_RAW.w[0]), sy = plot.box.h / (VIEW_RAW.b[1] - VIEW_RAW.b[0]);
      const px = dx * sx, py = -dy * sy, pn = Math.hypot(px, py);
      g.arrow(x, y, x + px / pn * len, y + py / pn * len, C.magenta, { width: 2.5, glow: 12 });
      g.tag('−∇MSE', x + px / pn * len, y + py / pn * len - 14, C.magenta);
    }
  },
  metrics() {
    const k = Math.min(PATH2.length - 1, Math.round(seg(currentT, 3, 14) * (PATH2.length - 1))), p = PATH2[k], [gw, gb] = raw.gradient(p);
    return [{ label: 'Bước', value: String(k), tone: 'violet' }, { label: 'MSE', value: vn(raw.loss(p), 2), tone: 'amber' }, { label: '‖∇MSE‖', value: vn(Math.hypot(gw, gb), 2), tone: 'magenta' }];
  },
  pointer(e) {
    if (e.type === 'down') state.dragging = true;
    if (e.type === 'up' || e.type === 'leave') state.dragging = false;
    if (e.type === 'move' && e.dragging) { state.camera.drag(e.dx, e.dy); return true; }
    return false;
  },
  table: () => ({ headers: ['Bước', 'w', 'b', 'MSE'], rows: [0, 1, 2, 3, 5, 10, 20, 50, 100, 150, 200, 230].map(k => [k, vn(PATH2[k][0], 3), vn(PATH2[k][1], 2), vn(raw.loss(PATH2[k]), 3)]) }),
};

// ---------- Chapter 3: learning rates ----------
const STEPS3 = 60;
const rates = () => [LR_PRESETS[0], LR_PRESETS[1], state.lr, LR_PRESETS[3]];
const paths3 = () => rates().map(lr => raw.descend(START_RAW, lr, STEPS3));
const lrText = (lr: number) => lr.toLocaleString('vi-VN', { maximumFractionDigits: 3 });
const verdict = (lr: number) => (lr >= raw.maxStableLr ? 'phân kỳ' : lr > 0.9 * raw.maxStableLr ? 'chữ chi, sát ngưỡng' : lr < 0.02 ? 'quá chậm' : 'hội tụ');

const chapter3: Scene = {
  title: 'Learning rate',
  duration: 12,
  beats: () => [
    { at: 0, text: 'Bốn viên bi cùng xuất phát, chỉ khác độ dài bước η.' },
    { at: 2.5, text: 'η = 0,01 bò rất chậm; η = 0,05 đều đặn nhưng vẫn chưa tới đáy sau 60 bước.' },
    { at: 6, text: `η gần ngưỡng ${vn(raw.maxStableLr, 3)} đi chữ chi mạnh nhưng tiến nhanh nhất dọc thung lũng.` },
    { at: 8.5, text: 'Chỉ vượt ngưỡng một chút (η = 0,1), mỗi bước lại văng xa hơn: MSE bùng nổ. Ngưỡng là 2 / λ_max.' },
  ],
  formula: () => `hội tụ ⇔ η < 2 / λ_max = 2 / ${vn(raw.curvatures[0], 2)} = ${vn(raw.maxStableLr, 4)}`,
  render(f) {
    const { g, t, narrow } = f;
    currentT = t;
    const [left, right] = split(f.safe, narrow ? [3, 2] : [1.45, 1], narrow);
    const plot = mapRaw(f, left, 'Bốn learning rate trên cùng mặt loss');
    const k = seg(t, 1, 11) * STEPS3, list = paths3();
    list.forEach((path, i) => trail(plot, path, Math.min(k, path.length - 1), PRESET_COLOR[i], { dots: i !== 0 }));
    rates().forEach((lr, i) => g.tag(`η = ${lrText(lr)} · ${verdict(lr)}`, plot.box.x + 8, plot.box.y + 14 + i * 26, PRESET_COLOR[i], 'left'));
    chart(f, right, list.map((path, i) => ({ values: path.map(p => raw.loss(p)), color: PRESET_COLOR[i] })), k);
  },
  metrics: () => paths3().map((path, i) => ({ label: `MSE · η = ${lrText(rates()[i])}`, value: (() => { const v = raw.loss(path[Math.min(path.length - 1, Math.round(seg(currentT, 1, 11) * STEPS3))]); return v > 1e4 ? 'bùng nổ' : vn(v, 1); })(), tone: (['cyan', 'lime', 'amber', 'magenta'] as const)[i] })),
  table: () => ({ headers: ['η', 'MSE sau 10 bước', 'sau 30 bước', 'sau 60 bước', 'Bước tới 1% tối thiểu'], rows: rates().map(lr => { const p = raw.descend(START_RAW, lr, STEPS3); const L = (k: number) => (p[k] ? (raw.loss(p[k]) > 1e6 ? 'bùng nổ' : vn(raw.loss(p[k]), 2)) : 'bùng nổ'); const s = raw.stepsTo(START_RAW, lr); return [lrText(lr), L(10), L(30), L(60), Number.isFinite(s) ? s : 'không bao giờ']; }) }),
};

// ---------- Chapter 4: standardize ----------
const PATH_RAW4 = raw.descend(START_RAW, 0.09, 40), PATH_STD4 = std.descend(START_STD, 0.3, 40);
const STEPS_RAW = raw.stepsTo(START_RAW, 0.09), STEPS_STD = std.stepsTo(START_STD, 0.3);

const chapter4: Scene = {
  title: 'Chuẩn hóa đặc trưng',
  duration: 12,
  beats: () => [
    { at: 0, text: 'Cùng 392 xe, cùng đường thẳng xuất phát. Bên trái x tính bằng nghìn pound, bên phải x đã chuẩn hóa: trừ trung bình, chia độ lệch chuẩn.' },
    { at: 3, text: `Chuẩn hóa biến thung lũng hẹp (số điều kiện ${vn(raw.condition, 0)}) thành cái bát tròn (số điều kiện ${vn(std.condition, 0)}).` },
    { at: 6, text: `Bát tròn cho phép bước dài hơn 10 lần. Gradient chỉ thẳng về đáy: ${STEPS_STD} bước so với ${STEPS_RAW} bước.` },
    { at: 9.5, text: 'Hai bên tìm ra cùng một đường thẳng. Chuẩn hóa không đổi mô hình, chỉ đổi hình dạng mặt loss.' },
  ],
  formula: () => `z = (x − ${vn(WEIGHT_MEAN, 2)}) / ${vn(WEIGHT_SD, 3)}\nλ (thô) = ${vn(raw.curvatures[0], 2)} và ${vn(raw.curvatures[1], 3)};  λ (chuẩn hóa) = 2 và 2`,
  render(f) {
    const { g, t, narrow } = f;
    currentT = t;
    const [left, right] = split(f.safe, [1, 1], narrow);
    const k = seg(t, 1.5, 10.5) * 40;
    const a = mapRaw(f, left, `x thô · η = 0,09 · cần ${STEPS_RAW} bước`, ['w', 'b']);
    trail(a, PATH_RAW4, k, C.magenta);
    const b = lossMap(f, right, std, VIEW_STD, zStd, `x chuẩn hóa · η = 0,3 · cần ${STEPS_STD} bước`, ['w', 'b'], true);
    trail(b, PATH_STD4, k, C.lime);
    for (const [plot, problem, path] of [[a, raw, PATH_RAW4], [b, std, PATH_STD4]] as const) {
      const v = problem.loss(at(path, k));
      g.tag(`MSE ${vn(v, 1)}`, plot.box.x + plot.box.w - 8, plot.box.y + 14, v < problem.minLoss * 1.01 ? C.lime : C.amber, 'right');
    }
  },
  metrics: () => [
    { label: 'Số điều kiện (thô)', value: vn(raw.condition, 0), tone: 'magenta' }, { label: 'Số điều kiện (chuẩn hóa)', value: vn(std.condition, 0), tone: 'lime' },
    { label: 'Bước tới 1% tối thiểu', value: `${STEPS_RAW} → ${STEPS_STD}`, tone: 'amber' },
  ],
  table: () => ({ headers: ['', 'x thô (nghìn lb)', 'x chuẩn hóa'], rows: [
    ['λ_max', vn(raw.curvatures[0], 3), vn(std.curvatures[0], 3)], ['λ_min', vn(raw.curvatures[1], 4), vn(std.curvatures[1], 3)], ['Số điều kiện', vn(raw.condition, 1), vn(std.condition, 1)],
    ['η tối đa (2/λ_max)', vn(raw.maxStableLr, 4), vn(std.maxStableLr, 3)], ['η dùng ở đây', '0,09', '0,3'], ['Bước tới MSE ≤ 1,01 × tối thiểu', STEPS_RAW, STEPS_STD],
    ['Nghiệm (đổi về pound)', `${vn(raw.optimum[1], 2)} ${vn(raw.optimum[0], 3)}·x`, `${vn(stdToRaw(std.optimum)[1], 2)} ${vn(stdToRaw(std.optimum)[0], 3)}·x`],
  ] }),
};

export const initGradientDescentLab = () => glassLab('gradient-descent', [chapter1, chapter2, chapter3, chapter4], lab => {
  const lr = lab.range('lr', value => vn(value, 3), value => { state.lr = value; lab.replay(); });
  lab.on('[data-lr]', 'click', element => { lr.set(Number(element.dataset.lr)); state.lr = lr.value; lab.replay(); });
});
