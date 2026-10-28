// Bias–variance lab on Auto MPG: fit polynomials to 30 random cars, redo it on many random subsets (curves and a
// dartboard of predictions), decompose the error, then watch variance fall as the training set grows.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, Plot, alpha, clamp, seg, split, vn, type Point, type Rect } from '../../_shared/glass-lab/gfx';
import { categoryChart } from '../../_shared/glass-lab/views';
import { DEGREES, HP, MPG, N, NOISE, SIZES, decompose, fit, mse, predict, reference, subset, type Decomposition } from './model';

export const explainerKind = 'educational-simulation';

const state = { degree: null as number | null, run: 0, nIndex: null as number | null };
let currentT = 0;
const cache = new Map<string, Decomposition>();
const row = (degree: number, n: number) => { const key = `${degree}:${n}`; if (!cache.has(key)) cache.set(key, decompose(degree, n)); return cache.get(key)!; };
const rows = (n: number) => DEGREES.map(d => row(d, n));
const fits = new Map<string, number[]>();
/** Fitted coefficients for subset `run` of size n, memoised because every frame redraws the same models. */
const model = (n: number, run: number, degree: number) => { const key = `${n}:${run}:${degree}`; let c = fits.get(key); if (!c) { c = fit(subset(n, run).train, degree); fits.set(key, c); } return c; };
const GRID = Array.from({ length: 161 }, (_, i) => 40 + i * 1.25);
const log10 = (v: number) => Math.log10(Math.max(v, 1e-9));
const sup = (n: number) => String(n).replace('-', '⁻').replace(/\d/g, d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(d)]);
const big = (v: number) => (v >= 1e4 ? `${vn(v / 10 ** Math.floor(Math.log10(v)), 1)} × 10${sup(Math.floor(Math.log10(v)))}` : vn(v, v >= 100 ? 0 : 1));
const logTick = (v: number) => (Number.isInteger(v) ? (v <= 2 ? vn(10 ** v, v < 0 ? -v : 0) : `10${sup(v)}`) : '');

function carsPlot(f: Frame, rect: Rect, title: string) {
  f.g.panel(rect.x, rect.y, rect.w, rect.h, title, C.cyan);
  const plot = new Plot(f.g, rect, [40, 240], [5, 48], { l: 42, r: 14, t: 40, b: 34 });
  plot.axes({ xTicks: [50, 100, 150, 200], yTicks: [10, 20, 30, 40], xLabel: 'mã lực', yLabel: 'mpg', fx: v => vn(v, 0), fy: v => vn(v, 0) });
  return plot;
}
function allCars(plot: Plot, highlight: readonly number[] = []) {
  const { ctx } = plot.g;
  ctx.save(); ctx.fillStyle = alpha(C.cyan, 0.2);
  for (let i = 0; i < N; i++) ctx.fillRect(plot.px(HP[i]) - 1.4, plot.py(MPG[i]) - 1.4, 2.8, 2.8);
  ctx.restore();
  for (const i of highlight) plot.g.circle(plot.px(HP[i]), plot.py(MPG[i]), 4.2, C.amber, { glow: 12 });
}
const curve = (coef: number[]) => GRID.map(h => [h, predict(coef, h)] as Point);
const refCurve = GRID.map(h => [h, reference(h)] as Point);

// ---------- Chapter 1 ----------
const timelineDegree = (t: number) => (t < 3.5 ? 1 : t < 7 ? 2 : 8);
const degree1 = () => state.degree ?? timelineDegree(currentT);
const chapter1: Scene = {
  title: 'Underfit và overfit trên 30 xe',
  duration: 11,
  beats: () => [
    { at: 0, text: 'Chỉ có 30 xe huấn luyện (chấm cam) lấy ngẫu nhiên từ 392 xe. 362 xe còn lại dùng để kiểm tra.' },
    { at: 1.5, text: 'Đường thẳng (bậc 1) không theo được độ cong: sai nhiều cả trên xe huấn luyện. Đây là underfitting.' },
    { at: 3.5, text: 'Bậc 2 bám theo hình dạng chung và cho lỗi kiểm tra thấp nhất.' },
    { at: 7, text: 'Bậc 8 uốn qua sát từng xe huấn luyện, lỗi huấn luyện giảm thêm, nhưng vọt lên ở hai đầu nơi ít xe: lỗi kiểm tra tăng gấp nhiều lần. Đây là overfitting.' },
  ],
  formula: () => `mpg ≈ w₀ + w₁·hp + … + w_d·hp^d,  d = ${degree1()}\nlỗi kiểm tra đo trên ${N - 30} xe không dùng để huấn luyện`,
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.3, 1] : [1.45, 1], narrow);
    const d = degree1(), { train, test } = subset(30, state.run), coef = fit(train, d);
    const plot = carsPlot(f, left, `Đa thức bậc ${d} học từ 30 xe`);
    plot.clip(() => {
      allCars(plot, train);
      for (const i of train) plot.line([[HP[i], MPG[i]], [HP[i], predict(coef, HP[i])]], alpha(C.magenta, 0.6), { width: 1.2 });
      plot.line(curve(coef), C.amber, { width: 3, glow: 16 });
    });
    const tr = mse(coef, train), te = mse(coef, test);
    g.tag(`lỗi huấn luyện ${big(tr)}`, plot.box.x + plot.box.w - 8, plot.box.y + 14, C.amber, 'right');
    g.tag(`lỗi kiểm tra ${big(te)}`, plot.box.x + plot.box.w - 8, plot.box.y + 40, te > 2 * tr ? C.red : C.lime, 'right');
    const data = rows(30);
    categoryChart(f, right, 'Trung bình 150 lần lấy mẫu (thang log)', DEGREES.map(String), [
      { label: 'lỗi kiểm tra', values: data.map(r => log10(r.test)), color: C.red, format: v => big(10 ** v) },
      { label: 'lỗi huấn luyện', values: data.map(r => log10(r.train)), color: C.amber, format: v => big(10 ** v) },
    ], [1, 7], d - 1, 1, logTick, 'bậc đa thức');
  },
  metrics() { const { train, test } = subset(30, state.run), coef = fit(train, degree1()); return [{ label: 'Bậc', value: String(degree1()), tone: 'violet' }, { label: 'Lỗi huấn luyện', value: big(mse(coef, train)), tone: 'amber' }, { label: 'Lỗi kiểm tra', value: big(mse(coef, test)), tone: 'red' }]; },
  table: () => ({ headers: ['Bậc', 'Lỗi huấn luyện TB', 'Lỗi kiểm tra TB'], rows: rows(30).map(r => [r.degree, vn(r.train, 2), big(r.test)]), caption: 'Trung bình trên 150 tập 30 xe lấy ngẫu nhiên; kiểm tra trên 362 xe còn lại.' }),
};

// ---------- Chapter 2 ----------
const PROBES = [95, 180] as const;
const chapter2: Scene = {
  title: 'Nhiều tập huấn luyện: bias và variance',
  duration: 12,
  beats: () => [
    { at: 0, text: 'Lặp lại: mỗi lần lấy 30 xe khác nhau rồi khớp lại cùng bậc đa thức. Mỗi đường mờ là một lần.' },
    { at: 3.5, text: 'Đường đen là trung bình các mô hình. Nét đứt xanh lá là đường làm mượt trên cả 392 xe, ta coi như “sự thật”.' },
    { at: 6.5, text: `Bảng bắn bia bên phải: mỗi chấm là sai số của một mô hình tại ${PROBES[0]} và ${PROBES[1]} mã lực. Tâm cụm lệch khỏi hồng tâm là bias, độ tản của cụm là variance.` },
    { at: 9.5, text: 'Bậc thấp: cụm chụm nhưng lệch tâm. Bậc cao: cụm quanh tâm nhưng tản tung tóe. Kéo thanh bậc đa thức để so sánh.' },
  ],
  formula: 'bias(x) = E[ŷ(x)] − f(x)    variance(x) = E[(ŷ(x) − E[ŷ(x)])²]\nE lấy trên các tập huấn luyện 30 xe có thể có',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.2, 1] : [1.4, 1], narrow);
    const d = state.degree ?? 2;
    const plot = carsPlot(f, left, `25 mô hình bậc ${d}, mỗi mô hình từ 30 xe khác nhau`);
    const shown = Math.floor(seg(t, 0.3, 3.5) * 25);
    const mean = GRID.map(() => 0);
    const models = Array.from({ length: 25 }, (_, r) => model(30, r, d));
    plot.clip(() => {
      allCars(plot);
      models.forEach((coef, r) => { if (r < shown) plot.line(curve(coef), alpha(C.amber, 0.38), { width: 1.2 }); });
      models.forEach(coef => GRID.forEach((h, i) => { mean[i] += predict(coef, h) / models.length; }));
      if (t > 3.5) {
        plot.line(refCurve, C.lime, { width: 2.6, dash: [7, 5], glow: 8 });
        plot.line(GRID.map((h, i) => [h, mean[i]] as Point), C.white, { width: 3, glow: 10 });
      }
      for (const h of PROBES) plot.line([[h, 5], [h, 48]], alpha(C.violet, 0.5), { width: 1, dash: [3, 4] });
    });
    dartboard(f, right, d, seg(t, 6.5, 9));
  },
  metrics() { const r = row(state.degree ?? 2, 30); return [{ label: 'Bậc', value: String(state.degree ?? 2), tone: 'violet' }, { label: 'bias²', value: big(r.bias2), tone: 'lime' }, { label: 'variance', value: big(r.variance), tone: 'amber' }]; },
  table: () => ({ headers: ['Bậc', 'bias²', 'variance'], rows: rows(30).map(r => [r.degree, big(r.bias2), big(r.variance)]) }),
};

function dartboard(f: Frame, rect: Rect, degree: number, reveal: number) {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, 'Bắn bia: sai số tại hai mức mã lực', C.magenta);
  const size = Math.min(rect.w - 40, rect.h - 70), cx = rect.x + rect.w / 2, cy = rect.y + 40 + size / 2, R = size / 2, scale = R / 12;
  [1, 2 / 3, 1 / 3].forEach((k, i) => g.circle(cx, cy, R * k, i === 2 ? alpha(C.magenta, 0.12) : null, { stroke: alpha(C.white, 0.18), width: 1 }));
  g.line([[cx - R, cy], [cx + R, cy]], alpha(C.white, 0.15), { width: 1 }); g.line([[cx, cy - R], [cx, cy + R]], alpha(C.white, 0.15), { width: 1 });
  const ref = PROBES.map(reference);
  let sx = 0, sy = 0;
  const count = 150, shown = Math.floor(count * reveal);
  for (let r = 0; r < count; r++) {
    const coef = model(30, r, degree);
    const ex = predict(coef, PROBES[0]) - ref[0], ey = predict(coef, PROBES[1]) - ref[1];
    sx += ex / count; sy += ey / count;
    if (r < shown) g.circle(cx + clamp(ex * scale, -R, R), cy - clamp(ey * scale, -R, R), 2.6, C.amber, { alpha: 0.75 });
  }
  if (reveal > 0.95) {
    g.line([[cx, cy], [cx + clamp(sx * scale, -R, R), cy - clamp(sy * scale, -R, R)]], C.lime, { width: 2, glow: 10 });
    g.circle(cx + clamp(sx * scale, -R, R), cy - clamp(sy * scale, -R, R), 6, null, { stroke: C.lime, width: 2.5 });
  }
  g.text(`ngang: sai số tại ${PROBES[0]} hp · dọc: tại ${PROBES[1]} hp · vòng ngoài = ±12 mpg`, rect.x + rect.w / 2, rect.y + rect.h - 16, { size: 11, color: C.muted, align: 'center' });
}

// ---------- Chapter 3 ----------
const chapter3: Scene = {
  title: 'Phân rã lỗi',
  duration: 11,
  beats: () => {
    const r = rows(30), best = r.reduce((a, b) => (b.test < a.test ? b : a));
    return [
      { at: 0, text: 'Lỗi trên xe mới = bias² + variance + nhiễu. Nhiễu (khoảng 18 mpg²) là phần không mô hình nào giảm được.' },
      { at: 3, text: `Bậc 1: bias² ≈ ${vn(r[0].bias2, 1)} vì đường thẳng không cong được. Từ bậc 2, bias² gần như bằng 0.` },
      { at: 6, text: `Variance tăng theo bậc: ${vn(r[1].variance, 1)} ở bậc 2, ${big(r[4].variance)} ở bậc 5. Với 30 xe, mô hình linh hoạt dao động dữ dội ở vùng ít xe.` },
      { at: 8.5, text: `Tổng nhỏ nhất ở bậc ${best.degree}. Đường lỗi ước lượng (bias² + variance + nhiễu) khớp với lỗi đo trực tiếp trên xe kiểm tra.` },
    ];
  },
  formula: () => `E[(y − ŷ)²] = bias² + variance + σ²,  σ² ≈ ${vn(NOISE, 1)} (phần dư của đường làm mượt)`,
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.1, 1] : [1.2, 1], narrow);
    const d = state.degree ?? 3, n = 30;
    const plot = carsPlot(f, left, `Trung bình ± độ lệch chuẩn của 150 mô hình bậc ${d}`);
    const models = Array.from({ length: 150 }, (_, r) => model(n, r, d));
    const preds = GRID.map(h => models.map(c => predict(c, h)));
    const mean = preds.map(p => p.reduce((a, b) => a + b, 0) / p.length), sd = preds.map((p, i) => Math.sqrt(p.reduce((a, b) => a + (b - mean[i]) ** 2, 0) / p.length));
    plot.clip(() => {
      allCars(plot);
      const k = seg(t, 0.3, 2.5);
      g.polygon([...GRID.map((h, i) => plot.p([h, mean[i] + sd[i] * k])), ...GRID.map((h, i) => plot.p([h, mean[i] - sd[i] * k])).reverse()], alpha(C.amber, 0.22));
      plot.line(refCurve, C.lime, { width: 2.6, dash: [7, 5], glow: 8 });
      plot.line(GRID.map((h, i) => [h, mean[i]] as Point), C.white, { width: 3, glow: 10 });
    });
    const r = rows(30), reveal = seg(t, 2.5, 9);
    categoryChart(f, right, 'Theo bậc: thành phần của lỗi (thang log)', DEGREES.map(String), [
      { label: 'lỗi đo trên xe kiểm tra', values: r.map(x => log10(x.test)), color: C.red, format: v => big(10 ** v) },
      { label: 'bias² + variance + σ²', values: r.map(x => log10(x.expected)), color: C.white, format: v => big(10 ** v) },
      { label: 'variance', values: r.map(x => log10(x.variance)), color: C.amber, format: v => big(10 ** v) },
      { label: 'bias²', values: r.map(x => log10(x.bias2)), color: C.lime, format: v => big(10 ** v) },
    ], [-1, 7], d - 1, reveal, logTick, 'bậc đa thức');
  },
  metrics() { const r = row(state.degree ?? 3, 30); return [{ label: 'bias²', value: big(r.bias2), tone: 'lime' }, { label: 'variance', value: big(r.variance), tone: 'amber' }, { label: 'σ²', value: vn(NOISE, 1), tone: 'violet' }, { label: 'Lỗi kiểm tra', value: big(r.test), tone: 'red' }]; },
  table: () => ({ headers: ['Bậc', 'bias²', 'variance', 'σ²', 'Tổng ước lượng', 'Lỗi kiểm tra đo được'], rows: rows(30).map(r => [r.degree, big(r.bias2), big(r.variance), vn(NOISE, 2), big(r.expected), big(r.test)]), caption: 'n = 30 xe, trung bình 150 lần lấy mẫu.' }),
};

// ---------- Chapter 4 ----------
const nNow = () => SIZES[state.nIndex ?? Math.min(SIZES.length - 1, Math.floor(seg(currentT, 0.8, 9.5) * SIZES.length))];
const chapter4: Scene = {
  title: 'Thêm dữ liệu',
  duration: 11.5,
  beats: () => [
    { at: 0, text: 'Giữ nguyên bậc đa thức, tăng số xe huấn luyện từ 15 lên 300.' },
    { at: 3, text: 'Dải dao động co lại rất nhanh: mỗi xe nhiễu ảnh hưởng ít hơn tới đường cong. Bias gần như không đổi.' },
    { at: 6.5, text: 'Biểu đồ bên phải: variance của mọi bậc đều giảm khi n tăng, bậc càng cao giảm càng mạnh.' },
    { at: 9, text: 'Vì thế “phức tạp” là khái niệm tương đối với lượng dữ liệu: bậc 5 overfit với 30 xe nhưng ổn với 300 xe.' },
  ],
  formula: () => `n = ${nNow()} xe, bậc ${state.degree ?? 5}\nvariance ≈ σ² · (số tham số) / n  khi n đủ lớn`,
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.1, 1] : [1.25, 1], narrow);
    const d = state.degree ?? 5, n = nNow();
    const plot = carsPlot(f, left, `25 mô hình bậc ${d}, mỗi mô hình từ ${n} xe`);
    plot.clip(() => {
      allCars(plot);
      for (let r = 0; r < 25; r++) plot.line(curve(model(n, r, d)), alpha(C.amber, 0.35), { width: 1.2 });
      plot.line(refCurve, C.lime, { width: 2.6, dash: [7, 5], glow: 8 });
    });
    g.tag(`variance ${big(row(d, n).variance)}`, plot.box.x + plot.box.w - 8, plot.box.y + 14, C.amber, 'right');
    const show = [1, 2, 5, 8];
    const colors = [C.lime, C.cyan, C.amber, C.magenta];
    categoryChart(f, right, 'Variance theo số xe huấn luyện (thang log)', SIZES.map(String), show.map((deg, k) => ({ label: `bậc ${deg}`, values: SIZES.map(s => log10(row(deg, s).variance)), color: colors[k], format: v => big(10 ** v) })), [-2, 7], SIZES.indexOf(n as typeof SIZES[number]), 1, logTick, 'số xe huấn luyện n');
  },
  metrics() { const r = row(state.degree ?? 5, nNow()); return [{ label: 'n', value: String(nNow()), tone: 'violet' }, { label: 'bias²', value: big(r.bias2), tone: 'lime' }, { label: 'variance', value: big(r.variance), tone: 'amber' }, { label: 'Lỗi kiểm tra', value: big(r.test), tone: 'red' }]; },
  table: () => ({ headers: ['n', 'bậc 1', 'bậc 2', 'bậc 5', 'bậc 8'], rows: SIZES.map(s => [s, ...[1, 2, 5, 8].map(d => big(row(d, s).test))]), caption: 'Lỗi kiểm tra trung bình (mpg²) theo số xe huấn luyện.' }),
};

export const initBiasVarianceLab = () => glassLab('bias-variance', [chapter1, chapter2, chapter3, chapter4], lab => {
  lab.range('degree', value => String(value), value => { state.degree = value; });
  lab.range('n', value => `${SIZES[value]} xe`, value => { state.nIndex = value; });
  lab.on('[data-resample]', 'click', () => { state.run = (state.run + 1) % 150; });
  lab.on('[data-autoplay]', 'click', () => { state.degree = null; state.nIndex = null; lab.replay(); });
});
