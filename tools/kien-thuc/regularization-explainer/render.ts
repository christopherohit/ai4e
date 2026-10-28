// L1 / L2 lab on the Diabetes data: the diamond and the circle on a real two-feature loss, shrink vs soft-threshold,
// the famous Lasso path with 10-fold cross-validation, and the bootstrap instability of two correlated cholesterol
// measures.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, Plot, alpha, clamp, fitEqual, lerp, seg, split, vn, type Point, type Rect } from '../../_shared/glass-lab/gfx';
import { ticks } from '../../_shared/glass-lab/views';
import { FULL, LAMBDAS, SHORT, bootstrap, crossValidate, design, lasso, lassoPath, ols, ridge, ridgePath, softThreshold, ALL } from './model';

export const explainerKind = 'educational-simulation';

const state = { lam2d: null as number | null, shrink: null as number | null, index: null as number | null };
let currentT = 0;
const COLORS = [C.muted, C.violet, C.amber, C.lime, C.magenta, C.red, C.cyan, '#8fa3ff', '#ffd27a', '#6fe7d2'];

// ---------- Chapter 1: geometry on BMI and blood pressure ----------
const PAIR = design(ALL, [2, 3]);
const L2_MAX = 45;
const loss2 = ([a, b]: Point) => 0.5 * (PAIR.gram[0][0] * a * a + 2 * PAIR.gram[0][1] * a * b + PAIR.gram[1][1] * b * b) - PAIR.xty[0] * a - PAIR.xty[1] * b;
const OLS2 = ols(PAIR) as unknown as Point;
const lam2 = () => state.lam2d ?? lerp(0, 32, seg(currentT, 0.8, 10));
const TRACE_RIDGE = Array.from({ length: 2 * L2_MAX + 1 }, (_, i) => ridge(PAIR, i / 2) as unknown as Point);
const TRACE_LASSO = Array.from({ length: 2 * L2_MAX + 1 }, (_, i) => lasso(PAIR, i / 2) as unknown as Point);

function geometry(f: Frame, rect: Rect, kind: 'l2' | 'l1') {
  const { g } = f, lambda = lam2();
  const beta = (kind === 'l2' ? ridge(PAIR, lambda) : lasso(PAIR, lambda)) as unknown as Point;
  g.panel(rect.x, rect.y, rect.w, rect.h, kind === 'l2' ? 'L2 · Ridge: vùng giới hạn hình tròn' : 'L1 · Lasso: vùng giới hạn hình thoi', kind === 'l2' ? C.cyan : C.amber);
  const pad = { l: 44, r: 14, t: 40, b: 34 }, [xr, yr] = fitEqual(rect, pad, [-12, 46], [-12, 34]);
  const plot = new Plot(g, rect, xr, yr, pad);
  const base = loss2(OLS2);
  plot.heat((a, b) => loss2([a, b]) - base, v => { const k = 1 - clamp(Math.log1p(v / 8) / Math.log1p(900 / 8)); return [14 + 40 * k * k, 18 + 70 * k * k, 44 + 120 * k * k, 255]; }, 70);
  plot.contours((a, b) => loss2([a, b]) - base, [5, 30, 90, 200, 400, 700], () => alpha(C.violet, 0.45), { width: 1 });
  plot.axes({ xTicks: ticks(xr), yTicks: ticks(yr), xLabel: 'β BMI', yLabel: 'β huyết áp', fx: v => vn(v, 0), fy: v => vn(v, 0) });
  plot.line([[xr[0], 0], [xr[1], 0]], alpha(C.white, 0.35), { width: 1 });
  plot.line([[0, yr[0]], [0, yr[1]]], alpha(C.white, 0.35), { width: 1 });
  const touch = loss2(beta) - base;
  plot.clip(() => {
    if (touch > 0.01) plot.contours((a, b) => loss2([a, b]) - base, [touch], () => C.white, { width: 2, glow: 10 });
    const t = kind === 'l2' ? Math.hypot(beta[0], beta[1]) : Math.abs(beta[0]) + Math.abs(beta[1]);
    const color = kind === 'l2' ? C.cyan : C.amber;
    const region: Point[] = kind === 'l2' ? Array.from({ length: 97 }, (_, i) => [t * Math.cos(i / 96 * 2 * Math.PI), t * Math.sin(i / 96 * 2 * Math.PI)]) : [[t, 0], [0, t], [-t, 0], [0, -t], [t, 0]];
    g.polygon(region.map(plot.p), alpha(color, 0.16), color, 2.2);
    plot.line(kind === 'l2' ? TRACE_RIDGE : TRACE_LASSO, alpha(color, 0.7), { width: 1.4, dash: [4, 4] });
  });
  const [ox, oy] = plot.p(OLS2);
  g.circle(ox, oy, 5, null, { stroke: C.white, width: 2 });
  g.text('OLS', ox + 8, oy - 8, { size: 11, color: C.ink });
  const [bx, by] = plot.p(beta);
  g.circle(bx, by, 7, kind === 'l2' ? C.cyan : C.amber, { glow: 20 });
  g.tag(`(${vn(beta[0], 1)}; ${vn(beta[1], 1)})`, bx + 12, by - 16, kind === 'l2' ? C.cyan : C.amber, 'left');
  return beta;
}

const chapter1: Scene = {
  title: 'Kim cương và hình tròn',
  duration: 11.5,
  beats: () => [
    { at: 0, text: 'Mô hình chỉ dùng hai chỉ số của 442 bệnh nhân: BMI và huyết áp. Elip tím là đường đồng mức của loss; tâm là nghiệm OLS.' },
    { at: 2.5, text: 'Phạt tương đương với việc nhốt hệ số trong một vùng: hình tròn với L2, hình thoi với L1. λ càng lớn, vùng càng nhỏ.' },
    { at: 5.5, text: 'Lời giải là chỗ elip đầu tiên chạm vùng giới hạn (elip trắng). Hình tròn trơn đều nên điểm chạm hiếm khi nằm trên trục.' },
    { at: 8, text: 'Hình thoi có đỉnh nhọn trên trục: từ λ ≈ 26,5, điểm chạm rơi đúng vào đỉnh và β huyết áp bằng 0 tuyệt đối. Ridge chỉ làm nó nhỏ dần.' },
  ],
  formula: () => `λ = ${vn(lam2(), 1)}\nRidge: min (1/2n)‖y − Xβ‖² + (λ/2)‖β‖²     Lasso: min (1/2n)‖y − Xβ‖² + λ‖β‖₁`,
  render(f) {
    currentT = f.t;
    const [left, right] = split(f.safe, [1, 1], f.narrow);
    geometry(f, left, 'l2'); geometry(f, right, 'l1');
  },
  metrics() {
    const r = ridge(PAIR, lam2()), l = lasso(PAIR, lam2());
    return [{ label: 'λ', value: vn(lam2(), 1), tone: 'violet' }, { label: 'Ridge β huyết áp', value: vn(r[1], 2), tone: 'cyan' }, { label: 'Lasso β huyết áp', value: vn(l[1], 2), tone: l[1] === 0 ? 'lime' : 'amber' }];
  },
  table: () => ({ headers: ['λ', 'Ridge β BMI', 'Ridge β HA', 'Lasso β BMI', 'Lasso β HA'], rows: [0, 5, 10, 20, 26.5, 30, 40, 45].map(l => { const r = ridge(PAIR, l), s = lasso(PAIR, l); return [vn(l, 1), vn(r[0], 2), vn(r[1], 2), vn(s[0], 2), vn(s[1], 2)]; }), caption: 'Hệ số trên đặc trưng đã chuẩn hóa (đơn vị: điểm tiến triển bệnh mỗi độ lệch chuẩn).' }),
};

// ---------- Chapter 2: shrink vs soft-threshold ----------
const shrinkNow = () => state.shrink ?? lerp(0, 1.6, seg(currentT, 0.8, 8));
const chapter2: Scene = {
  title: 'Chia hay trừ',
  duration: 10.5,
  beats: () => [
    { at: 0, text: 'Khi các đặc trưng trực giao, cả hai có công thức đóng theo hệ số bình phương tối thiểu β̂.' },
    { at: 2.5, text: 'Ridge chia mọi hệ số cho (1 + λ): co theo tỉ lệ, không hệ số nào về đúng 0.' },
    { at: 5, text: 'Lasso trừ λ rồi cắt ở 0: mọi hệ số có |β̂| ≤ λ bị xóa hẳn (vùng giữa hai vạch đứng).' },
    { at: 7.5, text: 'Lý do nằm ở đạo hàm của phạt: L2 kéo về 0 với lực λβ, yếu dần gần 0; L1 kéo với lực λ không đổi, đủ để ghim hệ số tại 0.' },
  ],
  formula: () => `λ = ${vn(shrinkNow(), 2)}\nRidge: β = β̂ / (1 + λ)     Lasso: β = sign(β̂) · max(|β̂| − λ, 0)`,
  render(f) {
    const { g, t } = f; currentT = t;
    const lambda = shrinkNow();
    const [left, right] = split(f.safe, [1, 1], f.narrow);
    g.panel(left.x, left.y, left.w, left.h, 'Hệ số sau khi phạt theo β̂', C.amber);
    const a = new Plot(g, left, [-3, 3], [-3, 3]);
    a.axes({ xTicks: [-3, -2, -1, 0, 1, 2, 3], yTicks: [-3, -2, -1, 0, 1, 2, 3], xLabel: 'β̂ (bình phương tối thiểu)', yLabel: 'β sau khi phạt', fx: v => vn(v, 0), fy: v => vn(v, 0) });
    a.clip(() => {
      if (lambda > 0) g.polygon([a.p([-lambda, -3]), a.p([lambda, -3]), a.p([lambda, 3]), a.p([-lambda, 3])], alpha(C.amber, 0.08));
      a.line([[-3, -3], [3, 3]], alpha(C.white, 0.35), { width: 1.2, dash: [5, 5] });
      a.curve(b => b / (1 + lambda), C.cyan, { width: 3, glow: 12 });
      a.curve(b => softThreshold(b, lambda), C.amber, { width: 3, glow: 12 }, 240);
      for (const b of [-2.4, -0.6, 0.4, 1.3, 2.6]) {
        g.circle(a.px(b), a.py(b / (1 + lambda)), 4.5, C.cyan, { glow: 10 });
        g.circle(a.px(b), a.py(softThreshold(b, lambda)), 4.5, C.amber, { glow: 10 });
      }
    });
    g.tag('Ridge: chia', a.box.x + 8, a.box.y + 14, C.cyan, 'left');
    g.tag('Lasso: trừ rồi cắt', a.box.x + 8, a.box.y + 40, C.amber, 'left');
    g.panel(right.x, right.y, right.w, right.h, 'Lực kéo về 0 (đạo hàm của phạt)', C.violet);
    const b = new Plot(g, right, [-3, 3], [-2, 2]);
    b.axes({ xTicks: [-3, -2, -1, 0, 1, 2, 3], yTicks: [-2, -1, 0, 1, 2], xLabel: 'β', fx: v => vn(v, 0), fy: v => vn(v, 0) });
    b.clip(() => {
      b.line([[-3, 0], [3, 0]], alpha(C.white, 0.3), { width: 1 });
      b.curve(x => lambda * x, C.cyan, { width: 3, glow: 12 });
      b.line([[-3, -lambda], [0, -lambda]], C.amber, { width: 3, glow: 12 });
      b.line([[0, lambda], [3, lambda]], C.amber, { width: 3, glow: 12 });
    });
    g.tag('L2: λβ', b.box.x + b.box.w - 8, b.box.y + 14, C.cyan, 'right');
    g.tag('L1: λ · sign(β)', b.box.x + b.box.w - 8, b.box.y + 40, C.amber, 'right');
  },
  metrics: () => [{ label: 'λ', value: vn(shrinkNow(), 2), tone: 'violet' }, { label: 'β̂ = 0,4 → Ridge', value: vn(0.4 / (1 + shrinkNow()), 3), tone: 'cyan' }, { label: 'β̂ = 0,4 → Lasso', value: vn(softThreshold(0.4, shrinkNow()), 3), tone: 'amber' }],
  table: () => ({ headers: ['β̂', 'Ridge', 'Lasso'], rows: [-2.4, -0.6, 0.4, 1.3, 2.6].map(b => [vn(b, 1), vn(b / (1 + shrinkNow()), 3), vn(softThreshold(b, shrinkNow()), 3)]) }),
};

// ---------- Chapter 3: the paths ----------
const LASSO_PATH = lassoPath(FULL), RIDGE_PATH = ridgePath(FULL);
let cv: ReturnType<typeof crossValidate> | null = null;
const CV = () => (cv ??= crossValidate());
const indexNow = () => state.index ?? Math.round(lerp(LAMBDAS.length - 1, 0, seg(currentT, 1, 10.5)));
const nonZero = (beta: number[]) => beta.filter(v => Math.abs(v) > 1e-9).length;
const best = (values: number[]) => values.indexOf(Math.min(...values));

function pathPanel(f: Frame, rect: Rect, title: string, path: number[][], color: string, index: number) {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, title, color);
  const plot = new Plot(g, rect, [-2, 2], [-42, 42], { l: 44, r: 52, t: 40, b: 30 });
  plot.axes({ xTicks: [-2, -1, 0, 1, 2], yTicks: [-40, -20, 0, 20, 40], xLabel: 'log₁₀ λ  (phạt mạnh dần →)', fx: v => vn(v, 0), fy: v => vn(v, 0) });
  plot.line([[-2, 0], [2, 0]], alpha(C.white, 0.3), { width: 1 });
  const x = Math.log10(LAMBDAS[index]);
  plot.clip(() => {
    for (let j = 0; j < 10; j++) {
      const pts = LAMBDAS.map((l, i) => [Math.log10(l), path[i][j]] as Point).filter((_, i) => i >= index);
      plot.line(pts, COLORS[j], { width: j === 4 || j === 5 ? 2.6 : 1.8, glow: 6 });
      g.circle(plot.px(x), plot.py(path[index][j]), 3.5, COLORS[j], { glow: 8 });
    }
    plot.line([[x, -42], [x, 42]], alpha(C.white, 0.5), { width: 1, dash: [4, 4] });
  });
  // Direct labels at the cursor for coefficients that are clearly non-zero.
  const placed: number[] = [];
  [...Array(10).keys()].filter(j => Math.abs(path[index][j]) > 1.5).sort((a, b) => path[index][b] - path[index][a]).forEach(j => {
    let y = plot.py(path[index][j]);
    while (placed.some(p => Math.abs(p - y) < 12)) y += 12;
    placed.push(y);
    g.text(SHORT[j], plot.box.x + plot.box.w + 6, y, { size: 11, color: COLORS[j], weight: 650 });
  });
}

const chapter3: Scene = {
  title: 'Đường đi hệ số',
  duration: 12,
  beats: () => {
    const c = CV(), bl = best(c.lasso), br = best(c.ridge);
    return [
      { at: 0, text: 'Dùng cả 10 chỉ số. Quét λ từ rất lớn (mọi hệ số bằng 0) về rất nhỏ (gần OLS).' },
      { at: 2.5, text: 'Lasso “thả” từng biến vào mô hình theo thứ tự: BMI, s5, huyết áp, s3, giới tính… Đây là hình kinh điển trong bài báo LARS (Efron và cộng sự, 2004).' },
      { at: 6, text: 'Ridge giữ đủ 10 biến ở mọi λ, chỉ co chúng nhỏ dần đều. Chú ý s1 và s2 (nét đậm) bung ra hai phía khi phạt yếu.' },
      { at: 9, text: `Cross-validation 10 phần: OLS ${vn(c.ols, 0)}, Lasso tốt nhất ${vn(c.lasso[bl], 0)} với ${nonZero(LASSO_PATH[bl])} biến, Ridge tốt nhất ${vn(c.ridge[br], 0)}. Với 442 bệnh nhân và 10 biến, phạt hầu như không giảm lỗi mà chủ yếu làm mô hình gọn và ổn định hơn.` },
    ];
  },
  formula: () => `λ = ${vn(LAMBDAS[indexNow()], LAMBDAS[indexNow()] < 1 ? 3 : 1)}: Lasso giữ ${nonZero(LASSO_PATH[indexNow()])} / 10 biến`,
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [top, bottom] = split(f.safe, narrow ? [2.2, 1] : [1.7, 1], true, 12);
    const [a, b] = split(top, [1, 1], narrow, 12);
    const index = indexNow();
    pathPanel(f, a, 'Lasso (L1)', LASSO_PATH, C.amber, index);
    pathPanel(f, b, 'Ridge (L2)', RIDGE_PATH, C.cyan, index);
    cvPanel(f, bottom, index, t);
  },
  metrics() { const i = indexNow(), c = CV(); return [{ label: 'λ', value: vn(LAMBDAS[i], LAMBDAS[i] < 1 ? 3 : 1), tone: 'violet' }, { label: 'Biến giữ lại (Lasso)', value: `${nonZero(LASSO_PATH[i])} / 10`, tone: 'amber' }, { label: 'CV Lasso', value: vn(c.lasso[i], 0), tone: 'amber' }, { label: 'CV Ridge', value: vn(c.ridge[i], 0), tone: 'cyan' }]; },
  table: () => ({ headers: ['λ', 'Biến Lasso giữ', 'CV Lasso', 'CV Ridge', ...SHORT.map(s => `Lasso ${s}`)], rows: LAMBDAS.map((l, i) => [vn(l, 3), nonZero(LASSO_PATH[i]), vn(CV().lasso[i], 0), vn(CV().ridge[i], 0), ...LASSO_PATH[i].map(v => vn(v, 1))]).filter((_, i) => i % 4 === 0), caption: 'Hệ số trên đặc trưng chuẩn hóa; CV = MSE cross-validation 10 phần.' }),
};

function cvPanel(f: Frame, rect: Rect, index: number, t: number) {
  const { g } = f, c = CV();
  g.panel(rect.x, rect.y, rect.w, rect.h, 'MSE cross-validation 10 phần theo λ', C.lime);
  const plot = new Plot(g, rect, [-2, 2], [2800, 5800], { l: 52, r: 16, t: 36, b: 26 });
  plot.axes({ xTicks: [-2, -1, 0, 1, 2], yTicks: [3000, 4000, 5000], fx: v => vn(v, 0), fy: v => vn(v, 0) });
  const reveal = seg(t, 8.5, 10.5);
  plot.clip(() => {
    plot.line([[-2, c.ols], [2, c.ols]], alpha(C.white, 0.45), { width: 1.2, dash: [5, 4] });
    const show = (values: number[], color: string) => plot.line(LAMBDAS.map((l, i) => [Math.log10(l), values[i]] as Point).filter(([x]) => x <= lerp(-2, 2, Math.max(reveal, state.index !== null ? 1 : 0))), color, { width: 2.4, glow: 8 });
    show(c.lasso, C.amber); show(c.ridge, C.cyan);
    plot.line([[Math.log10(LAMBDAS[index]), 2800], [Math.log10(LAMBDAS[index]), 5800]], alpha(C.white, 0.4), { width: 1, dash: [3, 3] });
  });
  const bl = best(c.lasso);
  if (reveal > 0.9) g.circle(plot.px(Math.log10(LAMBDAS[bl])), plot.py(c.lasso[bl]), 5, null, { stroke: C.amber, width: 2 });
  g.text(`OLS ${vn(c.ols, 0)}`, plot.box.x + 6, plot.py(c.ols) - 9, { size: 10.5, color: C.muted });
  g.text('Lasso', plot.box.x + plot.box.w - 70, plot.box.y + 10, { size: 11, color: C.amber, weight: 650 });
  g.text('Ridge', plot.box.x + plot.box.w - 30, plot.box.y + 10, { size: 11, color: C.cyan, weight: 650 });
}

// ---------- Chapter 4: correlated features ----------
const BOOT_LAMBDA = { ridge: 0.1, lasso: 1 };
let boot: ReturnType<typeof bootstrap> | null = null;
const BOOT = () => (boot ??= bootstrap(80, BOOT_LAMBDA));
const stats = (values: number[]) => { const m = values.reduce((a, b) => a + b, 0) / values.length; return { mean: m, sd: Math.sqrt(values.reduce((a, b) => a + (b - m) ** 2, 0) / values.length) }; };

const chapter4: Scene = {
  title: 'Hai biến gần như trùng nhau',
  duration: 12,
  beats: () => {
    const b = BOOT(), s = (k: 'ols' | 'ridge' | 'lasso', j: number) => stats(b.map(r => r[k][j]));
    return [
      { at: 0, text: `s1 (cholesterol toàn phần) và s2 (LDL) tương quan 0,90. Mỗi chấm là kết quả trên một mẫu bootstrap: chọn lại 442 bệnh nhân có hoàn lại.` },
      { at: 2.5, text: `OLS (hồng): β s1 = ${vn(s('ols', 4).mean, 0)} ± ${vn(s('ols', 4).sd, 0)}, β s2 = ${vn(s('ols', 5).mean, 0)} ± ${vn(s('ols', 5).sd, 0)}. Hai hệ số lớn, trái dấu, nhảy loạn: chúng triệt tiêu nhau.` },
      { at: 5.5, text: `Ridge (λ = 0,1, xanh): cụm thu về sát gốc, độ lệch chỉ còn ±${vn(s('ridge', 4).sd, 1)}. Cùng lỗi dự đoán như OLS nhưng ổn định gấp nhiều lần.` },
      { at: 8.5, text: `Lasso (λ = 1, cam): chấm nằm trên các trục. ${Math.round(b.filter(r => r.lasso[5] === 0).length / b.length * 100)}% số lần nó bỏ hẳn s2, ${Math.round(b.filter(r => r.lasso[4] === 0).length / b.length * 100)}% bỏ s1: giữ biến nào là khá tùy ý.` },
    ];
  },
  formula: 'Elastic Net: λ₁‖β‖₁ + λ₂‖β‖² — thưa như Lasso, giữ nhóm biến tương quan như Ridge',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.3, 1] : [1.25, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, '80 mẫu bootstrap: hệ số của s1 và s2', C.magenta);
    const pad = { l: 44, r: 14, t: 40, b: 34 }, [xr, yr] = fitEqual(left, pad, [-90, 25], [-35, 70]);
    const plot = new Plot(g, left, xr, yr, pad);
    plot.axes({ xTicks: ticks(xr), yTicks: ticks(yr), xLabel: 'β s1 · cholesterol TP', yLabel: 'β s2 · LDL', fx: v => vn(v, 0), fy: v => vn(v, 0) });
    plot.line([[xr[0], 0], [xr[1], 0]], alpha(C.white, 0.35), { width: 1 });
    plot.line([[0, yr[0]], [0, yr[1]]], alpha(C.white, 0.35), { width: 1 });
    const b = BOOT();
    const groups: ['ols' | 'ridge' | 'lasso', string, number][] = [['ols', C.magenta, 0.4], ['ridge', C.cyan, 5.5], ['lasso', C.amber, 8.5]];
    plot.clip(() => groups.forEach(([key, color, start]) => {
      const shown = Math.floor(seg(t, start, start + 2.2) * b.length);
      b.slice(0, shown).forEach(r => g.circle(plot.px(r[key][4]), plot.py(r[key][5]), 3.6, color, { glow: 8, alpha: 0.85 }));
    }));
    const full = { ols: ols(FULL), ridge: ridge(FULL, BOOT_LAMBDA.ridge), lasso: lasso(FULL, BOOT_LAMBDA.lasso) };
    g.panel(right.x, right.y, right.w, right.h, 'Trên toàn bộ 442 bệnh nhân', C.amber);
    const rowsY = right.y + 60, rowH = Math.min(70, (right.h - 90) / 3), x0 = right.x + 110, w = right.w - 140, scale = w / 90;
    g.line([[x0 + 45 * scale, rowsY - 16], [x0 + 45 * scale, rowsY + rowH * 3 - 10]], alpha(C.white, 0.3), { width: 1 });
    groups.forEach(([key, color], i) => {
      const y = rowsY + i * rowH, beta = full[key];
      g.text(key === 'ols' ? 'OLS' : key === 'ridge' ? 'Ridge λ = 0,1' : 'Lasso λ = 1', right.x + 16, y + 12, { size: 12, weight: 650, color });
      [[4, 's1'], [5, 's2']].forEach(([j, name], k) => {
        const v = beta[j as number], yy = y + k * 20;
        g.ctx.save(); g.ctx.fillStyle = color; g.ctx.shadowColor = color; g.ctx.shadowBlur = 10;
        g.ctx.fillRect(Math.min(x0 + 45 * scale, x0 + (45 + v) * scale), yy, Math.max(2, Math.abs(v) * scale), 12); g.ctx.restore();
        g.text(`${name} ${vn(v, 1)}`, x0 + 45 * scale + (v >= 0 ? Math.abs(v) * scale + 6 : -Math.abs(v) * scale - 6), yy + 6, { size: 11, color: C.ink, align: v >= 0 ? 'left' : 'right', mono: true });
      });
    });
    g.text(`tương quan s1–s2 = ${vn(0.8967, 2)}`, right.x + 16, right.y + right.h - 20, { size: 11.5, color: C.muted });
  },
  metrics() {
    const b = BOOT();
    return (['ols', 'ridge', 'lasso'] as const).map((k, i) => ({ label: `độ lệch β s1 · ${k === 'ols' ? 'OLS' : k === 'ridge' ? 'Ridge' : 'Lasso'}`, value: `±${vn(stats(b.map(r => r[k][4])).sd, 1)}`, tone: (['magenta', 'cyan', 'amber'] as const)[i] }));
  },
  table: () => { const b = BOOT(); return { headers: ['Cách', 'β s1 (TB ± độ lệch)', 'β s2 (TB ± độ lệch)', 'Số lần s1 = 0', 'Số lần s2 = 0'], rows: (['ols', 'ridge', 'lasso'] as const).map(k => { const s1 = stats(b.map(r => r[k][4])), s2 = stats(b.map(r => r[k][5])); return [k === 'ols' ? 'OLS' : k === 'ridge' ? 'Ridge λ = 0,1' : 'Lasso λ = 1', `${vn(s1.mean, 1)} ± ${vn(s1.sd, 1)}`, `${vn(s2.mean, 1)} ± ${vn(s2.sd, 1)}`, b.filter(r => r[k][4] === 0).length, b.filter(r => r[k][5] === 0).length]; }), caption: '80 mẫu bootstrap (hạt giống cố định).' }; },
};

export const initRegularizationLab = () => glassLab('regularization', [chapter1, chapter2, chapter3, chapter4], lab => {
  lab.range('lam2d', value => vn(value, 1), value => { state.lam2d = value; });
  lab.range('shrink', value => vn(value, 2), value => { state.shrink = value; });
  lab.range('lamindex', value => `λ = ${vn(LAMBDAS[value], LAMBDAS[value] < 1 ? 3 : 1)}`, value => { state.index = value; });
  lab.on('[data-autoplay]', 'click', () => { state.lam2d = null; state.shrink = null; state.index = null; lab.replay(); });
});
