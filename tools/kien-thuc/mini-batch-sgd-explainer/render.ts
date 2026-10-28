// Batch / mini-batch / SGD lab on 20 640 California block groups: one exact step per epoch, one noisy step per sample,
// the noise fan of mini-batches, then a race with the same budget of per-sample gradients.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, Plot, alpha, clamp, seg, split, vn, type Rect } from '../../_shared/glass-lab/gfx';
import type { Theta } from '../../_shared/glass-lab/regression';
import { at, heightOf, lossChart, lossMap, trail } from '../../_shared/glass-lab/views';
import { BATCHES, INCOME, LR, N, START, VALUE, VIEW, gradientNoise, problem, sampleGradients, train } from './model';

export const explainerKind = 'educational-simulation';

const height = heightOf(problem, 0.3, 40);
const map = (f: Frame, rect: Rect, title: string) => lossMap(f, rect, problem, VIEW, height, title, ['w (trăm nghìn $ / chục nghìn $)', 'b']);
const nf = (value: number) => Math.round(value).toLocaleString('vi-VN');
const state = { batchIndex: 2 };
let currentT = 0;

/** All 20 640 block groups as tiny dots, plus the current line and optional highlighted rows. */
function scatter(f: Frame, rect: Rect, theta: Theta, highlight: ArrayLike<number> = [], pulse = 0) {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, `${nf(N)} khu dân cư California (điều tra 1990)`, C.cyan);
  const plot = new Plot(g, rect, [0, 15.5], [0, 5.4], { l: 40, r: 14, t: 40, b: 34 });
  plot.axes({ xTicks: [0, 3, 6, 9, 12, 15], yTicks: [0, 1, 2, 3, 4, 5], xLabel: 'thu nhập trung vị (chục nghìn $ / năm)', yLabel: 'giá nhà trung vị (trăm nghìn $)', fx: v => vn(v, 0), fy: v => vn(v, 0) });
  const { ctx } = g;
  ctx.save(); ctx.fillStyle = alpha(C.cyan, 0.22);
  for (let i = 0; i < N; i++) ctx.fillRect(plot.px(INCOME[i]) - 0.8, plot.py(VALUE[i]) - 0.8, 1.6, 1.6);
  ctx.restore();
  plot.clip(() => {
    for (let k = 0; k < highlight.length; k++) {
      const i = highlight[k], x = plot.px(INCOME[i]), y = plot.py(VALUE[i]), yHat = plot.py(theta[0] * INCOME[i] + theta[1]);
      g.line([[x, y], [x, yHat]], alpha(C.magenta, highlight.length > 50 ? 0.25 : 0.8), { width: highlight.length > 50 ? 1 : 1.6 });
      g.circle(x, y, highlight.length > 50 ? 2.2 : 4 + pulse * 6, C.amber, { glow: 16, alpha: highlight.length > 50 ? 0.8 : 1 });
    }
    plot.line([[0, theta[1]], [15.5, theta[0] * 15.5 + theta[1]]], C.amber, { width: 3, glow: 18 });
  });
  g.text('giá bị chặn ở 5,0 trong dữ liệu gốc', plot.px(15.3), plot.py(5.1), { size: 10.5, color: C.muted, align: 'right' });
  return plot;
}

// ---------- Chapter 1: batch GD ----------
const EPOCHS1 = 12;
const BATCH_RUN = train(N, LR.batch, EPOCHS1);
const chapter1: Scene = {
  title: 'Batch GD: mỗi bước xem cả 20 640 khu',
  duration: 12,
  beats: () => [
    { at: 0, text: `Dự đoán giá nhà của mỗi khu từ thu nhập. Loss là MSE trên toàn bộ ${nf(N)} khu.` },
    { at: 2.5, text: 'Batch GD tính gradient chính xác trên mọi khu rồi mới đi một bước: mượt và chắc chắn.' },
    { at: 6, text: `Nhưng mỗi bước tốn ${nf(N)} lần tính gradient của từng mẫu. Một epoch chỉ cho đúng một bước.` },
    { at: 9.5, text: `Sau ${EPOCHS1} epoch, MSE mới từ ${vn(problem.loss(START), 2)} xuống ${vn(problem.loss(BATCH_RUN.points[EPOCHS1]), 2)}; cần khoảng ${problem.stepsTo(START, LR.batch)} epoch mới vào trong 1% của mức tối thiểu ${vn(problem.minLoss, 3)}.` },
  ],
  formula: `θ ← θ − η · (1/N) Σᵢ ∇ℓᵢ(θ),  N = ${nf(N)},  η = ${vn(LR.batch, 2)}`,
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.25, 1], narrow);
    const k = seg(t, 2.5, 11) * EPOCHS1, theta = at(BATCH_RUN.points, k);
    scatter(f, left, theta);
    const plot = map(f, right, 'Mặt loss nhìn từ trên xuống');
    trail(plot, BATCH_RUN.points, k, C.magenta);
  },
  metrics() {
    const k = Math.round(seg(currentT, 2.5, 11) * EPOCHS1);
    return [{ label: 'Epoch = bước', value: String(k), tone: 'violet' }, { label: 'Gradient mẫu đã tính', value: nf(k * N), tone: 'cyan' }, { label: 'MSE', value: vn(problem.loss(BATCH_RUN.points[k]), 3), tone: 'magenta' }];
  },
  table: () => ({ headers: ['Epoch', 'w', 'b', 'MSE'], rows: BATCH_RUN.points.map((p, k) => [k, vn(p[0], 4), vn(p[1], 4), vn(problem.loss(p), 4)]) }),
};

// ---------- Chapter 2: SGD ----------
const SGD_RUN = train(1, LR.sgd, 1, 7, 1, 60);
/** Updates shown at time t: one by one for the first seconds, then fast-forward through the whole epoch. */
const sgdStep = (t: number) => (t < 2 ? 0 : t < 7 ? (t - 2) * 6 : 30 + seg(t, 7, 13) ** 2.2 * (N - 30));

const chapter2: Scene = {
  title: 'SGD: mỗi bước chỉ một khu',
  duration: 14,
  beats: () => [
    { at: 0, text: 'SGD bốc ngẫu nhiên một khu (chấm cam), tính gradient chỉ trên khu đó rồi đi luôn một bước.' },
    { at: 2.5, text: 'Mỗi khu kéo đường thẳng về phía mình: hướng đi gần như ngẫu nhiên, đường thẳng giật liên tục.' },
    { at: 7, text: 'Nhưng mỗi bước rẻ hơn 20 640 lần. Tua nhanh hết một epoch: hàng nghìn bước nhỏ nhiễu cộng lại vẫn đi đúng hướng.' },
    { at: 11.5, text: `Chỉ sau ~1 000 khu, SGD đã thấp hơn batch GD sau ${EPOCHS1} epoch. Gần đáy nó không dừng hẳn mà rung quanh đáy.` },
  ],
  formula: `θ ← θ − η · ∇ℓᵢ(θ),  i lấy lần lượt theo thứ tự đã xáo trộn,  η = ${vn(LR.sgd, 3)}`,
  render(f) {
    const { t, narrow, time } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.25, 1], narrow);
    const k = sgdStep(t), theta = at(SGD_RUN.points, k);
    const recent = k < 60 ? [SGD_RUN.lastBatch[Math.floor(k)]?.[0] ?? 0] : [];
    scatter(f, left, theta, recent, k < 60 ? (Math.sin(time * 8) + 1) / 2 : 0);
    const plot = map(f, right, 'Đường đi của SGD');
    const every = k > 600 ? 20 : 1;
    const path = SGD_RUN.points.filter((_, i) => i % every === 0 || i === Math.floor(k));
    trail(plot, path, Math.min(path.length - 1, k / every), C.cyan, { dots: k < 60, width: k > 600 ? 1.1 : 2 });
  },
  metrics() {
    const k = Math.floor(sgdStep(currentT));
    return [{ label: 'Bước = khu đã dùng', value: nf(k), tone: 'cyan' }, { label: 'Epoch', value: vn(k / N, 2), tone: 'violet' }, { label: 'MSE trên cả 20 640 khu', value: vn(problem.loss(SGD_RUN.points[Math.min(k, SGD_RUN.points.length - 1)]), 3), tone: 'amber' }];
  },
  table: () => ({ headers: ['Số khu đã dùng', 'MSE'], rows: [0, 10, 50, 100, 500, 1000, 2000, 5000, 10000, 20640].map(k => [nf(k), vn(problem.loss(SGD_RUN.points[k]), 4)]) }),
};

// ---------- Chapter 3: mini-batch noise ----------
const MINI_RUNS = new Map<number, ReturnType<typeof train>>();
const miniRun = (batch: number) => { if (!MINI_RUNS.has(batch)) MINI_RUNS.set(batch, train(batch, batch === 1 ? LR.sgd : batch >= N ? LR.batch : LR.mini, 1, 7, Math.max(1, Math.floor(N / batch / 400)))); return MINI_RUNS.get(batch)!; };
const NOISE_START = BATCHES.map(b => gradientNoise(START, b));
const GRAD_START = Math.hypot(...problem.gradient(START));

const chapter3: Scene = {
  title: 'Mini-batch: trung bình B khu',
  duration: 12,
  beats: () => [
    { at: 0, text: 'Mỗi mũi tên mờ là bước mà một lô ngẫu nhiên khác sẽ cho. Lô càng nhỏ, các mũi tên càng tản.' },
    { at: 3, text: `Với 1 khu, độ lệch của gradient là ${vn(NOISE_START[0], 1)}, gấp ${vn(NOISE_START[0] / GRAD_START, 0)} lần độ lớn gradient thật (${vn(GRAD_START, 2)}). Nhiễu lấn át tín hiệu.` },
    { at: 6.5, text: 'Lấy trung bình B khu chia độ lệch cho khoảng √B: lô 64 giảm 8 lần, lô 4 096 giảm 64 lần.' },
    { at: 9.5, text: 'Mini-batch vừa đủ ít nhiễu để đi thẳng, vừa có hàng trăm bước mỗi epoch, lại khớp với phép tính song song trên GPU.' },
  ],
  formula: 'ĝ = (1/B) Σ_{i ∈ lô} ∇ℓᵢ(θ)\nđộ lệch(ĝ) = σ · √((N − B) / (B (N − 1)))',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.2, 1] : [1.25, 1], narrow);
    const batch = BATCHES[state.batchIndex], run = miniRun(batch);
    const k = seg(t, 0.5, 11.5) * (run.points.length - 1), theta = at(run.points, k);
    const plot = map(f, left, `Lô B = ${nf(batch)} · ${nf(Math.ceil(N / batch))} bước mỗi epoch`);
    const lr = run.lr;
    plot.clip(() => {
      for (const grad of sampleGradients(theta, batch, 48, 11)) {
        const [x0, y0] = plot.p(theta), [x1, y1] = plot.p([theta[0] - lr * grad[0] * 4, theta[1] - lr * grad[1] * 4]);
        g.arrow(x0, y0, x1, y1, C.amber, { width: 1.1, alpha: 0.32, head: 6 });
      }
      const full = problem.gradient(theta), [x0, y0] = plot.p(theta), [x1, y1] = plot.p([theta[0] - lr * full[0] * 4, theta[1] - lr * full[1] * 4]);
      g.arrow(x0, y0, x1, y1, C.lime, { width: 2.6, glow: 12 });
    });
    trail(plot, run.points, k, C.amber, { dots: false, width: 1.6 });
    g.tag('xanh lá: gradient thật · cam: các lô ngẫu nhiên (phóng to 4 lần)', plot.box.x + 8, plot.box.y + 14, C.muted, 'left');
    noiseBars(f, right, theta);
  },
  metrics() {
    const batch = BATCHES[state.batchIndex], run = miniRun(batch), theta = at(run.points, seg(currentT, 0.5, 11.5) * (run.points.length - 1));
    return [{ label: 'Kích thước lô B', value: nf(batch), tone: 'amber' }, { label: 'Độ lệch ĝ', value: vn(gradientNoise(theta, batch), 3), tone: 'magenta' }, { label: '‖∇L‖ thật', value: vn(Math.hypot(...problem.gradient(theta)), 3), tone: 'lime' }];
  },
  table: () => ({ headers: ['B', 'Bước mỗi epoch', 'Độ lệch ĝ tại điểm xuất phát', 'Tỉ số nhiễu / ‖∇L‖', 'MSE sau 1 epoch'], rows: BATCHES.map((b, i) => [nf(b), nf(Math.ceil(N / b)), vn(NOISE_START[i], 3), vn(NOISE_START[i] / GRAD_START, 2), vn(problem.loss(miniRun(b).points.at(-1)!), 4)]) }),
};

/** Log-scale bars of the mini-batch noise for each B, with the true gradient size as a reference line. */
function noiseBars(f: Frame, rect: Rect, theta: Theta) {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, 'Độ lệch chuẩn của ĝ theo kích thước lô', C.magenta);
  const noise = BATCHES.map(b => gradientNoise(theta, b)), grad = Math.hypot(...problem.gradient(theta));
  const box = { x: rect.x + 76, y: rect.y + 48, w: rect.w - 140, h: rect.h - 70 };
  const lo = Math.log10(0.01), hi = Math.log10(30), x = (v: number) => box.x + (Math.log10(Math.max(v, 0.01)) - lo) / (hi - lo) * box.w;
  const row = box.h / BATCHES.length;
  BATCHES.forEach((b, i) => {
    const y = box.y + i * row + row / 2, on = i === state.batchIndex;
    g.text(`B = ${nf(b)}`, box.x - 10, y, { size: 11.5, color: on ? C.ink : C.muted, align: 'right', weight: on ? 650 : 500 });
    const { ctx } = g;
    ctx.save();
    const grad2 = ctx.createLinearGradient(box.x, 0, x(noise[i]), 0);
    grad2.addColorStop(0, alpha(on ? C.amber : C.violet, 0.35)); grad2.addColorStop(1, on ? C.amber : C.violet);
    ctx.fillStyle = noise[i] > 0 ? grad2 : 'transparent';
    if (on) { ctx.shadowColor = C.amber; ctx.shadowBlur = 14; }
    ctx.beginPath(); ctx.roundRect(box.x, y - row * 0.3, Math.max(2, x(noise[i]) - box.x), row * 0.6, 6); ctx.fill();
    ctx.restore();
    g.text(noise[i] > 0 ? vn(noise[i], noise[i] < 1 ? 3 : 2) : '0 (gradient thật)', Math.max(box.x + 6, x(noise[i])) + 8, y, { size: 11.5, color: on ? C.amber : C.ink, mono: true });
  });
  g.line([[x(grad), box.y - 6], [x(grad), box.y + box.h]], C.lime, { width: 1.6, dash: [5, 4], glow: 8 });
  g.text(`‖∇L‖ = ${vn(grad, 2)}`, x(grad) + 5, box.y - 8, { size: 11, color: C.lime });
  g.text('thang log', box.x + box.w, box.y + box.h + 12, { size: 10.5, color: C.muted, align: 'right' });
  g.line([[clamp(box.x, 0, 1e9), box.y + box.h + 2], [box.x + box.w, box.y + box.h + 2]], C.faint, { width: 1 });
}

// ---------- Chapter 4: the race ----------
const EPOCHS4 = 2;
const RACE = [
  { name: 'Batch GD', color: C.magenta, run: train(N, LR.batch, EPOCHS4) },
  { name: 'Mini-batch 64', color: C.amber, run: train(64, LR.mini, EPOCHS4) },
  { name: 'SGD', color: C.cyan, run: train(1, LR.sgd, EPOCHS4, 7, 20) },
];
const indexAt = (samples: readonly number[], budget: number) => { let lo = 0, hi = samples.length - 1; while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (samples[mid] <= budget) lo = mid; else hi = mid - 1; } return lo; };
/** Loss against epochs spent; batch GD holds its value until a full pass is done, so draw it as steps. */
function staircase(run: ReturnType<typeof train>, color: string) {
  const values: number[] = [], x: number[] = [];
  run.points.forEach((p, i) => {
    if (run.batch >= N && i > 0) { values.push(values.at(-1)!); x.push(run.samples[i] / N - 1e-6); }
    values.push(problem.loss(p)); x.push(run.samples[i] / N);
  });
  return { values, x, color };
}
const budgetAt = (t: number) => seg(t, 1, 12.5) * EPOCHS4 * N;

const chapter4: Scene = {
  title: 'Cuộc đua cùng ngân sách',
  duration: 14,
  beats: () => [
    { at: 0, text: 'Cả ba được cùng một ngân sách: mỗi lần tính gradient của một khu tốn một đơn vị. Trục ngang là số đơn vị đã tiêu.' },
    { at: 3, text: 'SGD lao xuống ngay từ vài nghìn khu đầu tiên; mini-batch theo sát với đường đi mượt hơn.' },
    { at: 7, text: 'Batch GD chưa nhúc nhích cho tới khi xem hết 20 640 khu, rồi mới đi đúng một bước.' },
    { at: 11, text: `Sau 2 epoch: batch GD ${vn(problem.loss(RACE[0].run.points.at(-1)!), 3)}, mini-batch ${vn(problem.loss(RACE[1].run.points.at(-1)!), 3)}, SGD ${vn(problem.loss(RACE[2].run.points.at(-1)!), 3)} (tối thiểu ${vn(problem.minLoss, 3)}).` },
  ],
  formula: 'cùng chi phí mỗi epoch: N gradient mẫu\nsố bước mỗi epoch: 1 (batch) · ⌈N/B⌉ (mini-batch) · N (SGD)',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.1, 1], narrow);
    const budget = budgetAt(t);
    const plot = map(f, left, 'Ba cách, cùng điểm xuất phát');
    for (const racer of RACE) trail(plot, racer.run.points, indexAt(racer.run.samples, budget), racer.color, { dots: racer.run.batch >= N, width: racer.run.batch === 1 ? 1.1 : 1.8 });
    RACE.forEach((racer, i) => f.g.tag(`${racer.name} · η = ${vn(racer.run.lr, 3)}`, plot.box.x + 8, plot.box.y + 14 + i * 26, racer.color, 'left'));
    lossChart(f, right, RACE.map(racer => staircase(racer.run, racer.color)), budget / N, 2.6, problem.minLoss, 'MSE theo số epoch đã tiêu', 'epoch (×20 640 gradient mẫu)', EPOCHS4, v => vn(v, 1));
  },
  metrics: () => RACE.map((racer, i) => ({ label: racer.name, value: vn(problem.loss(racer.run.points[indexAt(racer.run.samples, budgetAt(currentT))]), 3), tone: (['magenta', 'amber', 'cyan'] as const)[i] })),
  table: () => ({ headers: ['Epoch đã tiêu', ...RACE.map(r => r.name)], rows: [0.05, 0.1, 0.25, 0.5, 1, 1.5, 2].map(e => [vn(e, 2), ...RACE.map(r => vn(problem.loss(r.run.points[indexAt(r.run.samples, e * N)]), 4))]) }),
};

export const initMiniBatchSgdLab = () => glassLab('mini-batch-sgd', [chapter1, chapter2, chapter3, chapter4], lab => {
  lab.range('batch', value => `${nf(BATCHES[value])} khu`, value => { state.batchIndex = value; lab.replay(); });
});
