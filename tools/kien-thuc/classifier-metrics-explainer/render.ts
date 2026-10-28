// Classifier evaluation lab: a threshold turns scores into decisions and a confusion matrix; accuracy hides class
// imbalance; precision, recall and F1 trade off along the threshold; ROC/AUC and the precision–recall curve summarise
// all thresholds; the same test means very different precision at a different prevalence.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, Plot, alpha, seg, split, vn, type Point, type Rect } from '../../_shared/glass-lab/gfx';
import { barChart, categoryChart } from '../../_shared/glass-lab/views';
import { D, NEG, POS, metrics, pr, precisionAt, roc } from './model';

export const explainerKind = 'educational-simulation';

const state = { threshold: null as number | null, prevalence: 2 };
let currentT = 0;
const ROC = roc(), PR = pr();
const PREVALENCES = [0.001, 0.01, 0.05, 0.1, POS / (POS + NEG), 0.5];
const thrNow = (from = 1, to = 10) => state.threshold ?? (currentT < from ? 0.5 : Math.max(0.02, Math.min(0.98, 0.5 + 0.45 * Math.sin((currentT - from) / (to - from) * Math.PI * 2))));
const pct = (v: number, d = 1) => `${vn(v * 100, d)}%`;

function histograms(f: Frame, rect: Rect, thr: number, title: string) {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, title, C.violet);
  const plot = new Plot(g, rect, [0, 1], [0, 1], { l: 30, r: 16, t: 40, b: 34 });
  plot.axes({ xTicks: [0, 0.25, 0.5, 0.75, 1], yTicks: [], xLabel: `điểm mô hình (xác suất ${D.positive})`, fx: v => vn(v, 2) });
  const bins = 25, neg = new Array(bins).fill(0), pos = new Array(bins).fill(0);
  D.scores.forEach((s, i) => (D.labels[i] ? pos : neg)[Math.min(bins - 1, Math.floor(s * bins))]++);
  const peak = Math.max(...neg, ...pos), mid = 0.5;
  for (let i = 0; i < bins; i++) {
    const x0 = plot.px(i / bins), x1 = plot.px((i + 1) / bins), above = (i + 0.5) / bins >= thr;
    const hn = (plot.py(mid) - plot.py(0.97)) * neg[i] / peak, hp = (plot.py(mid) - plot.py(0.97)) * pos[i] / peak;
    g.ctx.fillStyle = alpha(above ? C.red : C.cyan, 0.75); g.ctx.fillRect(x0 + 0.5, plot.py(mid) - hn, x1 - x0 - 1, hn);
    g.ctx.fillStyle = alpha(above ? C.amber : C.magenta, 0.75); g.ctx.fillRect(x0 + 0.5, plot.py(mid) + 2, x1 - x0 - 1, hp);
  }
  g.text(`${D.negative} (${NEG}) ↑`, plot.box.x + plot.box.w - 6, plot.py(0.93), { size: 11.5, color: C.cyan, align: 'right' });
  g.text(`${D.positive} (${POS}) ↓`, plot.box.x + 6, plot.py(0.07), { size: 11.5, color: C.amber });
  plot.line([[thr, 0], [thr, 1]], C.white, { width: 2, glow: 10 });
  g.tag(`ngưỡng ${vn(thr, 2)}`, plot.px(thr), plot.box.y + 12, C.white);
}
function matrix(f: Frame, rect: Rect, thr: number) {
  const { g } = f, m = metrics(thr);
  g.panel(rect.x, rect.y, rect.w, rect.h, 'Ma trận nhầm lẫn', C.amber);
  const cw = Math.min((rect.w - 110) / 2, (rect.h - 110) / 2), ox = rect.x + 90, oy = rect.y + 74;
  const cells = [[m.tp, 'TP', 'dương tính thật', C.lime], [m.fn, 'FN', 'bỏ sót', C.red], [m.fp, 'FP', 'báo động giả', C.amber], [m.tn, 'TN', 'âm tính thật', C.cyan]] as const;
  cells.forEach(([v, code, label, color], k) => {
    const x = ox + (k % 2) * cw, y = oy + Math.floor(k / 2) * cw;
    g.ctx.fillStyle = alpha(color, 0.12 + 0.6 * v / Math.max(POS, NEG)); g.ctx.fillRect(x + 3, y + 3, cw - 6, cw - 6);
    g.text(String(v), x + cw / 2, y + cw / 2 - 8, { size: Math.min(28, cw / 3), color: C.ink, align: 'center', weight: 700, mono: true });
    g.text(`${code} · ${label}`, x + cw / 2, y + cw / 2 + 16, { size: 11, color: C.ink, align: 'center' });
  });
  g.text(`dự đoán ${D.positive}`, ox + cw / 2, oy - 12, { size: 11, color: C.muted, align: 'center' });
  g.text(`dự đoán ${D.negative}`, ox + cw * 1.5, oy - 12, { size: 11, color: C.muted, align: 'center' });
  g.text(`thật: ${D.positive}`, ox - 8, oy + cw / 2, { size: 11, color: C.muted, align: 'right' });
  g.text(`thật: ${D.negative}`, ox - 8, oy + cw * 1.5, { size: 11, color: C.muted, align: 'right' });
}

// ---------- Chapter 1: threshold → confusion matrix ----------
const chapter1: Scene = {
  title: 'Ngưỡng và ma trận nhầm lẫn',
  duration: 11,
  idle: false,
  beats: () => [
    { at: 0, text: `Mô hình hồi quy logistic cho mỗi khối u trong tập kiểm tra (${D.testSize} ca, chưa thấy khi huấn luyện) một điểm từ 0 tới 1.` },
    { at: 2.5, text: `Chọn một ngưỡng: điểm ≥ ngưỡng thì dự đoán ${D.positive}. Hai histogram tách nhau tốt nhưng vẫn chồng lấn ở giữa.` },
    { at: 5, text: 'Bốn ô của ma trận nhầm lẫn: dương tính thật (TP), bỏ sót (FN), báo động giả (FP), âm tính thật (TN).' },
    { at: 8, text: 'Kéo ngưỡng xuống: bắt được thêm ca ác tính nhưng báo động giả tăng. Không có ngưỡng nào tốt cho mọi mục đích.' },
  ],
  formula: () => { const m = metrics(thrNow()); return `ngưỡng ${vn(thrNow(), 2)}: TP = ${m.tp}, FN = ${m.fn}, FP = ${m.fp}, TN = ${m.tn}`; },
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.1, 1] : [1.4, 1], narrow);
    const thr = thrNow(3, 11);
    histograms(f, left, thr, 'Phân bố điểm của hai lớp');
    matrix(f, right, thr);
  },
  metrics: () => { const m = metrics(thrNow(3, 11)); return [{ label: 'Ngưỡng', value: vn(thrNow(3, 11), 2), tone: 'violet' }, { label: 'Bỏ sót (FN)', value: String(m.fn), tone: 'red' }, { label: 'Báo động giả (FP)', value: String(m.fp), tone: 'amber' }]; },
};

// ---------- Chapter 2: accuracy can mislead ----------
const chapter2: Scene = {
  title: 'Accuracy có thể đánh lừa',
  duration: 11,
  beats: () => {
    const m = metrics(0.5);
    return [
      { at: 0, text: `Ở ngưỡng 0,5 mô hình đúng ${pct(m.accuracy)} số ca. Nghe tốt, nhưng accuracy gộp mọi loại lỗi làm một.` },
      { at: 3, text: `Một “mô hình” luôn trả lời ${D.negative} cũng đúng ${pct(NEG / (POS + NEG))} trên tập này, mà không phát hiện được ca ác tính nào.` },
      { at: 6, text: 'Khi lớp quan tâm hiếm (gian lận 0,1%, bệnh hiếm 1%), mô hình luôn trả lời “không” đạt 99% accuracy và hoàn toàn vô dụng.' },
      { at: 8.5, text: 'Vì vậy cần các chỉ số nhìn riêng từng loại lỗi: recall (độ nhạy), specificity (độ đặc hiệu), precision.' },
    ];
  },
  formula: 'accuracy = (TP + TN) / tổng;  luôn đoán lớp đa số: accuracy = tỷ lệ lớp đa số',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, [1, 1], narrow);
    const m = metrics(0.5);
    barChart(f, left, 'Trên tập kiểm tra (37% ác tính)', ['accuracy', 'recall', 'specificity'], [m.accuracy * 100, m.recall * 100, m.specificity * 100], { max: 110, colors: [C.violet, C.lime, C.cyan], reveal: seg(t, 0.5, 2.5), format: v => `${vn(v, 1)}%` });
    barChart(f, right, `Luôn đoán “${D.negative}”`, ['accuracy', 'recall', 'specificity'], [NEG / (POS + NEG) * 100, 0, 100], { max: 110, colors: [C.violet, C.red, C.cyan], reveal: seg(t, 3, 5), format: v => `${vn(v, 1)}%`, accent: C.red });
  },
  metrics: () => [{ label: 'Mô hình', value: pct(metrics(0.5).accuracy), tone: 'lime' }, { label: 'Luôn đoán lành tính', value: pct(NEG / (POS + NEG)), tone: 'red' }],
};

// ---------- Chapter 3: precision, recall, F1 ----------
const THRS = Array.from({ length: 19 }, (_, i) => (i + 1) / 20);
const chapter3: Scene = {
  title: 'Precision, recall và F1',
  duration: 12,
  idle: false,
  beats: () => [
    { at: 0, text: 'Recall (độ nhạy) = TP / (TP + FN): trong các ca ác tính thật, bắt được bao nhiêu phần trăm.' },
    { at: 3, text: 'Precision = TP / (TP + FP): trong các ca bị báo ác tính, bao nhiêu phần trăm đúng là ác tính.' },
    { at: 6, text: 'Hạ ngưỡng: recall tăng, precision giảm. Nâng ngưỡng: ngược lại. F1 là trung bình điều hòa của hai chỉ số.' },
    { at: 9, text: 'Sàng lọc ung thư ưu tiên recall (bỏ sót rất đắt), lọc thư rác ưu tiên precision (xóa nhầm thư quan trọng rất đắt).' },
  ],
  formula: () => { const m = metrics(thrNow(1, 11)); return `ngưỡng ${vn(thrNow(1, 11), 2)}: precision = ${m.tp}/${m.tp + m.fp} = ${pct(m.precision)},  recall = ${m.tp}/${m.tp + m.fn} = ${pct(m.recall)}\nF1 = 2PR/(P + R) = ${pct(m.f1)}`; },
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.1, 1], narrow);
    const thr = thrNow(1, 11);
    histograms(f, left, thr, 'Kéo ngưỡng');
    categoryChart(f, right, 'Theo ngưỡng', THRS.map(v => (Math.round(v * 20) % 4 === 0 ? vn(v, 1) : '')), [
      { label: 'precision', values: THRS.map(v => metrics(v).precision), color: C.amber, format: v => pct(v, 0) },
      { label: 'recall', values: THRS.map(v => metrics(v).recall), color: C.lime, format: v => pct(v, 0) },
      { label: 'F1', values: THRS.map(v => metrics(v).f1), color: C.violet, format: v => pct(v, 0) },
    ], [0.4, 1], Math.max(0, Math.min(THRS.length - 1, Math.round(thr * 20) - 1)), 1, v => pct(v, 0), 'ngưỡng');
  },
  metrics: () => { const m = metrics(thrNow(1, 11)); return [{ label: 'Precision', value: pct(m.precision), tone: 'amber' }, { label: 'Recall', value: pct(m.recall), tone: 'lime' }, { label: 'F1', value: pct(m.f1), tone: 'violet' }]; },
};

// ---------- Chapter 4: ROC and AUC ----------
const chapter4: Scene = {
  title: 'Đường ROC và AUC',
  duration: 11,
  idle: false,
  beats: () => [
    { at: 0, text: 'Mỗi ngưỡng cho một điểm (tỷ lệ báo động giả, recall). Nối mọi ngưỡng lại được đường ROC.' },
    { at: 3.5, text: 'Đường chéo là đoán ngẫu nhiên. Góc trên trái là hoàn hảo: bắt hết ca dương mà không báo động giả.' },
    { at: 6.5, text: `AUC = ${vn(ROC.auc, 4)}: xác suất một ca ác tính ngẫu nhiên có điểm cao hơn một ca lành tính ngẫu nhiên.` },
  ],
  formula: () => { const m = metrics(thrNow(1, 10)); return `FPR = FP / (FP + TN) = ${pct(m.fpr)},  TPR = recall = ${pct(m.recall)}\nAUC = P(điểm ca dương > điểm ca âm) = ${vn(ROC.auc, 4)}`; },
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1, 1.1], narrow);
    const thr = thrNow(1, 10);
    histograms(f, left, thr, 'Ngưỡng hiện tại');
    g.panel(right.x, right.y, right.w, right.h, `ROC · AUC = ${vn(ROC.auc, 3)}`, C.lime);
    const size = Math.min(right.w - 70, right.h - 80), plot = new Plot(g, { x: right.x, y: right.y, w: size + 70, h: size + 80 }, [0, 1], [0, 1], { l: 46, r: 24, t: 44, b: 36 });
    plot.axes({ xTicks: [0, 0.5, 1], yTicks: [0, 0.5, 1], xLabel: 'tỷ lệ báo động giả (FPR)', yLabel: 'recall (TPR)', fx: v => vn(v, 1), fy: v => vn(v, 1) });
    plot.line([[0, 0], [1, 1]], alpha(C.white, 0.3), { width: 1, dash: [4, 4] });
    const show = seg(t, 0.5, 3.5), n = Math.max(2, Math.floor(ROC.pts.length * show));
    const area: Point[] = [...ROC.pts.slice(0, n), [ROC.pts[n - 1][0], 0], [0, 0]];
    if (t > 6.5) g.polygon(area.map(p => [plot.px(p[0]), plot.py(p[1])] as Point), alpha(C.lime, 0.15));
    plot.line(ROC.pts.slice(0, n), C.lime, { width: 2.4, glow: 10 });
    const m = metrics(thr);
    g.circle(plot.px(m.fpr), plot.py(m.recall), 7, C.amber, { glow: 14 });
  },
  metrics: () => { const m = metrics(thrNow(1, 10)); return [{ label: 'FPR', value: pct(m.fpr), tone: 'amber' }, { label: 'TPR', value: pct(m.recall), tone: 'lime' }, { label: 'AUC', value: vn(ROC.auc, 4), tone: 'violet' }]; },
};

// ---------- Chapter 5: precision depends on prevalence ----------
const chapter5: Scene = {
  title: 'Precision phụ thuộc tỷ lệ bệnh',
  duration: 11,
  beats: () => {
    const m = metrics(0.5), p1 = precisionAt(m.recall, m.specificity, 0.01);
    return [
      { at: 0, text: `Đường precision–recall vẽ precision theo recall cho mọi ngưỡng. Average precision = ${vn(PR.ap, 3)} (scikit-learn: ${vn(D.prAuc, 3)}).` },
      { at: 3.5, text: `Tập kiểm tra có 37% ca ác tính. Trong thực tế sàng lọc, tỷ lệ có thể chỉ 1%.` },
      { at: 6, text: `Giữ nguyên độ nhạy ${pct(m.recall, 0)} và độ đặc hiệu ${pct(m.specificity, 0)}, áp vào nhóm có 1% bệnh: precision chỉ còn ${pct(p1, 0)}. Phần lớn ca dương tính là báo động giả.` },
      { at: 9, text: 'ROC và AUC không đổi theo tỷ lệ bệnh; precision thì có. Với lớp hiếm, hãy nhìn đường precision–recall.' },
    ];
  },
  formula: () => { const m = metrics(0.5), prev = PREVALENCES[state.prevalence]; return `precision = TPR·π / (TPR·π + FPR·(1 − π)),  π = ${vn(prev * 100, prev < 0.01 ? 1 : 0)}%\n= ${pct(precisionAt(m.recall, m.specificity, prev))}`; },
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, `Precision–recall · AP = ${vn(PR.ap, 3)}`, C.amber);
    const plot = new Plot(g, left, [0, 1], [0, 1.02], { l: 46, r: 20, t: 44, b: 36 });
    plot.axes({ xTicks: [0, 0.5, 1], yTicks: [0, 0.5, 1], xLabel: 'recall', yLabel: 'precision', fx: v => vn(v, 1), fy: v => vn(v, 1) });
    plot.line(PR.pts, C.amber, { width: 2.4, glow: 10 });
    plot.line([[0, POS / (POS + NEG)], [1, POS / (POS + NEG)]], alpha(C.white, 0.3), { width: 1, dash: [4, 4] });
    g.text('đường ngẫu nhiên = tỷ lệ lớp dương', plot.box.x + plot.box.w - 6, plot.py(POS / (POS + NEG)) - 8, { size: 10.5, color: C.muted, align: 'right' });
    const m = metrics(0.5);
    barChart(f, right, 'Precision ở ngưỡng 0,5 theo tỷ lệ bệnh π', PREVALENCES.map(v => (v === POS / (POS + NEG) ? 'tập test' : `${vn(v * 100, v < 0.01 ? 1 : 0)}%`)), PREVALENCES.map(v => precisionAt(m.recall, m.specificity, v) * 100), { max: 110, highlight: state.prevalence, reveal: seg(t, 3.5, 6), format: v => `${vn(v, 0)}%`, accent: C.red });
  },
  metrics: () => { const m = metrics(0.5); return [{ label: 'Precision (tập test)', value: pct(m.precision), tone: 'amber' }, { label: 'Precision khi π = 1%', value: pct(precisionAt(m.recall, m.specificity, 0.01)), tone: 'red' }]; },
};

export const initClassifierMetricsLab = () => glassLab('classifier-metrics', [chapter1, chapter2, chapter3, chapter4, chapter5], lab => {
  lab.range('threshold', value => vn(value / 100, 2), value => { state.threshold = value / 100; });
  lab.range('prevalence', value => (PREVALENCES[value] === POS / (POS + NEG) ? 'tập test' : `${vn(PREVALENCES[value] * 100, PREVALENCES[value] < 0.01 ? 1 : 0)}%`), value => { state.prevalence = value; });
  lab.on('[data-autoplay]', 'click', () => { state.threshold = null; lab.replay(); });
});
