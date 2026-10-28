// Mixture-of-Experts lab: dense vs sparse parameter counts, a trained router choosing 2 of 8 experts for real digit
// images, which expert specialises in which digit, what happens without the load-balancing loss, and why MoE trades
// memory for compute at inference time.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, alpha, seg, split, vn, type Rect } from '../../_shared/glass-lab/gfx';
import { barChart, categoryChart, lossChart } from '../../_shared/glass-lab/views';
import { MODELS, MOE, touched } from './model';

export const explainerKind = 'educational-simulation';

const EXPERT = ['#4fd6ff', '#7af0a0', '#ffb648', '#a98bff', '#ff5fa8', '#5b8cff', '#ff9e6b', '#c8f06b'];
const state = { sample: 3 };
let currentT = 0;

function digit(f: Frame, x: number, y: number, size: number, pixels: number[]) {
  const c = size / 8;
  pixels.forEach((v, i) => { const k = v / 16; f.g.ctx.fillStyle = `rgb(${Math.round(10 + 235 * k)},${Math.round(14 + 236 * k)},${Math.round(30 + 225 * k)})`; f.g.ctx.fillRect(x + (i % 8) * c, y + Math.floor(i / 8) * c, c + 0.5, c + 0.5); });
  f.g.ctx.strokeStyle = 'rgba(255,255,255,0.2)'; f.g.ctx.strokeRect(x, y, size, size);
}
function node(f: Frame, x: number, y: number, w: number, h: number, label: string, color: string, on: number, sub = '') {
  const { g } = f;
  g.ctx.save(); g.ctx.globalAlpha = 0.35 + 0.65 * on; g.ctx.fillStyle = alpha(color, 0.08 + 0.25 * on); g.ctx.strokeStyle = alpha(color, 0.4 + 0.6 * on); g.ctx.lineWidth = 1.4;
  if (on > 0.5) { g.ctx.shadowColor = color; g.ctx.shadowBlur = 16; }
  g.ctx.beginPath(); g.ctx.roundRect(x, y, w, h, 8); g.ctx.fill(); g.ctx.stroke(); g.ctx.restore();
  g.text(label, x + w / 2, y + h / 2 - (sub ? 6 : 0), { size: 12, color: C.ink, align: 'center', weight: 650, alpha: 0.4 + 0.6 * on });
  if (sub) g.text(sub, x + w / 2, y + h / 2 + 9, { size: 10.5, color: C.muted, align: 'center' });
}

// ---------- Chapter 1: dense vs sparse ----------
const chapter1: Scene = {
  title: 'Dense và thưa: tham số tổng và tham số kích hoạt',
  duration: 10,
  beats: () => [
    { at: 0, text: 'Trong mô hình dense, mọi tham số tham gia xử lý mọi token. Muốn mô hình “biết” nhiều hơn thì mỗi token cũng tốn nhiều phép tính hơn.' },
    { at: 2.5, text: 'Mixture of Experts thay khối FFN bằng nhiều FFN nhỏ (expert) và một router chọn vài expert cho mỗi token.' },
    { at: 5, text: 'Kết quả: tổng tham số rất lớn nhưng mỗi token chỉ dùng một phần nhỏ. DeepSeek-V3 có 671 tỷ tham số, mỗi token kích hoạt 37 tỷ (5,5%).' },
    { at: 8, text: 'Biểu đồ phải: tỷ lệ tham số kích hoạt của các mô hình MoE công bố, từ 4,3% tới 28%.' },
  ],
  formula: 'FLOP mỗi token ∝ tham số kích hoạt,  bộ nhớ ∝ tổng tham số\nMixtral: 8 expert, chọn 2 → 12,9 / 46,7 tỷ tham số mỗi token',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.3, 1], narrow);
    categoryChart(f, left, 'Tham số (tỷ, thang log): tổng và kích hoạt mỗi token', MODELS.map(m => m.name.replace('Qwen3-', 'Qwen3 ').replace(' Maverick', '')), [
      { label: 'tổng', values: MODELS.map(m => Math.log10(m.total)), color: C.violet, format: v => vn(10 ** v, 0) },
      { label: 'kích hoạt', values: MODELS.map(m => Math.log10(m.active)), color: C.lime, format: v => vn(10 ** v, 1) },
    ], [0, 3], null, seg(t, 2.5, 6), v => vn(10 ** v, 0));
    barChart(f, right, 'Tỷ lệ kích hoạt mỗi token', MODELS.map(m => m.name.split(' ')[0].replace('Qwen3-30B-A3B', 'Qwen3-30B').replace('Qwen3-235B-A22B', 'Qwen3-235B')), MODELS.map(m => m.active / m.total * 100), { max: 32, reveal: seg(t, 7.5, 9.5), format: v => `${vn(v, 1)}%`, color: C.lime, accent: C.lime });
  },
  metrics: () => [{ label: 'DeepSeek-V3', value: '37 / 671 tỷ', tone: 'lime' }, { label: 'Mixtral', value: 'top-2 / 8', tone: 'violet' }],
  table: () => ({ headers: ['Mô hình', 'Tổng (tỷ)', 'Kích hoạt (tỷ)', 'Expert', 'Chọn mỗi token', 'Tỷ lệ'], rows: MODELS.map(m => [m.name, vn(m.total, 1), vn(m.active, 1), m.experts, m.topK, `${vn(m.active / m.total * 100, 1)}%`]) }),
};

// ---------- Chapter 2: the router ----------
const chapter2: Scene = {
  title: 'Router chọn 2 trong 8 expert',
  duration: 12,
  beats: () => {
    const s = MOE.balanced.samples[state.sample];
    return [
      { at: 0, text: `Một ảnh chữ số ${s.label} thật (8×8 điểm ảnh) đi vào lớp MoE đã huấn luyện. Ở LLM, mỗi “ảnh” này là hidden state của một token.` },
      { at: 2, text: 'Router là một lớp tuyến tính nhỏ: cho ra 8 điểm, qua softmax thành xác suất cho từng expert.' },
      { at: 5, text: `Chỉ giữ top-2: expert ${s.experts[0] + 1} (${vn(s.routerProbs[s.experts[0]] * 100, 0)}%) và expert ${s.experts[1] + 1} (${vn(s.routerProbs[s.experts[1]] * 100, 0)}%). Sáu expert còn lại không chạy.` },
      { at: 8, text: `Hai đầu ra được trộn theo trọng số chuẩn hóa ${vn(s.gates[0], 2)} và ${vn(s.gates[1], 2)}. Dự đoán: ${s.pred}${s.pred === s.label ? ' (đúng)' : ' (sai)'}.` },
    ];
  },
  formula: () => { const s = MOE.balanced.samples[state.sample]; return `g = softmax(W_r · x),  T = top-2(g)\ny = Σ_{i∈T} (gᵢ / Σ_{j∈T} gⱼ) · Expertᵢ(x) = ${vn(s.gates[0], 2)}·E${s.experts[0] + 1}(x) + ${vn(s.gates[1], 2)}·E${s.experts[1] + 1}(x)`; },
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const s = MOE.balanced.samples[state.sample], area = f.safe;
    g.panel(area.x, area.y, area.w, area.h, `Lớp MoE: ảnh chữ số ${s.label} → router → 2 expert`, C.violet);
    const size = Math.min(110, area.h * 0.3), dx = area.x + 24, dy = area.y + area.h / 2 - size / 2;
    digit(f, dx, dy, size, s.pixels);
    const rx = dx + size + (narrow ? 20 : 50), rw = narrow ? 70 : 100;
    node(f, rx, area.y + area.h / 2 - 26, rw, 52, 'Router', C.white, seg(t, 1.5, 2.5), 'W_r · x');
    g.arrow(dx + size + 4, area.y + area.h / 2, rx - 4, area.y + area.h / 2, C.muted, { width: 1.4 });
    // router probabilities as bars beside each expert
    const ex = rx + rw + (narrow ? 30 : 80), ew = narrow ? 90 : 140, eh = Math.min(40, (area.h - 80) / 8 - 6);
    const probsOn = seg(t, 2, 4.5), pick = seg(t, 5, 6.5), mix = seg(t, 8, 9.5);
    const ox = ex + ew + (narrow ? 30 : 70);
    for (let e = 0; e < 8; e++) {
      const y = area.y + 50 + e * (eh + 6), chosen = s.experts.includes(e);
      const on = chosen ? Math.max(0.3, pick) : 0.3 * (1 - pick) + 0.12;
      g.line([[rx + rw + 2, area.y + area.h / 2], [ex - 2, y + eh / 2]], alpha(EXPERT[e], (chosen ? 0.25 + 0.6 * pick : 0.25 * (1 - pick)) * probsOn), { width: chosen ? 1.8 : 1 });
      node(f, ex, y, ew, eh, `Expert ${e + 1}`, EXPERT[e], on);
      const p = s.routerProbs[e];
      g.ctx.fillStyle = alpha(EXPERT[e], 0.8); g.ctx.fillRect(ex + ew + 6, y + eh / 2 - 5, 60 * p * probsOn, 10);
      g.text(`${vn(p * 100, 0)}%`, ex + ew + 70 * Math.max(0.02, p * probsOn) + 4, y + eh / 2, { size: 11, color: chosen && pick > 0.5 ? C.ink : C.muted });
      if (chosen && mix > 0) g.line([[ex + ew + 2, y + eh / 2], [ox + 60, area.y + area.h / 2]], alpha(EXPERT[e], 0.7 * mix), { width: 2, glow: 8 });
    }
    if (mix > 0) {
      node(f, ox + 60, area.y + area.h / 2 - 30, narrow ? 80 : 120, 60, `Dự đoán: ${s.pred}`, s.pred === s.label ? C.lime : C.red, mix, `${vn(s.gates[0], 2)} · E${s.experts[0] + 1} + ${vn(s.gates[1], 2)} · E${s.experts[1] + 1}`);
    }
  },
  metrics: () => { const s = MOE.balanced.samples[state.sample]; return [{ label: 'Chữ số', value: String(s.label), tone: 'violet' }, { label: 'Expert chọn', value: s.experts.map(e => e + 1).join(' + '), tone: 'amber' }, { label: 'Dự đoán', value: String(s.pred), tone: s.pred === s.label ? 'lime' : 'red' }]; },
  table: () => ({ headers: ['Chữ số', ...Array.from({ length: 8 }, (_, e) => `E${e + 1}`), 'Chọn', 'Dự đoán'], rows: MOE.balanced.samples.map(s => [s.label, ...s.routerProbs.map(p => `${vn(p * 100, 0)}%`), s.experts.map(e => e + 1).join('+'), s.pred]) }),
};

// ---------- Chapter 3: specialisation ----------
function heatmap(f: Frame, rect: Rect, heat: number[][], title: string, accent: string, reveal = 1) {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, title, accent);
  const cw = Math.min((rect.w - 70) / 8, (rect.h - 80) / 10), ox = rect.x + 44, oy = rect.y + 56;
  heat.forEach((row, d) => row.forEach((v, e) => {
    if ((d * 8 + e) / 80 > reveal) return;
    g.ctx.fillStyle = alpha(EXPERT[e], Math.min(1, v * 2));
    g.ctx.fillRect(ox + e * cw + 1, oy + d * cw + 1, cw - 2, cw - 2);
    if (v >= 0.2 && cw > 26) g.text(vn(v * 100, 0), ox + e * cw + cw / 2, oy + d * cw + cw / 2, { size: 10, color: '#081022', align: 'center', weight: 700 });
  }));
  for (let d = 0; d < 10; d++) g.text(String(d), ox - 10, oy + d * cw + cw / 2, { size: 12, color: C.ink, align: 'right', weight: 650 });
  for (let e = 0; e < 8; e++) g.text(`E${e + 1}`, ox + e * cw + cw / 2, oy - 10, { size: 11, color: EXPERT[e], align: 'center', weight: 650 });
}
const chapter3: Scene = {
  title: 'Expert nào lo chữ số nào?',
  duration: 10,
  beats: () => [
    { at: 0, text: 'Đếm trên 540 ảnh kiểm tra: mỗi chữ số được gửi tới expert nào, bao nhiêu phần trăm (mỗi hàng cộng lại 100%, vì mỗi ảnh chọn 2 expert).' },
    { at: 3, text: 'Không ai gán nhãn cho expert. Router tự học gửi chữ số 0 tới expert 3 và 7, chữ số 4 tới expert 6 và 8: các expert chuyên môn hóa.' },
    { at: 6.5, text: 'Ở LLM, chuyên môn hóa thường theo kiểu token (dấu câu, số, mã nguồn, một ngôn ngữ) hơn là theo chủ đề lớn như “toán” hay “lịch sử”.' },
  ],
  formula: 'ô (chữ số d, expert e) = tỷ lệ lượt chọn expert e trong các ảnh chữ số d',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.2, 1] : [1.1, 1], narrow);
    heatmap(f, left, MOE.balanced.heat, 'Chuyên môn hóa (có loss cân bằng tải)', C.violet, seg(t, 0.3, 3));
    const s = MOE.balanced.samples;
    f.g.panel(right.x, right.y, right.w, right.h, 'Mỗi chữ số và 2 expert router chọn', C.amber);
    const cell = Math.min(54, (right.h - 70) / 5 - 8), cols = 2;
    s.forEach((smp, i) => {
      const x = right.x + 20 + (i % cols) * (right.w / 2 - 10), y = right.y + 50 + Math.floor(i / cols) * (cell + 8);
      digit(f, x, y, cell, smp.pixels);
      smp.experts.forEach((e, k) => { f.g.ctx.fillStyle = EXPERT[e]; f.g.ctx.fillRect(x + cell + 10 + k * 40, y + cell / 2 - 10, 34, 20); f.g.text(`E${e + 1}`, x + cell + 27 + k * 40, y + cell / 2, { size: 11, color: '#081022', align: 'center', weight: 700 }); });
    });
  },
  metrics: () => [{ label: 'Độ chính xác MoE', value: `${vn(MOE.balanced.accuracy * 100, 1)}%`, tone: 'lime' }, { label: 'Expert', value: `${MOE.experts}, chọn ${MOE.topK}`, tone: 'violet' }],
};

// ---------- Chapter 4: load balancing ----------
const chapter4: Scene = {
  title: 'Cân bằng tải: tránh expert chết',
  duration: 12,
  beats: () => [
    { at: 0, text: 'Router học bằng gradient, nên dễ rơi vào vòng lặp: expert được chọn nhiều thì học tốt hơn, nên lại càng được chọn nhiều.' },
    { at: 3, text: `Huấn luyện không có loss cân bằng: expert 3 không được chọn lần nào (0%), expert 4 gánh ${vn(MOE.unbalanced.history.at(-1)!.usage[3] * 100, 0)}% lượt.` },
    { at: 6.5, text: 'Loss phụ của Switch Transformer phạt khi phần lượt chọn fᵢ và xác suất trung bình Pᵢ dồn vào ít expert. Với α = 0,02, mọi expert nhận 10–17%.' },
    { at: 9.5, text: 'Cân bằng tải quan trọng cho tốc độ: mỗi expert thường nằm trên một GPU khác nhau (expert parallelism), một expert quá tải sẽ làm cả batch chờ.' },
  ],
  formula: 'L_aux = α · E · Σᵢ fᵢ · Pᵢ   (fᵢ: phần lượt chọn expert i, Pᵢ: xác suất router trung bình cho i)\ntối thiểu khi fᵢ = Pᵢ = 1/E',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1, 1.1], narrow);
    const [a, b] = split(left, [1, 1], true, 12);
    const labels = Array.from({ length: 8 }, (_, e) => `E${e + 1}`);
    barChart(f, a, `Không có loss cân bằng · ${vn(MOE.unbalanced.accuracy * 100, 1)}%`, labels, MOE.unbalanced.history.at(-1)!.usage.map(v => v * 100), { max: 35, colors: EXPERT, reveal: seg(t, 2.5, 4.5), format: v => `${vn(v, 0)}%`, accent: C.red });
    barChart(f, b, `α = 0,02 · ${vn(MOE.balanced.accuracy * 100, 1)}%`, labels, MOE.balanced.history.at(-1)!.usage.map(v => v * 100), { max: 35, colors: EXPERT, reveal: seg(t, 6.5, 8.5), format: v => `${vn(v, 0)}%`, accent: C.lime });
    const h = MOE.unbalanced.history;
    lossChart(f, right, Array.from({ length: 8 }, (_, e) => ({ values: h.map(x => x.usage[e] * 100), color: EXPERT[e] })), h.length - 1, 40, null, 'Phần lượt chọn từng expert theo epoch (không cân bằng, %)', 'epoch', h.length - 1);
  },
  metrics: () => [{ label: 'Expert chết', value: String(MOE.unbalanced.history.at(-1)!.usage.filter(u => u === 0).length), tone: 'red' }, { label: 'Có cân bằng: min', value: `${vn(Math.min(...MOE.balanced.history.at(-1)!.usage) * 100, 0)}%`, tone: 'lime' }],
};

// ---------- Chapter 5: inference cost ----------
const BATCHES = [1, 2, 4, 8, 16, 32, 64, 128];
const chapter5: Scene = {
  title: 'Suy luận: ít phép tính, nhiều bộ nhớ',
  duration: 11,
  beats: () => [
    { at: 0, text: 'Mỗi token chỉ chạy vài expert, nên phép tính mỗi token như một mô hình nhỏ. Nhưng mọi expert vẫn phải nằm sẵn trong bộ nhớ GPU.' },
    { at: 3.5, text: 'Ở batch 1, mỗi bước decode chỉ đọc trọng số của các expert được chọn: nhanh như mô hình nhỏ.' },
    { at: 6.5, text: 'Batch lớn thì các token chọn khác nhau, gần như mọi expert đều bị chạm tới: Mixtral (2/8) chạm tới ~99% expert chỉ với 16 token; DeepSeek-V3 (8/256) cần khoảng 128 token.' },
    { at: 9, text: `Mô hình nhỏ minh họa: MoE ${vn(MOE.balanced.accuracy * 100, 1)}% với ${MOE.balanced.activeParams} tham số kích hoạt / ${MOE.balanced.totalParams} tổng; mạng dense cùng cỡ kích hoạt đạt ${vn(MOE.dense.accuracy * 100, 1)}%. Lợi ích của MoE xuất hiện ở quy mô lớn.` },
  ],
  formula: 'phần expert bị chạm trong một batch B ≈ 1 − (1 − k/E)^B',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.2, 1], narrow);
    categoryChart(f, left, 'Phần expert phải đọc trong một bước decode', BATCHES.map(String), [
      { label: 'Mixtral 2/8', values: BATCHES.map(b => touched(2, 8, b)), color: C.amber, format: v => `${vn(v * 100, 0)}%` },
      { label: 'Qwen3 8/128', values: BATCHES.map(b => touched(8, 128, b)), color: C.cyan, format: v => `${vn(v * 100, 0)}%` },
      { label: 'DeepSeek-V3 8/256', values: BATCHES.map(b => touched(8, 256, b)), color: C.lime, format: v => `${vn(v * 100, 0)}%` },
    ], [0, 1], null, seg(t, 3.5, 7), v => `${vn(v * 100, 0)}%`, 'batch (token)');
    barChart(f, right, 'Mixtral 8×7B: tính toán và bộ nhớ', ['FLOP/token (tỷ tham số)', 'Bộ nhớ fp16 (GB)'], [12.9, 46.7 * 2], { colors: [C.lime, C.violet], reveal: seg(t, 0.5, 3), format: v => vn(v, 1), accent: C.violet });
  },
  metrics: () => [{ label: 'Mixtral, batch 16', value: `${vn(touched(2, 8, 16) * 100, 1)}% expert`, tone: 'amber' }, { label: 'DeepSeek-V3, batch 16', value: `${vn(touched(8, 256, 16) * 100, 0)}% expert`, tone: 'lime' }],
};

export const initMoeLab = () => glassLab('moe', [chapter1, chapter2, chapter3, chapter4, chapter5], lab => {
  lab.on('[data-sample]', 'click', el => { state.sample = Number(el.dataset.sample); document.querySelectorAll<HTMLElement>('[data-sample]').forEach(b => b.setAttribute('aria-pressed', String(b === el))); lab.replay(); });
});
