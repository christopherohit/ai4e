// Quantization lab: number formats and memory, rounding a real weight row onto a 2^b-level grid, why outliers make one
// scale per matrix fail and per-channel / per-group scales fix it, measured perplexity and answers of the quantized
// model, and why fewer bytes per weight mean faster decoding.
import { glassLab, type Scene } from '../../_shared/glass-lab/engine';
import { C, Plot, alpha, seg, split, vn } from '../../_shared/glass-lab/gfx';
import { barChart } from '../../_shared/glass-lab/views';
import { BASE, FORMATS, PRECISIONS, Q, SIZES, decodeBound, quantizeVector, rmse } from './model';

export const explainerKind = 'measured-replay';

const state = { bits: 4, group: 0 as 0 | 128 | 32, config: null as number | null };
let currentT = 0;
const GROUPS = { 0: 'cả hàng (1 thang đo)', 128: 'nhóm 128', 32: 'nhóm 32' } as const;

// ---------- Chapter 1: formats and memory ----------
const chapter1: Scene = {
  title: 'Một trọng số cần bao nhiêu bit?',
  duration: 10,
  beats: () => [
    { at: 0, text: 'Mô hình được huấn luyện ở FP32 hoặc BF16: mỗi trọng số 32 hoặc 16 bit, gồm bit dấu, số mũ (phạm vi) và phần định trị (độ chính xác).' },
    { at: 3, text: 'Lượng tử hóa lưu trọng số bằng ít bit hơn: số nguyên 8 bit, 4 bit, hoặc số thực 8 bit (FP8), kèm một thang đo để đổi về giá trị thật.' },
    { at: 6, text: 'Bộ nhớ chỉ cho trọng số: mô hình 70B ở FP16 cần 140 GB, quá một GPU 80 GB; ở INT4 chỉ còn khoảng 36 GB.' },
  ],
  formula: 'bộ nhớ trọng số = số tham số × bit mỗi trọng số / 8\n70 tỷ × 16 bit = 140 GB · 70 tỷ × 4,125 bit (INT4 nhóm 128) ≈ 36 GB',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.25, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, 'Bố cục bit của các định dạng', C.violet);
    const unit = (left.w - 150) / 32, rowH = Math.min(46, (left.h - 70) / FORMATS.length);
    FORMATS.forEach((fm, k) => {
      const y = left.y + 52 + k * rowH, on = seg(t, k * 0.5, k * 0.5 + 1);
      g.text(fm.name, left.x + 16, y + rowH / 2 - 4, { size: 12.5, color: C.ink, weight: 650, alpha: on });
      let x = left.x + 120;
      const parts: [number, string, string][] = fm.exponent ? [[fm.sign, C.red, 'dấu'], [fm.exponent, C.amber, 'mũ'], [fm.mantissa, C.cyan, 'định trị']] : [[fm.mantissa, C.lime, 'số nguyên']];
      for (const [n, color, label] of parts) {
        for (let b = 0; b < n; b++) { g.ctx.fillStyle = alpha(color, 0.75 * on); g.ctx.fillRect(x + b * unit + 0.5, y + 4, unit - 1, rowH - 18); }
        if (n * unit > 34 && on > 0.5) g.text(label, x + n * unit / 2, y + rowH - 6, { size: 9.5, color: C.muted, align: 'center' });
        x += n * unit;
      }
      g.text(`${fm.sign + fm.exponent + fm.mantissa} bit`, x + 8, y + rowH / 2 - 4, { size: 11, color: C.muted, alpha: on });
    });
    const labels = SIZES.flatMap(s => PRECISIONS.map(p => `${s.name} ${p.name}`));
    const values = SIZES.flatMap(s => PRECISIONS.map(p => s.params * p.bits / 8 / 1e9));
    barChart(f, right, 'Bộ nhớ trọng số (GB)', labels, values, { reveal: seg(t, 6, 8), format: v => vn(v, 0), colors: [C.violet, C.cyan, C.lime, C.violet, C.cyan, C.lime], accent: C.amber });
  },
  metrics: () => [{ label: '70B FP16', value: '140 GB', tone: 'red' }, { label: '70B INT4', value: '≈36 GB', tone: 'lime' }],
};

// ---------- Chapter 2: rounding a real weight row ----------
const ROW = Q.sampleRow;
const quantNow = () => quantizeVector(ROW, state.bits, state.group || ROW.length);
const chapter2: Scene = {
  title: 'Làm tròn một hàng trọng số thật',
  duration: 12,
  idle: false,
  beats: () => [
    { at: 0, text: `512 trọng số đầu của một hàng thật trong ${Q.sampleName.replace('model.', '')} của Qwen2.5-0.5B. Phần lớn rất gần 0, vài giá trị lớn hơn hẳn.` },
    { at: 3, text: `Lượng tử hóa đối xứng ${state.bits} bit: chia cho thang đo s = max|w| / ${2 ** (state.bits - 1) - 1}, làm tròn về số nguyên trong [−${2 ** (state.bits - 1)}, ${2 ** (state.bits - 1) - 1}], rồi nhân lại với s.` },
    { at: 6.5, text: 'Các vạch dọc là các giá trị biểu diễn được. Mọi trọng số bị kéo về vạch gần nhất; khoảng cách giữa vạch là sai số tối đa.' },
    { at: 9.5, text: 'Thang đo theo giá trị lớn nhất, nên một ngoại lai làm vạch thưa ra và các trọng số nhỏ (phần đông) đều bị làm tròn về 0. Thử nhóm 32 ở bảng điều khiển.' },
  ],
  formula: () => { const q = quantNow(); return `ŵ = s · clamp(round(w / s), −2^(b−1), 2^(b−1) − 1),  s = max|w| / (2^(b−1) − 1)\nb = ${state.bits}, ${GROUPS[state.group]}: RMSE = ${vn(rmse(ROW, q.out) * 1000, 3)}·10⁻³, ${ROW.filter((v, i) => q.out[i] === 0 && v !== 0).length}/512 trọng số thành 0`; },
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [top, bottom] = split(f.safe, narrow ? [1, 1] : [1.1, 1], true, 12);
    const q = quantNow(), max = Math.max(...ROW.map(Math.abs)) * 1.05;
    g.panel(top.x, top.y, top.w, top.h, `Phân bố 512 trọng số và lưới ${2 ** state.bits} mức (${GROUPS[state.group]})`, C.violet);
    const plot = new Plot(g, top, [-max, max], [0, 1], { l: 30, r: 20, t: 40, b: 30 });
    plot.axes({ xTicks: [-max * 0.75, -max * 0.375, 0, max * 0.375, max * 0.75], yTicks: [], fx: v => vn(v, 3) });
    // histogram
    const bins = 80, counts = new Array(bins).fill(0);
    ROW.forEach(v => { counts[Math.min(bins - 1, Math.floor((v + max) / (2 * max) * bins))]++; });
    const peak = Math.max(...counts), show = seg(t, 0.3, 2.5);
    counts.forEach((c, i) => { const x0 = plot.px(-max + 2 * max * i / bins), x1 = plot.px(-max + 2 * max * (i + 1) / bins); g.ctx.fillStyle = alpha(C.cyan, 0.7); g.ctx.fillRect(x0, plot.py(0) - (plot.py(0) - plot.py(0.92)) * c / peak * show, x1 - x0 - 1, (plot.py(0) - plot.py(0.92)) * c / peak * show); });
    // grid of representable values (first group's scale)
    if (t > 3) {
      const s = q.scales[0], qmax = 2 ** (state.bits - 1) - 1;
      for (let k = -qmax - 1; k <= qmax; k++) { const x = plot.px(k * s); if (x < plot.box.x || x > plot.box.x + plot.box.w) continue; g.line([[x, plot.box.y + 4], [x, plot.box.y + plot.box.h]], alpha(k === 0 ? C.white : C.amber, (k === 0 ? 0.75 : 0.55) * seg(t, 3, 4)), { width: 1.2 }); }
    }
    // bottom: original vs quantized values for the first 128 weights
    g.panel(bottom.x, bottom.y, bottom.w, bottom.h, '128 trọng số đầu: gốc (xanh) và sau lượng tử hóa (cam)', C.amber);
    const bp = new Plot(g, bottom, [0, 127], [-max, max], { l: 44, r: 16, t: 40, b: 24 });
    bp.axes({ yTicks: [-max * 0.6, 0, max * 0.6], fy: v => vn(v, 3) });
    const reveal = seg(t, 6.5, 8.5);
    for (let i = 0; i < 128; i++) {
      const x = bp.px(i);
      g.line([[x, bp.py(0)], [x, bp.py(ROW[i])]], alpha(C.cyan, 0.5), { width: 2 });
      if (reveal > 0) g.circle(x, bp.py(ROW[i] + (q.out[i] - ROW[i]) * reveal), 2.2, C.amber);
    }
  },
  metrics: () => { const q = quantNow(); return [{ label: 'Bit', value: String(state.bits), tone: 'violet' }, { label: 'Thang đo', value: GROUPS[state.group], tone: 'cyan' }, { label: 'RMSE', value: `${vn(rmse(ROW, q.out) * 1000, 2)}·10⁻³`, tone: 'amber' }, { label: 'Bị làm tròn về 0', value: `${ROW.filter((v, i) => q.out[i] === 0 && v !== 0).length}/512`, tone: 'red' }]; },
};

// ---------- Chapter 3: outliers and granularity ----------
const RATIOS = Q.layerStats.map(l => l.absMax / l.std);
const chapter3: Scene = {
  title: 'Ngoại lai và độ mịn của thang đo',
  duration: 11,
  beats: () => [
    { at: 0, text: `Tỷ lệ max|w| / độ lệch chuẩn của ${Q.layerStats.length} ma trận trong Qwen2.5-0.5B: trung vị ${vn(RATIOS.slice().sort((a, b) => a - b)[Math.floor(RATIOS.length / 2)], 0)}, lớn nhất ${vn(Math.max(...RATIOS), 0)}. Phân phối có đuôi rất dài.` },
    { at: 3.5, text: 'Với một thang đo cho cả ma trận (per-tensor), ngoại lai lớn nhất quyết định khoảng cách vạch cho hàng triệu trọng số còn lại.' },
    { at: 6.5, text: 'Per-channel: mỗi hàng một thang đo. Per-group: mỗi 32 hoặc 128 trọng số liền nhau một thang đo (GPTQ, AWQ, GGUF). Ngoại lai chỉ làm hỏng nhóm của nó.' },
    { at: 9, text: 'Chi phí: thêm một thang đo 16 bit cho mỗi nhóm, tức 0,5 bit/trọng số với nhóm 32 và 0,125 bit với nhóm 128.' },
  ],
  formula: 'bit thực tế = b + 16 / kích thước nhóm  (INT4 nhóm 128 = 4,125 bit; nhóm 32 = 4,5 bit)',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.3, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, 'max|w| / σ của từng ma trận (theo lớp)', C.violet);
    const plot = new Plot(g, left, [0, RATIOS.length], [0, Math.max(...RATIOS) * 1.08], { l: 44, r: 16, t: 40, b: 30 });
    plot.axes({ yTicks: [0, 50, 100, 150, 200].filter(v => v < Math.max(...RATIOS) * 1.08), xLabel: 'ma trận (24 lớp × 7 phép chiếu)' });
    const show = seg(t, 0.3, 3);
    RATIOS.forEach((r, i) => { if (i / RATIOS.length > show) return; const name = Q.layerStats[i].name, color = name.includes('mlp') ? C.amber : C.cyan; g.ctx.fillStyle = alpha(color, 0.8); g.ctx.fillRect(plot.px(i), plot.py(r), Math.max(1, plot.px(1) - plot.px(0) - 0.5), plot.py(0) - plot.py(r)); });
    g.text('xanh: attention · cam: MLP', left.x + left.w - 16, left.y + 50, { size: 11, color: C.muted, align: 'right' });
    const four = Q.results.filter(r => r.bits === 4);
    barChart(f, right, 'Sai số trọng số INT4 theo độ mịn (RMSE ×10⁻³)', four.map(r => String(r.granularity).replace('tensor', 'tensor').replace('channel', 'kênh')), four.map(r => r.weightRmse * 1000), { reveal: seg(t, 6.5, 8.5), format: v => vn(v, 2), colors: [C.red, C.amber, C.lime, C.lime], accent: C.amber });
  },
  metrics: () => [{ label: 'max|w|/σ lớn nhất', value: vn(Math.max(...RATIOS), 0), tone: 'red' }, { label: 'INT4 per-tensor RMSE', value: `${vn(Q.results.find(r => r.name === 'int4 per-tensor')!.weightRmse * 1000, 2)}·10⁻³`, tone: 'red' }, { label: 'INT4 nhóm 32', value: `${vn(Q.results.find(r => r.name === 'int4 group-32')!.weightRmse * 1000, 2)}·10⁻³`, tone: 'lime' }],
  table: () => ({ headers: ['Ma trận', 'max|w|', 'σ', 'max/σ', 'Kurtosis'], rows: Q.layerStats.map(l => [l.name, vn(l.absMax, 3), vn(l.std, 4), vn(l.absMax / l.std, 0), vn(l.kurtosis, 1)]) }),
};

// ---------- Chapter 4: measured quality ----------
const CONFIG_LABEL = (name: string) => name.replace('per-tensor', '1 thang đo').replace('per-channel', 'theo kênh').replace('group-', 'nhóm ');
const configNow = () => state.config ?? Math.min(Q.results.length - 1, Math.floor(seg(currentT, 1, 11) * Q.results.length));
const chapter4: Scene = {
  title: 'Mô hình còn dùng được không?',
  duration: 13,
  idle: false,
  beats: () => [
    { at: 0, text: `Đo perplexity của Qwen2.5-0.5B trên ${vn(Q.evalTokens, 0)} token WikiText-2 sau khi lượng tử hóa mọi ma trận trong 24 lớp. Càng thấp càng tốt; FP32 = ${vn(BASE.perplexity, 2)}.` },
    { at: 3, text: 'INT8 theo kênh gần như không đổi. INT4 với một thang đo cho cả ma trận làm mô hình sụp đổ hoàn toàn.' },
    { at: 6.5, text: 'Cùng 4 bit, chỉ đổi độ mịn thang đo: theo kênh, nhóm 128 rồi nhóm 32, perplexity về dần mức gốc.' },
    { at: 9.5, text: 'Dưới 4 bit, cách làm tròn đơn giản (RTN) không còn đủ: cần GPTQ, AWQ hoặc huấn luyện có lượng tử hóa.' },
  ],
  formula: () => { const r = Q.results[configNow()]; return `${CONFIG_LABEL(r.name)}: ${vn(r.bitsPerWeight, 3)} bit/trọng số, perplexity ${r.perplexity > 1000 ? vn(r.perplexity, 0) : vn(r.perplexity, 2)} (FP32 ${vn(BASE.perplexity, 2)})`; },
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.2, 1] : [1.3, 1], narrow);
    const cur = configNow();
    barChart(f, left, 'Perplexity WikiText-2 (log₁₀, thấp hơn là tốt)', Q.results.map(r => CONFIG_LABEL(r.name).replace('int', 'I').replace(' 1 thang đo', '·T').replace(' theo kênh', '·K').replace(' nhóm ', '·g').replace('fp32', 'FP32')), Q.results.map(r => Math.log10(r.perplexity)), { max: 8.2, highlight: cur, reveal: seg(t, 0.5, 3), format: v => (10 ** v > 1000 ? `${vn(10 ** v / 1e6, 1)}M` : vn(10 ** v, 1)), colors: Q.results.map((r, i) => (i === cur ? C.amber : r.perplexity < BASE.perplexity * 1.3 ? C.lime : r.perplexity < 100 ? C.cyan : C.red)) });
    const r = Q.results[cur];
    g.panel(right.x, right.y, right.w, right.h, `Hỏi: “${Q.prompt}”`, C.cyan);
    g.text(CONFIG_LABEL(r.name), right.x + 16, right.y + 54, { size: 13, color: C.amber, weight: 650 });
    const words = r.continuation.replace(/\s+/g, ' ').trim().split(' ');
    let line = '', y = right.y + 82;
    for (const w of words) { const next = line ? `${line} ${w}` : w; if (g.measure(next, 13) > right.w - 32) { g.text(line, right.x + 16, y, { size: 13, color: C.ink }); line = w; y += 20; } else line = next; }
    if (line) g.text(line, right.x + 16, y, { size: 13, color: C.ink });
  },
  metrics: () => { const r = Q.results[configNow()]; return [{ label: 'Cấu hình', value: CONFIG_LABEL(r.name), tone: 'violet' }, { label: 'Bit/trọng số', value: vn(r.bitsPerWeight, 3), tone: 'cyan' }, { label: 'Perplexity', value: r.perplexity > 1000 ? `${vn(r.perplexity / 1e6, 1)} triệu` : vn(r.perplexity, 2), tone: r.perplexity < BASE.perplexity * 1.3 ? 'lime' : 'red' }]; },
  table: () => ({ headers: ['Cấu hình', 'Bit/trọng số', 'Perplexity', 'RMSE trọng số', 'Trả lời'], rows: Q.results.map(r => [CONFIG_LABEL(r.name), vn(r.bitsPerWeight, 3), vn(r.perplexity, 2), vn(r.weightRmse, 6), r.continuation.replace(/\s+/g, ' ').slice(0, 90)]) }),
};

// ---------- Chapter 5: speed ----------
const chapter5: Scene = {
  title: 'Ít byte hơn, decode nhanh hơn',
  duration: 10,
  beats: () => [
    { at: 0, text: 'Decode ở batch nhỏ bị giới hạn bởi băng thông bộ nhớ: mỗi token phải đọc toàn bộ trọng số một lần.' },
    { at: 3, text: 'Tốc độ tối đa ≈ băng thông / số byte trọng số. Với H100 (3,35 TB/s), mô hình 7B: FP16 khoảng 239 token/s, INT4 khoảng 928 token/s.' },
    { at: 6.5, text: 'Lượng tử hóa chỉ trọng số (W4A16) tăng tốc decode; muốn tăng tốc prefill (compute-bound) cần lượng tử hóa cả activation (W8A8, FP8) để dùng tensor core số nguyên hoặc FP8.' },
  ],
  formula: 'token/s (batch 1) ≤ băng thông HBM / (số tham số × bit / 8)',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, [1, 1], narrow);
    barChart(f, left, 'Giới hạn trên token/s, 7B trên H100', PRECISIONS.map(p => p.name), PRECISIONS.map(p => decodeBound(7e9, p.bits)), { reveal: seg(t, 3, 5), format: v => vn(v, 0), colors: [C.violet, C.cyan, C.lime], accent: C.lime });
    barChart(f, right, 'Giới hạn trên token/s, 70B trên H100', PRECISIONS.map(p => p.name), PRECISIONS.map(p => decodeBound(70e9, p.bits)), { reveal: seg(t, 4, 6), format: v => vn(v, 0), colors: [C.violet, C.cyan, C.lime], accent: C.lime });
  },
  metrics: () => [{ label: '7B FP16', value: `${vn(decodeBound(7e9, 16), 0)} token/s`, tone: 'violet' }, { label: '7B INT4', value: `${vn(decodeBound(7e9, 4.125), 0)} token/s`, tone: 'lime' }],
};

export const initQuantizationLab = () => glassLab('quantization', [chapter1, chapter2, chapter3, chapter4, chapter5], lab => {
  lab.range('bits', value => `${value} bit`, value => { state.bits = value; });
  lab.on('[data-group]', 'click', el => { state.group = Number(el.dataset.group) as 0 | 128 | 32; document.querySelectorAll<HTMLElement>('[data-group]').forEach(b => b.setAttribute('aria-pressed', String(b === el))); lab.invalidate(); });
  lab.range('config', value => CONFIG_LABEL(Q.results[value].name), value => { state.config = value; });
});

