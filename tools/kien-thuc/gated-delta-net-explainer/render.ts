// Gated DeltaNet lab: why a fixed-size state instead of a KV cache, the delta rule step by step on a 4×4 memory, how
// many associations a 32×32 state holds under each update rule, what the forget gate decides, and why hybrid models
// keep a few attention layers.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, alpha as fade, seg, split, vn, type Rect } from '../../_shared/glass-lab/gfx';
import { barChart, categoryChart, lossChart } from '../../_shared/glass-lab/views';
import { ANSWERS, CAPACITY, CHANNELS, COUNTS, D, TOKENS, capacity, gdnStateBytes, qwenMemory, read, simulate, topicSwitch } from './model';

export const explainerKind = 'educational-simulation';

const state = { alpha: 0.95, beta: 0.9, gate: 'switch' as 1 | 0.9 | 'switch' };
let currentT = 0;
const GB = (b: number) => b / 2 ** 30;
const CONTEXTS = [4096, 32768, 131072, 262144, 1048576];
const ctxLabel = (n: number) => (n >= 1048576 ? '1M' : `${n / 1024}K`);
const fmtGB = (v: number) => (v < 1 ? `${vn(v * 1024, 0)} MB` : `${vn(v, 1)} GB`);

// ---------- Chapter 1: KV cache vs fixed state ----------
const chapter1: Scene = {
  title: 'Bộ nhớ: KV cache lớn dần, trạng thái thì không',
  duration: 10,
  beats: () => {
    const m = qwenMemory(262144);
    return [
      { at: 0, text: 'Attention giữ Key và Value của mọi token đã đọc để đọc lại sau này: KV cache tăng tuyến tính theo độ dài ngữ cảnh.' },
      { at: 3, text: `Với cấu hình Qwen3.8-Flash-Next, nếu cả 48 layer đều là attention, một chuỗi 262K token cần ${fmtGB(GB(m.allAttention))} KV cache (bf16).` },
      { at: 5.5, text: `Gated DeltaNet nén lịch sử vào một ma trận trạng thái cố định: ${vn(gdnStateBytes / 2 ** 20, 1)} MB mỗi layer, dù đọc 1 nghìn hay 1 triệu token.` },
      { at: 8, text: `Lịch hybrid 36 GDN + 12 attention: ${fmtGB(GB(m.hybrid))} ở 262K, giảm gần 4 lần; phần GDN chỉ ${fmtGB(GB(m.gdnPart))} và không đổi.` },
    ];
  },
  formula: 'KV cache = 2 · layers · kv_heads · head_dim · 2 byte · T     trạng thái GDN = heads · d_k · d_v · 2 byte (không phụ thuộc T)',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.3, 1] : [1.6, 1], narrow);
    categoryChart(f, left, 'Bộ nhớ mỗi chuỗi (GB, thang log₁₀)', CONTEXTS.map(ctxLabel), [
      { label: '48 attention', values: CONTEXTS.map(n => Math.log10(GB(qwenMemory(n).allAttention))), color: C.red, format: v => fmtGB(10 ** v) },
      { label: 'hybrid 36 GDN + 12 attn', values: CONTEXTS.map(n => Math.log10(GB(qwenMemory(n).hybrid))), color: C.lime, format: v => fmtGB(10 ** v) },
      { label: 'riêng 36 trạng thái GDN', values: CONTEXTS.map(n => Math.log10(GB(qwenMemory(n).gdnPart))), color: C.cyan, format: v => fmtGB(10 ** v) },
    ], [-1.5, 2.2], null, seg(t, 1, 8.5), v => fmtGB(10 ** v), 'độ dài ngữ cảnh');
    const m = qwenMemory(262144);
    barChart(f, right, 'Ở 262K token', ['48 attention', 'hybrid', 'trạng thái GDN'], [GB(m.allAttention), GB(m.hybrid), GB(m.gdnPart)], { reveal: seg(t, 3, 9), format: fmtGB, colors: [C.red, C.lime, C.cyan], accent: C.lime });
  },
  metrics: () => { const m = qwenMemory(262144); return [{ label: '48 attention', value: fmtGB(GB(m.allAttention)), tone: 'red' }, { label: 'Hybrid', value: fmtGB(GB(m.hybrid)), tone: 'lime' }]; },
};

// ---------- Chapter 2: the delta rule on a 4×4 memory ----------
function matrix(f: Frame, rect: Rect, m: number[][], title: string, color: string, highlight = -1) {
  const { g } = f, cell = Math.min((rect.w - 70) / 4, (rect.h - 64) / 4), ox = rect.x + 60, oy = rect.y + 40;
  g.text(title, rect.x + rect.w / 2, rect.y + 20, { size: 12, color, align: 'center', weight: 650 });
  ANSWERS.forEach((a, j) => g.text(a, ox + (j + 0.5) * cell, oy - 6, { size: 10, color: C.muted, align: 'center' }));
  m.forEach((row, i) => {
    g.text(CHANNELS[i], ox - 6, oy + (i + 0.5) * cell, { size: 10, color: C.muted, align: 'right' });
    row.forEach((v, j) => {
      g.ctx.fillStyle = v >= 0 ? fade(C.cyan, Math.min(0.9, Math.abs(v))) : fade(C.red, Math.min(0.9, Math.abs(v)));
      g.ctx.fillRect(ox + j * cell + 2, oy + i * cell + 2, cell - 4, cell - 4);
      if (j === highlight) { g.ctx.strokeStyle = C.white; g.ctx.lineWidth = 1.5; g.ctx.strokeRect(ox + j * cell + 2, oy + i * cell + 2, cell - 4, cell - 4); }
      g.text(vn(v, 2), ox + (j + 0.5) * cell, oy + (i + 0.5) * cell, { size: Math.min(12, cell / 4), color: C.ink, align: 'center', mono: true });
    });
  });
}
const STEP_TIME = 1.2;
const chapter2: Scene = {
  title: 'Delta rule: chỉ ghi phần bộ nhớ đoán sai',
  duration: TOKENS.length * STEP_TIME + 4,
  beats: () => [
    { at: 0, text: 'Bộ nhớ là ma trận S (4 kênh key × 4 đáp án). Tám token lần lượt đi qua: mỗi token mang một key (câu hỏi) và một value (đáp án).' },
    { at: STEP_TIME * 1, text: 'Mỗi bước: làm mờ S bằng α, đọc dự đoán Sᵀk cho key hiện tại, tính sai số e = v − Sᵀk, rồi ghi β·k·eᵀ.' },
    { at: STEP_TIME * 4.2, text: 'Token 5 nói thủ đô nay là Hà Nội. Bộ nhớ đang đoán Huế, nên sai số âm ở cột Huế và dương ở cột Hà Nội: liên kết cũ bị sửa, không chỉ cộng thêm.' },
    { at: STEP_TIME * 6.2, text: 'Token 7 và 8 lặp lại thông tin đã biết: sai số gần 0 nên gần như không ghi gì. Linear attention vẫn cộng k·vᵀ mỗi lần.' },
  ],
  formula: () => `S̃ = α·S,   e = v − S̃ᵀk,   S = S̃ + β·k·eᵀ     (α = ${vn(state.alpha, 2)}, β = ${vn(state.beta, 2)})\nhỏi “thủ đô?”: GDN → ${ANSWERS[argmax(read(simulate(state.alpha, state.beta).at(-1)!.state, 'capital'))]}, linear attention → ${linearAnswer()}`,
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const steps = simulate(state.alpha, state.beta), k = Math.min(TOKENS.length - 1, Math.floor(t / STEP_TIME)), step = steps[k];
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, `Token ${k + 1}: ${step.token.text}`, C.violet);
    const [mL, mR] = split({ x: left.x + 8, y: left.y + 30, w: left.w - 16, h: left.h - 40 }, [1, 1], narrow);
    matrix(f, mL, step.state, 'Gated DeltaNet S', C.lime, argmax(step.token.v));
    matrix(f, mR, step.linear, 'Linear attention Σ k·vᵀ', C.red, argmax(step.token.v));
    barChart(f, right, 'Dự đoán trước khi ghi (Sᵀk) và sai số e', ANSWERS.flatMap(a => [`${a} dự đoán`, 'sai số']), ANSWERS.flatMap((_, j) => [step.prediction[j], step.error[j]]).map(v => Math.abs(v)), { max: 1, format: v => vn(v, 2), colors: ANSWERS.flatMap((_, j) => [C.cyan, step.error[j] < 0 ? C.red : C.lime]), accent: C.violet });
  },
  metrics: () => { const s = simulate(state.alpha, state.beta).at(-1)!; return [{ label: 'Sai số truy xuất GDN', value: vn(s.gdnError, 2), tone: 'lime' }, { label: 'Linear attention', value: vn(s.linearError, 2), tone: 'red' }]; },
};
function argmax(v: number[]) { return v.indexOf(Math.max(...v)); }
function linearAnswer() { const v = read(simulate(state.alpha, state.beta).at(-1)!.linear, 'capital'); const top = [...v.keys()].sort((a, b) => v[b] - v[a]); return v[top[1]] > 0.6 * v[top[0]] ? `${ANSWERS[top[0]]} lẫn ${ANSWERS[top[1]]}` : ANSWERS[top[0]]; }

// ---------- Chapter 3: capacity ----------
const RULE = { linear: ['linear attention (cộng dồn)', C.red], delta: ['delta rule', C.cyan], gated: ['gated delta (α = 0,97)', C.lime] } as const;
const chapter3: Scene = {
  title: 'Một trạng thái 32×32 nhớ được bao nhiêu?',
  duration: 11,
  beats: () => [
    { at: 0, text: `Thí nghiệm số: lần lượt ghi N cặp (key, value) ngẫu nhiên, mỗi vector ${D} chiều độ dài 1, vào một ma trận ${D}×${D}; cuối cùng đọc lại từng cặp.` },
    { at: 2.5, text: `Linear attention chỉ cộng: các cặp chồng lên nhau, sai số tăng mãi (N = 128: ${vn(CAPACITY[0].all.at(-1)!, 2)}, lớn hơn cả độ dài của value).` },
    { at: 5.5, text: `Delta rule trừ đi phần đã đoán trước khi ghi: các cặp mới nhất luôn đọc lại tốt (sai số ~${vn(CAPACITY[1].recent.at(-1)!, 2)}), các cặp cũ bị ghi đè dần thay vì làm nhiễu mọi thứ.` },
    { at: 8.5, text: 'Bộ nhớ cố định không thể nhớ mọi thứ khi N vượt xa số chiều: câu hỏi là quên cái gì. Delta rule quên cái cũ; linear attention làm hỏng tất cả.' },
  ],
  formula: 'linear: S ← S + k vᵀ        delta: S ← S + k (v − Sᵀk)ᵀ        gated: S ← α S + k (v − α Sᵀk)ᵀ',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, [1, 1], narrow);
    categoryChart(f, left, 'Sai số đọc lại, trung bình mọi cặp', COUNTS.map(String), CAPACITY.map(c => ({ label: RULE[c.rule][0], values: c.all, color: RULE[c.rule][1], format: v => vn(v, 2) })), [0, 2.2], null, seg(t, 2.5, 8), v => vn(v, 1), 'số cặp đã ghi N');
    categoryChart(f, right, 'Sai số đọc lại, 8 cặp mới nhất', COUNTS.map(String), CAPACITY.map(c => ({ label: RULE[c.rule][0], values: c.recent, color: RULE[c.rule][1], format: v => vn(v, 2) })), [0, 2.2], null, seg(t, 5.5, 9), v => vn(v, 1), 'số cặp đã ghi N');
  },
  metrics: () => [{ label: 'N = 128, linear', value: vn(CAPACITY[0].recent.at(-1)!, 2), tone: 'red' }, { label: 'N = 128, delta (8 mới nhất)', value: vn(CAPACITY[1].recent.at(-1)!, 2), tone: 'cyan' }],
  table: () => ({ headers: ['N', ...CAPACITY.flatMap(c => [`${RULE[c.rule][0]} (mọi cặp)`, '(8 mới nhất)'])], rows: COUNTS.map((n, i) => [n, ...CAPACITY.flatMap(c => [vn(c.all[i], 3), vn(c.recent[i], 3)])]) }),
};

// ---------- Chapter 4: the forget gate ----------
const GATE_LABEL = { 1: 'α = 1 (không quên)', 0.9: 'α = 0,9 cố định', switch: 'α theo dữ liệu (≈0 lúc đổi chủ đề)' } as const;
const chapter4: Scene = {
  title: 'Cổng quên α: quên đúng lúc',
  duration: 11,
  beats: () => [
    { at: 0, text: 'Mô phỏng: 40 bước đầu là chủ đề A (6 cặp key–value lặp lại), 40 bước sau đổi sang chủ đề B. Đường hồng: sai số khi hỏi lại A; đường xanh: khi hỏi B.' },
    { at: 3, text: 'Không quên (α = 1): A vẫn còn sau khi đổi chủ đề, B học dần. Bộ nhớ chứa cả hai, chồng lên nhau trong cùng một ma trận.' },
    { at: 6, text: 'α cố định nhỏ hơn 1 làm hại cả hai: cái gì cũng phai, kể cả B đang học (sai số B dừng ở mức cao).' },
    { at: 8.5, text: 'Trong Gated DeltaNet, α được tính từ chính token đầu vào: gần 1 khi tiếp tục cùng ngữ cảnh, gần 0 khi cần xóa. Chọn các nút để so sánh ba chế độ.' },
  ],
  formula: () => `S ← α_t S + β k (v − α_t Sᵀk)ᵀ,   α_t = exp(−softplus(·)·…) ∈ (0, 1)  ·  chế độ: ${GATE_LABEL[state.gate]}`,
  render(f) {
    const { t } = f; currentT = t;
    const run = topicSwitch(state.gate);
    const cursor = Math.min(run.length - 1, Math.floor(run.length * seg(t, 0.5, 8)));
    lossChart(f, f.safe, [{ values: run.map(r => r.oldErr), color: C.magenta }, { values: run.map(r => r.newErr), color: C.cyan }], cursor, 1.2, null, `Sai số đọc lại chủ đề A (hồng) và B (xanh) · ${GATE_LABEL[state.gate]}`, 'bước');
  },
  metrics: () => { const r = topicSwitch(state.gate).at(-1)!; return [{ label: 'Hỏi lại A cuối cùng', value: vn(r.oldErr, 2), tone: 'magenta' }, { label: 'Hỏi B cuối cùng', value: vn(r.newErr, 2), tone: 'cyan' }]; },
};

// ---------- Chapter 5: why hybrid ----------
const AGE = capacity('delta', 128).errors;
const chapter5: Scene = {
  title: 'Vì sao vẫn giữ vài layer attention',
  duration: 10,
  beats: () => [
    { at: 0, text: 'Cùng thí nghiệm 128 cặp trong trạng thái 32×32: sai số đọc lại theo tuổi của cặp (trái: cũ nhất, phải: mới nhất).' },
    { at: 3, text: `Cặp mới nhất đọc lại tốt (${vn(AGE.at(-1)!, 2)}), cặp ghi từ lâu gần như mất (${vn(AGE[0], 2)}). Một chi tiết chính xác ở rất xa phía trước có thể đã bị nén mất.` },
    { at: 6, text: 'Attention giữ nguyên mọi Key–Value, nên đọc lại chính xác bất kỳ token nào (sai số 0), với giá là KV cache lớn dần.' },
    { at: 8, text: 'Hybrid kết hợp hai loại: 3 layer GDN giữ bản tóm tắt rẻ, cứ 1 layer attention (QSA) để truy xuất chính xác. Qwen3.8-Flash-Next: 36 GDN + 12 QSA.' },
  ],
  formula: 'GDN: bộ nhớ O(1) theo T, đọc lại xấp xỉ     attention: bộ nhớ O(T), đọc lại chính xác',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.3, 1] : [1.7, 1], narrow);
    lossChart(f, left, [{ values: AGE, color: C.cyan }, { values: AGE.map(() => 0.004), color: C.amber }], Math.floor(127 * seg(t, 0.5, 6)), 1.6, null, 'Sai số đọc lại theo thứ tự ghi: GDN (xanh), attention (vàng)', 'cặp thứ (cũ → mới)');
    const avg = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
    barChart(f, right, 'Sai số đọc lại trung bình', ['GDN · 32 cặp cũ nhất', 'GDN · 32 cặp mới nhất', 'attention · mọi cặp'], [avg(AGE.slice(0, 32)), avg(AGE.slice(-32)), 0], { max: 1.6, reveal: seg(t, 3, 8), format: v => vn(v, 2), colors: [C.red, C.cyan, C.amber], accent: C.amber });
  },
  metrics: () => [{ label: 'Cặp mới nhất', value: vn(AGE.at(-1)!, 2), tone: 'cyan' }, { label: '32 cặp cũ nhất', value: vn(AGE.slice(0, 32).reduce((s, v) => s + v, 0) / 32, 2), tone: 'red' }],
};

export const initGatedDeltaNetLab = () => glassLab('gated-delta-net', [chapter1, chapter2, chapter3, chapter4, chapter5], lab => {
  lab.range('alpha', v => vn(v, 2), v => { state.alpha = v; lab.invalidate(); });
  lab.range('beta', v => vn(v, 2), v => { state.beta = v; lab.invalidate(); });
  lab.on('[data-gate]', 'click', el => { const g = el.dataset.gate!; state.gate = g === 'switch' ? 'switch' : (Number(g) as 1 | 0.9); document.querySelectorAll<HTMLElement>('[data-gate]').forEach(b => b.setAttribute('aria-pressed', String(b === el))); lab.replay(); });
});
