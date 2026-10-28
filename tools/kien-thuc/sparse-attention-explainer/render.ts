// Sparse attention lab: real attention maps and rows of Qwen2.5-0.5B (where the mass actually goes: attention sink,
// recent tokens, a few far spikes), fixed patterns (local window, sink + local), QSA's block selection step by step on
// a 32-token code example, and the measured coverage of each pattern by budget and by layer.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, Plot, alpha, seg, split, vn, type Rect } from '../../_shared/glass-lab/gfx';
import { barChart, categoryChart } from '../../_shared/glass-lab/views';
import { BLOCK, BLOCKS, REAL, TOKENS, allowed, completeBlocks, coverage, indexerScores, tokenText, topBlocks, weights } from './model';

export const explainerKind = 'educational-simulation';

const state = { pic: 2, window: 64, budget: 64, toyBudget: 2 };
let currentT = 0;
const pct = (v: number, d = 0) => `${vn(v * 100, d)}%`;
const pic = () => REAL.pictures[state.pic];
const O = REAL.overall;

// ---------- Chapter 1: a real attention map ----------
function heat(f: Frame, rect: Rect, map: number[][], reveal: number) {
  const { g } = f, n = map.length, s = Math.min(rect.w - 60, rect.h - 70), cell = s / n, ox = rect.x + (rect.w - s) / 2, oy = rect.y + 44;
  const peak = Math.max(...map.flat().filter((_, i) => i % (n + 1) !== 0)) || 1;
  const rows = Math.ceil(n * reveal);
  for (let i = 0; i < rows; i++) for (let j = 0; j <= i; j++) {
    const v = Math.min(1, Math.sqrt(map[i][j] / peak));
    g.ctx.fillStyle = `rgba(${Math.round(80 + 175 * v)},${Math.round(60 + 150 * v)},${Math.round(255 - 120 * v)},${0.08 + 0.92 * v})`;
    g.ctx.fillRect(ox + j * cell, oy + i * cell, cell + 0.4, cell + 0.4);
  }
  g.ctx.strokeStyle = 'rgba(255,255,255,0.18)'; g.ctx.strokeRect(ox, oy, s, s);
  g.text('key (token được đọc) →', ox + s, oy + s + 16, { size: 11, color: C.muted, align: 'right' });
  g.text('query ↓', ox - 6, oy + 8, { size: 11, color: C.muted, align: 'right' });
  return { ox, oy, s };
}
const chapter1: Scene = {
  title: 'Attention thật đọc những đâu?',
  duration: 10,
  beats: () => [
    { at: 0, text: `Bản đồ attention thật của ${REAL.model} trên ${REAL.seq} token văn bản Wikipedia (layer ${pic().layer + 1}, head ${pic().head + 1}), gộp 16×16 token mỗi ô. Hàng là token đang hỏi, cột là token được đọc.` },
    { at: 3, text: 'Dense attention tính đủ mọi ô dưới đường chéo: 1 024 token là ~525 nghìn cặp mỗi head, mỗi layer; 128K token là 8,6 tỷ cặp.' },
    { at: 6, text: 'Nhưng khối lượng không trải đều: một cột sáng ở token đầu tiên (attention sink), một dải sáng sát đường chéo (token gần), vài điểm rải rác ở xa. Phần lớn các ô gần như bằng 0.' },
    { at: 8.5, text: 'Sparse attention dựa vào điều đó: chỉ tính những ô có khả năng lớn. Câu hỏi là chọn ô nào mà không phải tính hết trước.' },
  ],
  formula: 'dense: T·(T+1)/2 tích vô hướng mỗi head     sparse: T·B (B = ngân sách token mỗi query)',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.2, 1] : [1.3, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, `Layer ${pic().layer + 1}, head ${pic().head + 1} · đo thật`, C.violet);
    heat(f, left, pic().map, seg(t, 0, 3));
    const lens = [1024, 8192, 32768, 131072];
    barChart(f, right, 'Số cặp query–key mỗi head (log₁₀)', lens.map(n => (n >= 1024 ? `${n / 1024}K` : String(n))), lens.map(n => Math.log10(n * (n + 1) / 2)), { max: 11, reveal: seg(t, 3, 6), format: v => (10 ** v >= 1e9 ? `${vn(10 ** v / 1e9, 1)} tỷ` : `${vn(10 ** v / 1e6, 1)} tr`), color: C.red, accent: C.red });
  },
  metrics: () => [{ label: 'Token', value: String(REAL.seq), tone: 'violet' }, { label: 'Layer × head', value: `${REAL.layers} × ${REAL.heads}`, tone: 'cyan' }],
};

// ---------- Chapter 2: one real query row ----------
function rowView(f: Frame, rect: Rect, row: number[], keep: ((j: number) => boolean) | null, title: string, reveal = 1) {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, title, C.cyan);
  const plot = new Plot(g, rect, [0, row.length], [0, 1], { l: 16, r: 14, t: 40, b: 30 });
  plot.axes({ xTicks: [0, 256, 512, 768, 1023], yTicks: [], xLabel: 'vị trí token được đọc', fx: v => String(Math.round(v)) });
  const peak = Math.max(...row), n = Math.floor(row.length * reveal), w = plot.box.w / row.length;
  if (keep) for (let j = 0; j < row.length; j += REAL.block) if (keep(j)) { g.ctx.fillStyle = alpha(C.lime, 0.16); g.ctx.fillRect(plot.px(j), plot.box.y, w * REAL.block, plot.box.h); }
  for (let j = 0; j < n; j++) {
    const h = Math.sqrt(row[j] / peak) * plot.box.h;
    g.ctx.fillStyle = keep && !keep(j) ? alpha(C.red, 0.85) : C.cyan;
    g.ctx.fillRect(plot.px(j), plot.box.y + plot.box.h - h, Math.max(1, w), h);
  }
  return plot;
}
const chapter2: Scene = {
  title: 'Một hàng attention: sink, gần, và vài điểm xa',
  duration: 11,
  beats: () => {
    const row = pic().row, top = [...row.keys()].sort((a, b) => row[b] - row[a]).slice(0, 4);
    return [
      { at: 0, text: `Hàng attention thật của token cuối (vị trí 1023), layer ${pic().layer + 1}, head ${pic().head + 1}. Chiều cao là căn bậc hai của trọng số để thấy cả các giá trị nhỏ.` },
      { at: 3, text: `Bốn token nhận nhiều nhất: ${top.map(j => `“${tokenText(j)}” (${j}, ${pct(row[j], 1)})`).join(', ')}.` },
      { at: 6, text: `Token đầu tiên nhận ${pct(row[0], 1)} dù không liên quan nội dung: hiện tượng attention sink, softmax cần một chỗ “đổ” khối lượng thừa. Nhiều head ở layer giữa như vậy; layer đầu thì nhìn token gần, layer cuối trải rộng hơn.` },
      { at: 9, text: 'Đổi head bằng các nút để so sánh. Không có một mẫu cố định nào đúng cho mọi head.' },
    ];
  },
  formula: () => { const row = pic().row; return `token đầu (sink): ${pct(row[0], 1)}   ·   64 token gần nhất: ${pct(row.slice(-64).reduce((s, v) => s + v, 0), 1)}   ·   phần còn lại: ${pct(1 - row[0] - row.slice(-64).reduce((s, v) => s + v, 0), 1)}`; },
  render(f) {
    const { t } = f; currentT = t;
    rowView(f, f.safe, pic().row, null, `Trọng số attention của token cuối · layer ${pic().layer + 1}, head ${pic().head + 1}`, seg(t, 0, 3));
  },
  metrics: () => { const row = pic().row; return [{ label: 'Sink (token 0)', value: pct(row[0], 1), tone: 'violet' }, { label: '64 token gần', value: pct(row.slice(-64).reduce((s, v) => s + v, 0), 1), tone: 'cyan' }]; },
};

// ---------- Chapter 3: fixed patterns ----------
const chapter3: Scene = {
  title: 'Mẫu cố định: cửa sổ gần và attention sink',
  duration: 10,
  beats: () => [
    { at: 0, text: `Đo trên mọi layer, mọi head, 512 query ở nửa sau chuỗi: chỉ đọc ${state.window} token gần nhất (sliding window) giữ được ${pct(O[`local${state.window}`])} khối lượng attention.` },
    { at: 3.5, text: `Giữ thêm 4 token đầu (attention sink, StreamingLLM) cùng ngân sách: ${pct(O[`sink${state.window}`])}. Bốn token rẻ nhất mang phần lớn khối lượng bị thiếu.` },
    { at: 6.5, text: 'Phần còn thiếu là các token ở xa nhưng liên quan nội dung: một tên riêng nhắc lại, một định nghĩa phía trước. Mẫu cố định không biết chúng ở đâu; cần chọn theo nội dung.' },
  ],
  formula: () => `độ phủ = Σ_{j ∈ tập đọc} A[i, j]   (A: softmax dense)     cửa sổ ${state.window}: ${pct(O[`local${state.window}`], 1)}  ·  sink + cửa sổ: ${pct(O[`sink${state.window}`], 1)}`,
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.5, 1], narrow);
    const W = state.window, q = REAL.seq - 1;
    const keep = t < 3.5 ? (j: number) => j > q - W : (j: number) => j > q - (W - REAL.sink) || j < REAL.sink;
    rowView(f, left, pic().row, keep, `Cùng hàng attention, vùng xanh là token được đọc (${t < 3.5 ? `cửa sổ ${W}` : `4 sink + cửa sổ ${W - 4}`})`);
    categoryChart(f, right, 'Độ phủ trung bình (mọi layer, head)', REAL.windows.map(String), [
      { label: 'cửa sổ', values: REAL.windows.map(w => O[`local${w}`]), color: C.red, format: v => pct(v) },
      { label: 'sink + cửa sổ', values: REAL.windows.map(w => O[`sink${w}`]), color: C.amber, format: v => pct(v) },
    ], [0, 1], REAL.windows.indexOf(W), seg(t, 1, 6), v => pct(v), 'ngân sách token');
  },
  metrics: () => [{ label: `Cửa sổ ${state.window}`, value: pct(O[`local${state.window}`], 1), tone: 'red' }, { label: 'Sink + cửa sổ', value: pct(O[`sink${state.window}`], 1), tone: 'amber' }],
};

// ---------- Chapter 4: QSA step by step on 32 code tokens ----------
const Q = TOKENS.length - 1;
function toyStrip(f: Frame, rect: Rect, t: number) {
  const { g } = f, scores = indexerScores(Q), blocks = topBlocks(Q, state.toyBudget), keep = allowed('qsa', Q, state.toyBudget), w = weights(Q, keep);
  const cw = (rect.w - 30) / TOKENS.length, y0 = rect.y + 50, ox = rect.x + 15;
  const s1 = seg(t, 2, 4), s2 = seg(t, 4.5, 6.5), s3 = seg(t, 7, 9);
  const maxScore = Math.max(...scores.map(s => s ?? 0)) || 1;
  for (let b = 0; b < BLOCKS; b++) {
    const x = ox + b * BLOCK * cw, picked = blocks.includes(b), complete = b < completeBlocks(Q);
    g.ctx.strokeStyle = alpha(picked && s2 > 0 ? C.lime : C.white, 0.25 + 0.5 * (picked ? s2 : 0)); g.ctx.lineWidth = picked ? 2 : 1;
    g.ctx.strokeRect(x + 1, y0 - 4, BLOCK * cw - 2, 40);
    if (complete && scores[b] !== null) {
      const h = 50 * (scores[b]! / maxScore) * s1;
      g.ctx.fillStyle = alpha(picked ? C.lime : C.violet, 0.8); g.ctx.fillRect(x + BLOCK * cw * 0.25, y0 + 110 - h, BLOCK * cw * 0.5, h);
      g.text(vn(scores[b]!, 1), x + BLOCK * cw / 2, y0 + 122, { size: 10, color: C.muted, align: 'center', alpha: s1 });
    }
  }
  TOKENS.forEach((tok, j) => {
    const x = ox + j * cw + cw / 2, on = keep.includes(j);
    g.text(tok.text, x, y0 + 16, { size: Math.min(12, cw * 0.8), color: on || s2 === 0 ? C.ink : C.faint, align: 'center', mono: true });
    const h = 60 * w[j] * s3 * 2;
    g.ctx.fillStyle = C.amber; g.ctx.fillRect(x - cw * 0.3, y0 + 210 - h, cw * 0.6, h);
  });
  g.text('điểm indexer từng block (q · trung bình key)', ox, y0 + 58, { size: 11, color: C.violet, alpha: s1 });
  g.text('attention sau khi mở block đã chọn', ox, y0 + 150, { size: 11, color: C.amber, alpha: s3 });
  g.text(`query: “${TOKENS[Q].text}” (token cuối)`, ox + rect.w - 30, y0 - 18, { size: 11.5, color: C.ink, align: 'right' });
}
const chapter4: Scene = {
  title: 'QSA từng bước: chấm điểm block, chọn top-k, mở ra token',
  duration: 11,
  beats: () => [
    { at: 0, text: 'Ví dụ nhỏ 32 token code (def area(r): return PI*r*r …). Qwen Sparse Attention không chấm từng token mà gộp key thành micro-block 4 token.' },
    { at: 2, text: 'Bước 1: indexer (vài head nhỏ, ReLU) chấm điểm mỗi block hoàn chỉnh bằng tích vô hướng với key trung bình của block. Rẻ: số block ít hơn số token 4 lần, head nhỏ hơn.' },
    { at: 4.5, text: `Bước 2: giữ ${state.toyBudget} block điểm cao nhất (khung xanh) cùng block đang dở của chính query. Token ngoài block bị bỏ qua hoàn toàn.` },
    { at: 7, text: `Bước 3: attention đầy đủ chỉ trên các token đã chọn (cột vàng). Độ phủ so với dense: ${pct(coverage('qsa', Q, state.toyBudget))}. Qwen3.8 dùng ngân sách 2 048 token mỗi query.` },
  ],
  formula: 'điểm(b) = Σ_h ReLU(q_hᴵ · k̄_b),   k̄_b = trung bình 4 key của block b,   tập đọc = top-k block ∪ block hiện tại',
  render(f) {
    const { t } = f; currentT = t;
    f.g.panel(f.safe.x, f.safe.y, f.safe.w, f.safe.h, 'Ví dụ 32 token · mô phỏng', C.lime);
    toyStrip(f, f.safe, t);
  },
  metrics: () => [{ label: 'Block chọn', value: `${state.toyBudget} / ${completeBlocks(Q)}`, tone: 'lime' }, { label: 'Độ phủ', value: pct(coverage('qsa', Q, state.toyBudget)), tone: 'amber' }],
};

// ---------- Chapter 5: measured coverage by budget and layer ----------
const chapter5: Scene = {
  title: 'Đo thật: chọn block theo nội dung giữ được bao nhiêu?',
  duration: 12,
  beats: () => {
    const B = state.budget;
    return [
      { at: 0, text: `Cùng phép đo trên Qwen2.5-0.5B, chọn block 4 token theo điểm q · key trung bình như QSA (chưa huấn luyện indexer riêng), ngân sách ${B} token mỗi query.` },
      { at: 3, text: `Chọn theo nội dung giữ ${pct(O[`index${B}`])} khối lượng; chọn hoàn hảo theo khối lượng thật (giới hạn trên) ${pct(O[`oracle${B}`])}; sink + cửa sổ cùng ngân sách ${pct(O[`sink${B}`])}.` },
      { at: 6.5, text: `Ngân sách 256 token (25% chuỗi 1 024): chọn theo nội dung đạt ${pct(O.index256)}. Ở chuỗi dài, cùng ngân sách là một phần rất nhỏ, nên chất lượng indexer quyết định tất cả: QSA huấn luyện riêng indexer để bắt chước attention dense.` },
      { at: 9.5, text: 'Theo layer: hai layer cuối khó làm thưa nhất (attention trải rộng), các layer giữa phần lớn là sink + vài block. Mô hình hybrid chỉ dùng attention ở 1/4 số layer.' },
    ];
  },
  formula: () => `ngân sách ${state.budget}: theo nội dung ${pct(O[`index${state.budget}`], 1)}  ·  giới hạn trên ${pct(O[`oracle${state.budget}`], 1)}  ·  sink + cửa sổ ${pct(O[`sink${state.budget}`], 1)}`,
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, [1, 1.3], narrow);
    categoryChart(f, left, 'Độ phủ theo ngân sách token', REAL.budgets.map(String), [
      { label: 'sink + cửa sổ', values: REAL.budgets.map(b => O[`sink${b}`]), color: C.amber, format: v => pct(v) },
      { label: 'top-k block (q·k̄)', values: REAL.budgets.map(b => O[`index${b}`]), color: C.lime, format: v => pct(v) },
      { label: 'giới hạn trên', values: REAL.budgets.map(b => O[`oracle${b}`]), color: C.white, format: v => pct(v) },
    ], [0.4, 1], REAL.budgets.indexOf(state.budget), seg(t, 1, 6), v => pct(v), 'ngân sách token');
    const layers = REAL.perLayer.map((_, i) => String(i + 1));
    categoryChart(f, right, 'Theo layer, ngân sách 64', layers.map((l, i) => (i % 4 === 0 || i === layers.length - 1 ? l : '')), [
      { label: 'cửa sổ 64', values: REAL.perLayer.map(l => l.local64), color: C.red, format: v => pct(v) },
      { label: 'sink + cửa sổ', values: REAL.perLayer.map(l => l.sink64), color: C.amber, format: v => pct(v) },
      { label: 'top-k block', values: REAL.perLayer.map(l => l.index64), color: C.lime, format: v => pct(v) },
    ], [0, 1], null, seg(t, 9.5, 11.5), v => pct(v), 'layer');
  },
  metrics: () => [{ label: `Top-k block, ${state.budget}`, value: pct(O[`index${state.budget}`], 1), tone: 'lime' }, { label: 'Giới hạn trên', value: pct(O[`oracle${state.budget}`], 1), tone: 'violet' }],
  table: () => ({ headers: ['Layer', 'Cửa sổ 64', 'Sink + cửa sổ 64', 'Top-k block 64', 'Giới hạn trên 64', 'Top-k block 256'], rows: REAL.perLayer.map((l, i) => [i + 1, pct(l.local64, 1), pct(l.sink64, 1), pct(l.index64, 1), pct(l.oracle64, 1), pct(l.index256, 1)]), caption: 'Trung bình 14 head × 512 query (vị trí 512–1023).' }),
};

export const initSparseAttentionLab = () => glassLab('sparse-attention', [chapter1, chapter2, chapter3, chapter4, chapter5], lab => {
  lab.on('[data-pic]', 'click', el => { state.pic = Number(el.dataset.pic); document.querySelectorAll<HTMLElement>('[data-pic]').forEach(b => b.setAttribute('aria-pressed', String(b === el))); lab.invalidate(); });
  lab.range('window', v => `${REAL.windows[v]} token`, v => { state.window = REAL.windows[v]; lab.invalidate(); });
  lab.range('budget', v => `${REAL.budgets[v]} token`, v => { state.budget = REAL.budgets[v]; lab.invalidate(); });
  lab.range('toy', v => `${v} block`, v => { state.toyBudget = v; lab.invalidate(); });
});
