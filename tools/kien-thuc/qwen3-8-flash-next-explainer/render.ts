// Qwen3.8-Flash-Next lab: one token through the 48-layer hybrid schedule, where the 125B parameters are and how few a
// token touches, why GDN + QSA keeps long contexts affordable, MoE routing over 512 experts, the 4-branch gated
// residual, and the n-gram table + MTP around the backbone. Architecture numbers come from the published config;
// routing and residual values are seeded illustrations.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, alpha, seg, split, vn, type Rect } from '../../_shared/glass-lab/gfx';
import { barChart, categoryChart } from '../../_shared/glass-lab/views';
import { BRANCHES, CH, CONFIG, coreReads, expertParams, memory, mixerOf, residual, route } from './model';

export const explainerKind = 'educational-simulation';

const state = { token: 'attention' as string, layer: 6, context: 262144 };
let currentT = 0;
const GB = (b: number) => b / 2 ** 30;
const fmtGB = (v: number) => (v < 1 ? `${vn(v * 1024, 0)} MB` : `${vn(v, 1)} GB`);
const CONTEXTS = [8192, 32768, 131072, 262144, 1048576];
const ctx = (n: number) => (n >= 1048576 ? '1M' : `${n / 1024}K`);

function box(f: Frame, x: number, y: number, w: number, h: number, label: string, color: string, on: number, sub = '') {
  const { g } = f;
  g.ctx.save(); g.ctx.fillStyle = alpha(color, 0.08 + 0.3 * on); g.ctx.strokeStyle = alpha(color, 0.35 + 0.65 * on); g.ctx.lineWidth = 1.3;
  if (on > 0.6) { g.ctx.shadowColor = color; g.ctx.shadowBlur = 14; }
  g.ctx.beginPath(); g.ctx.roundRect(x, y, w, h, 6); g.ctx.fill(); g.ctx.stroke(); g.ctx.restore();
  g.text(label, x + w / 2, y + h / 2 - (sub ? 5 : 0), { size: Math.min(12, h * 0.42), color: C.ink, align: 'center', weight: 650, alpha: 0.45 + 0.55 * on });
  if (sub) g.text(sub, x + w / 2, y + h / 2 + 8, { size: 9.5, color: C.muted, align: 'center' });
}

// ---------- Chapter 1: one token through 48 layers ----------
const chapter1: Scene = {
  title: 'Một token đi qua 48 layer hybrid',
  duration: 12,
  idle: true,
  beats: () => [
    { at: 0, text: 'Token được tra embedding 2 560 chiều (từ vocabulary 248 320). Ở layer 2, một bảng n-gram cộng thêm biểu diễn của cụm 2–3 token gần nhất.' },
    { at: 2.5, text: 'Mỗi layer gồm hai sublayer: một token mixer (trao đổi thông tin giữa các vị trí) và một MoE (biến đổi đặc trưng của chính token).' },
    { at: 5, text: 'Lịch token mixer: ba Gated DeltaNet (GDN, bộ nhớ trạng thái) rồi một Qwen Sparse Attention (QSA, truy xuất chính xác), lặp 12 lần: 36 GDN + 12 QSA.' },
    { at: 8, text: 'Giữa các sublayer, residual stream không phải một vector mà bốn nhánh 2 560 chiều (Gated Residual). Cuối cùng đầu ra chiếu thành 248 320 logits; module MTP đề xuất thêm token nháp.' },
  ],
  formula: 'layer ℓ: mixer = QSA nếu ℓ chia hết cho 4, ngược lại GDN;  R ← R + GR_write(mixer(GR_read(R)));  R ← R + GR_write(MoE(GR_read(R)))',
  render(f) {
    const { g, t, time, narrow } = f; currentT = t;
    const area = f.safe;
    g.panel(area.x, area.y, area.w, area.h, 'Lịch 48 layer · token đang đi qua (vàng)', C.violet);
    const cols = narrow ? 6 : 12, rows = 48 / cols, gx = area.x + 20, gy = area.y + 70, cw = (area.w - 40) / cols, ch = (area.h - 120) / rows;
    const pos = t < 5 ? -1 : Math.floor((time * 6) % 48);
    for (let l = 0; l < 48; l++) {
      const c = l % cols, r = Math.floor(l / cols), x = gx + c * cw, y = gy + r * ch, mix = mixerOf(l + 1), on = seg(t, 2.5 + l * 0.05, 3 + l * 0.05);
      const hot = l === pos ? 1 : 0;
      box(f, x + 3, y + 3, cw - 6, ch * 0.5 - 4, `${l + 1} ${mix}`, hot ? C.amber : mix === 'QSA' ? C.magenta : C.cyan, Math.max(on * 0.55, hot));
      box(f, x + 3, y + ch * 0.5 + 1, cw - 6, ch * 0.5 - 6, 'MoE', hot ? C.amber : C.lime, Math.max(on * 0.45, hot));
    }
    g.text('embedding + n-gram (layer 2) → 48 × [mixer → MoE] → logits + MTP', area.x + 20, area.y + 48, { size: 12, color: C.muted });
  },
  metrics: () => [{ label: 'GDN / QSA', value: '36 / 12', tone: 'cyan' }, { label: 'Hidden', value: String(CONFIG.hidden), tone: 'violet' }],
  table: () => ({ headers: ['Thành phần', 'Cấu hình'], rows: [['Layer', '48 (36 GDN + 12 QSA, mỗi layer có MoE)'], ['Hidden size', CONFIG.hidden], ['Vocabulary', '248 320'], ['GDN', '16 head Q/K, 48 head V, 128 chiều, conv 4'], ['QSA', 'indexer 4 head × 128; core 24 head, 2 KV head × 256; ngân sách 2 048, block 4'], ['MoE', '512 expert, chọn 10 + 1 shared, intermediate 640'], ['Gated Residual', '4 nhánh, rank 320'], ['Ngữ cảnh', '262 144 native (tới 1M)']] }),
};

// ---------- Chapter 2: where the parameters are ----------
const ROUTED = CONFIG.layers * CONFIG.moe.experts * expertParams;
const ACTIVE_EXPERT = CONFIG.layers * (CONFIG.moe.topK + CONFIG.moe.shared) * expertParams;
const chapter2: Scene = {
  title: '125 tỷ tham số, mỗi token dùng khoảng 6 tỷ',
  duration: 10,
  beats: () => [
    { at: 0, text: `Một expert là FFN 2 560 → 640 → 2 560 với cổng SiLU: 3 × 2 560 × 640 = ${vn(expertParams / 1e6, 2)} triệu tham số.` },
    { at: 2.5, text: `512 expert × 48 layer = ${vn(ROUTED / 1e9, 1)} tỷ tham số: gần như toàn bộ backbone 125B nằm trong các expert.` },
    { at: 5, text: `Mỗi token chỉ chạy 10 expert được chọn + 1 shared mỗi layer: ${vn(ACTIVE_EXPERT / 1e9, 2)} tỷ; cộng attention, GDN, embedding, router thành khoảng 6B kích hoạt.` },
    { at: 7.5, text: 'Ngoài backbone còn bảng n-gram 51B (mỗi token chỉ đọc vài hàng) và MTP 4B. Tổng trọng số phải lưu ~180B (≈360 GB ở bf16): “6B active” là phép tính, không phải bộ nhớ.' },
  ],
  formula: () => `expert = 3·d·d_ff = 3·2560·640 = ${vn(expertParams / 1e6, 2)}M     routed = 48·512·expert = ${vn(ROUTED / 1e9, 1)}B     active expert = 48·11·expert = ${vn(ACTIVE_EXPERT / 1e9, 2)}B`,
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, [1, 1], narrow);
    barChart(f, left, 'Tham số phải lưu (tỷ)', ['expert (routed)', 'phần còn lại backbone', 'bảng n-gram', 'MTP'], [ROUTED / 1e9, 125 - ROUTED / 1e9, 51, 4], { reveal: seg(t, 2.5, 8), format: v => vn(v, 1), colors: [C.lime, C.violet, C.amber, C.cyan], accent: C.violet });
    barChart(f, right, 'Tham số dùng cho mỗi token (tỷ)', ['expert chọn + shared', 'khác (≈)', 'tổng kích hoạt'], [ACTIVE_EXPERT / 1e9, 6 - ACTIVE_EXPERT / 1e9, 6], { max: 8, reveal: seg(t, 5, 8), format: v => vn(v, 1), colors: [C.lime, C.violet, C.white], accent: C.lime });
  },
  metrics: () => [{ label: 'Lưu', value: '≈180B', tone: 'violet' }, { label: 'Kích hoạt', value: '≈6B / token', tone: 'lime' }],
};

// ---------- Chapter 3: long context ----------
const chapter3: Scene = {
  title: 'Ngữ cảnh dài: GDN giữ bộ nhớ cố định, QSA đọc có ngân sách',
  duration: 11,
  beats: () => {
    const m = memory(state.context);
    return [
      { at: 0, text: `Bộ nhớ mỗi chuỗi ở ${ctx(state.context)} token (bf16): nếu cả 48 layer là attention cần ${fmtGB(GB(m.allAttention))}; hybrid chỉ giữ KV cache ở 12 layer QSA cộng trạng thái cố định của 36 GDN: ${fmtGB(GB(m.hybrid))}.` },
      { at: 4, text: `Phép tính cũng có ngân sách: mỗi query của QSA chỉ đọc ${coreReads(state.context).toLocaleString('vi-VN')} token (2 048 + phần đuôi block) thay vì ${state.context.toLocaleString('vi-VN')}.` },
      { at: 7.5, text: 'GDN tóm tắt rẻ nhưng xấp xỉ; QSA truy xuất chính xác nhưng tốn KV cache. Tỷ lệ 3 : 1 cân bằng hai thứ. Kéo Độ dài ngữ cảnh để xem.' },
    ];
  },
  formula: () => `KV/token/layer QSA = 2·2·256·2 B = 2 KB   ·   trạng thái GDN = 48·128·128·2 B = 1,5 MB/layer   ·   QSA đọc ${coreReads(state.context)} token/query`,
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, [1.3, 1], narrow);
    categoryChart(f, left, 'Bộ nhớ mỗi chuỗi (thang log₁₀ GB)', CONTEXTS.map(ctx), [
      { label: '48 attention', values: CONTEXTS.map(n => Math.log10(GB(memory(n).allAttention))), color: C.red, format: v => fmtGB(10 ** v) },
      { label: 'hybrid (Qwen3.8)', values: CONTEXTS.map(n => Math.log10(GB(memory(n).hybrid))), color: C.lime, format: v => fmtGB(10 ** v) },
    ], [-1, 2.2], CONTEXTS.indexOf(state.context), seg(t, 0.5, 4), v => fmtGB(10 ** v), 'ngữ cảnh');
    barChart(f, right, `Token đọc mỗi query ở ${ctx(state.context)} (log₁₀)`, ['attention dense', 'QSA'], [Math.log10(state.context), Math.log10(coreReads(state.context))], { max: 6.2, reveal: seg(t, 4, 7), format: v => Math.round(10 ** v).toLocaleString('vi-VN'), colors: [C.red, C.magenta], accent: C.magenta });
  },
  metrics: () => { const m = memory(state.context); return [{ label: '48 attention', value: fmtGB(GB(m.allAttention)), tone: 'red' }, { label: 'Hybrid', value: fmtGB(GB(m.hybrid)), tone: 'lime' }]; },
};

// ---------- Chapter 4: MoE routing ----------
const chapter4: Scene = {
  title: 'MoE: router chọn 10 trong 512 expert',
  duration: 11,
  beats: () => {
    const r = route(state.token, state.layer);
    return [
      { at: 0, text: `Token “${state.token}” ở layer ${state.layer}: router (một lớp tuyến tính) chấm điểm cả 512 expert.` },
      { at: 3, text: `Giữ 10 điểm cao nhất (sáng), chuẩn hóa softmax thành trọng số: lớn nhất ${vn(r[0].weight, 2)}. 502 expert còn lại không chạy cho token này.` },
      { at: 6, text: 'Shared expert (khung riêng) luôn chạy cho mọi token, không nằm trong 10 expert được chọn. Đầu ra là tổng có trọng số của 10 expert cộng shared.' },
      { at: 8.5, text: 'Mỗi layer có router riêng: đổi token hoặc layer, tập expert đổi theo. Routing ở đây là minh họa có seed cố định; xem bài Mixture of Experts để thấy router huấn luyện thật.' },
    ];
  },
  formula: 'MoE(x) = Σ_{e ∈ Top10(r(x))} p_e(x) · Expert_e(x) + Shared(x),   Expert(x) = W_down[SiLU(W_gate x) ⊙ W_up x]',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const r = route(state.token, state.layer), chosen = new Map(r.map(e => [e.expert, e.weight]));
    const [left, right] = split(f.safe, narrow ? [1.2, 1] : [1.6, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, `512 expert · token “${state.token}”, layer ${state.layer}`, C.lime);
    const cols = 32, rows = 16, cell = Math.min((left.w - 30) / cols, (left.h - 110) / rows), ox = left.x + (left.w - cols * cell) / 2, oy = left.y + 44;
    const s1 = seg(t, 0, 3), s2 = seg(t, 3, 5);
    for (let e = 0; e < 512; e++) {
      const x = ox + (e % cols) * cell, y = oy + Math.floor(e / cols) * cell, w = chosen.get(e);
      g.ctx.fillStyle = w !== undefined && s2 > 0 ? alpha(C.lime, 0.4 + 0.6 * s2) : alpha(C.white, 0.06 + 0.1 * s1 * ((e * 37) % 11) / 11);
      g.ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
    }
    const sy = oy + rows * cell + 14;
    box(f, left.x + left.w / 2 - 70, sy, 140, 34, 'Shared expert', C.amber, seg(t, 6, 7.5), 'luôn chạy');
    barChart(f, right, 'Trọng số 10 expert được chọn', r.map(e => `#${e.expert}`), r.map(e => e.weight), { reveal: seg(t, 3, 6), format: v => vn(v, 2), color: C.lime, accent: C.lime });
  },
  metrics: () => [{ label: 'Expert chạy', value: '10 + 1 / 512', tone: 'lime' }, { label: 'Trọng số lớn nhất', value: vn(route(state.token, state.layer)[0].weight, 2), tone: 'amber' }],
};

// ---------- Chapter 5: gated residual ----------
function vecRow(f: Frame, x: number, y: number, w: number, values: number[], color: string, label: string, max = 2) {
  const { g } = f, cw = w / values.length;
  g.text(label, x - 8, y + 10, { size: 11, color: C.muted, align: 'right' });
  values.forEach((v, i) => {
    const h = Math.min(1, Math.abs(v) / max) * 20;
    g.ctx.fillStyle = alpha(v >= 0 ? color : C.red, 0.85); g.ctx.fillRect(x + i * cw + 2, y + 10 - (v >= 0 ? h : 0), cw - 4, Math.max(1, h));
  });
}
const WRITE = [1.4, 0.6, 1, 0.3];
const chapter5: Scene = {
  title: 'Gated Residual: bốn nhánh, đọc theo kênh, ghi theo nhánh',
  duration: 12,
  beats: () => [
    { at: 0, text: 'Transformer thường có một residual stream. Qwen3.8 giữ bốn nhánh R₁…R₄, mỗi nhánh 2 560 chiều (ở đây 8 kênh để dễ nhìn, giá trị minh họa).' },
    { at: 3, text: 'Đọc: chuẩn hóa từng nhánh, nhân với cổng sigmoid riêng cho từng cặp nhánh × kênh (do một bottleneck rank 320 dự đoán), lấy trung bình thành một vector x cho sublayer.' },
    { at: 6.5, text: 'Sublayer (GDN, QSA hoặc MoE) tính y = F(x) như bình thường, chỉ trên một vector 2 560 chiều.' },
    { at: 9, text: `Ghi: cùng y được cộng vào mỗi nhánh với một hệ số vô hướng sᵢ ∈ (0, 2): ở đây s = ${WRITE.map(v => vn(v, 1)).join(', ')}. Mỗi nhánh tích lũy theo nhịp riêng.` },
  ],
  formula: 'x = ¼ Σᵢ Gᵢ ⊙ RMSNorm(Rᵢ),  Gᵢ ∈ (0,1)^2560     y = F(x)     Rᵢ ← Rᵢ + sᵢ·y,  sᵢ = 2σ(·) ∈ (0, 2)',
  render(f) {
    const { g, t } = f; currentT = t;
    const res = residual([1, 1, 1, 1], WRITE), area = f.safe;
    g.panel(area.x, area.y, area.w, area.h, `Gated Residual với ${CH} kênh · minh họa`, C.violet);
    const x0 = area.x + 110, w = Math.min(320, area.w * 0.3), rowH = Math.min(46, (area.h - 80) / 9);
    const colors = [C.cyan, C.magenta, C.amber, C.lime];
    BRANCHES.forEach((b, i) => vecRow(f, x0, area.y + 50 + i * rowH, w, b, colors[i], `R${i + 1}`));
    const s1 = seg(t, 3, 5.5), s2 = seg(t, 6.5, 8), s3 = seg(t, 9, 11);
    if (s1 > 0) {
      res.gates.forEach((gate, i) => vecRow(f, x0 + w + 90, area.y + 50 + i * rowH, w * 0.7, gate, colors[i], `G${i + 1}`, 1));
      g.arrow(x0 + w + 10, area.y + 50 + 2 * rowH, x0 + w + 70, area.y + 50 + 2 * rowH, C.muted, { width: 1.2, alpha: s1 });
      vecRow(f, x0, area.y + 60 + 4.4 * rowH, w, res.x, C.white, 'x');
    }
    if (s2 > 0) vecRow(f, x0, area.y + 60 + 5.4 * rowH, w, res.y, C.violet, 'y = F(x)', 1);
    if (s3 > 0) res.updated.forEach((b, i) => vecRow(f, x0 + w + 90, area.y + 60 + (4.4 + i) * rowH, w * 0.7, b, colors[i], `R${i + 1}′ (s=${vn(WRITE[i], 1)})`));
  },
  metrics: () => [{ label: 'Nhánh', value: '4 × 2 560', tone: 'violet' }, { label: 'Rank cổng', value: String(CONFIG.grRank), tone: 'cyan' }],
};

// ---------- Chapter 6: n-gram table and MTP ----------
const chapter6: Scene = {
  title: 'Quanh backbone: bảng n-gram và Multi-Token Prediction',
  duration: 11,
  beats: () => [
    { at: 0, text: 'Bảng n-gram (51B tham số): tại vị trí t, cụm (x_{t−1}, x_t) và (x_{t−2}, x_{t−1}, x_t) được băm thành địa chỉ, đọc vài hàng vector và cộng vào residual ở layer 2.' },
    { at: 3.5, text: 'Địa chỉ chỉ phụ thuộc token ID, nên biết trước khi tính: bảng có thể nằm ở bộ nhớ host, được prefetch trong lúc layer 1 đang chạy. 51B tham số mà không chiếm VRAM.' },
    { at: 7, text: 'MTP (1 layer, ~4B): đề xuất vài token nháp; backbone kiểm tra cả chuỗi nháp trong một lượt, chấp nhận phần đúng (speculative decoding). Nháp càng hay được chấp nhận, decode càng nhanh.' },
  ],
  formula: 'u_t = Lookup₂(hash(x_{t−1}, x_t)) ‖ Lookup₃(hash(x_{t−2}, x_{t−1}, x_t))     tốc độ MTP ≈ (1 + số token nháp được chấp nhận) / (chi phí một lượt kiểm tra)',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const area = f.safe;
    g.panel(area.x, area.y, area.w, area.h, 'Luồng dữ liệu quanh backbone', C.amber);
    const toks = ['tensor', '.', 'shape'], bx = area.x + 30, by = area.y + 60, tw = narrow ? 70 : 90;
    toks.forEach((tk, i) => box(f, bx + i * (tw + 10), by, tw, 32, tk, C.cyan, 1, `x_{t−${2 - i}}`.replace('x_{t−0}', 'x_t')));
    const s1 = seg(t, 0.5, 2.5);
    box(f, bx, by + 60, tw * 2 + 10, 30, 'hash bigram', C.amber, s1);
    box(f, bx, by + 100, tw * 3 + 20, 30, 'hash trigram', C.amber, s1);
    const tx = bx + tw * 3 + 70;
    box(f, tx, by + 60, narrow ? 120 : 170, 70, 'Bảng n-gram 51B', C.amber, seg(t, 1.5, 3), 'bộ nhớ host, prefetch');
    g.arrow(bx + tw * 3 + 25, by + 95, tx - 6, by + 95, C.muted, { width: 1.3, alpha: s1 });
    const lx = tx + (narrow ? 140 : 200);
    box(f, lx, by + 60, narrow ? 90 : 130, 70, 'Layer 2', C.violet, seg(t, 3.5, 5), '+ u_t vào residual');
    g.arrow(tx + (narrow ? 120 : 170) + 4, by + 95, lx - 6, by + 95, C.muted, { width: 1.3, alpha: seg(t, 3.5, 5) });
    const my = by + 180, s3 = seg(t, 7, 9);
    box(f, bx, my, 140, 40, 'Backbone', C.violet, s3, 'token “shape”');
    ['(', 'batch', ',', 'seq'].forEach((d, i) => box(f, bx + 170 + i * 74, my, 66, 40, d, i < 3 ? C.lime : C.red, s3 * seg(t, 7.5 + i * 0.3, 8 + i * 0.3), i < 3 ? 'chấp nhận' : 'bị từ chối'));
    g.text('MTP đề xuất 4 token nháp, backbone kiểm tra một lượt: 3 được chấp nhận', bx, my + 62, { size: 11.5, color: C.muted, alpha: s3 });
  },
  metrics: () => [{ label: 'N-gram', value: '51B (host)', tone: 'amber' }, { label: 'MTP', value: '1 layer, ~4B', tone: 'cyan' }],
};

export const initQwenLab = () => glassLab('qwen3-8-flash-next', [chapter1, chapter2, chapter3, chapter4, chapter5, chapter6], lab => {
  lab.range('context', v => ctx(CONTEXTS[v]), v => { state.context = CONTEXTS[v]; lab.invalidate(); });
  lab.range('layer', v => String(v), v => { state.layer = v; lab.invalidate(); });
  lab.on('[data-token]', 'click', el => { state.token = el.dataset.token!; document.querySelectorAll<HTMLElement>('[data-token]').forEach(b => b.setAttribute('aria-pressed', String(b === el))); lab.replay(); });
});
