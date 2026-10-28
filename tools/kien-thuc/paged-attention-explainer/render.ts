// PagedAttention lab: how big the KV cache is, what contiguous reservation does to GPU memory, how fixed-size pages
// with a block table fix it, the throughput that follows, and copy-on-write prefix sharing for parallel sampling.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, alpha, seg, split, vn, type Rect } from '../../_shared/glass-lab/gfx';
import { barChart, lossChart } from '../../_shared/glass-lab/views';
import { BLOCK, BLOCKS, COLS, INFER, MAX_LEN, ROWS, STEPS, STRATEGIES, sharing, simulate, type Run, type Strategy } from './model';

export const explainerKind = 'educational-simulation';

const PALETTE = ['#4fd6ff', '#7af0a0', '#ffb648', '#a98bff', '#ff5fa8', '#5b8cff', '#ff9e6b', '#c8f06b'];
const runs: Partial<Record<Strategy, Run>> = {};
const run = (s: Strategy) => (runs[s] ??= simulate(s));
const state = { step: null as number | null, samples: 4 };
let currentT = 0;
const stepNow = (from = 1, to = 11) => state.step ?? Math.min(STEPS - 1, Math.floor(seg(currentT, from, to) * (STEPS - 1)));
const utilization = (r: Run, i: number) => { const s = r.steps[i]; return s.allocated ? s.stored / (s.allocated * BLOCK) : 0; };
const mb = (bytes: number) => (bytes >= 1e6 ? `${vn(bytes / 1e6, 2)} MB` : `${vn(bytes / 1e3, 1)} KB`);

/** Draw GPU memory as a grid of blocks: colour = owning request, brightness = how full the block is. */
function memory(f: Frame, rect: Rect, r: Run, step: number, title: string, accent: string = C.violet, highlight = -1) {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, title, accent);
  const cell = Math.min((rect.w - 28) / COLS, (rect.h - 70) / ROWS), ox = rect.x + (rect.w - cell * COLS) / 2, oy = rect.y + 44;
  const own = r.owner[step], fl = r.fill[step];
  for (let i = 0; i < BLOCKS; i++) {
    const x = ox + (i % COLS) * cell, y = oy + Math.floor(i / COLS) * cell, o = own[i];
    if (o < 0) { g.ctx.fillStyle = 'rgba(255,255,255,0.05)'; g.ctx.fillRect(x + 0.6, y + 0.6, cell - 1.2, cell - 1.2); continue; }
    const color = PALETTE[o % PALETTE.length], k = fl[i] / BLOCK;
    if (k > 0) { g.ctx.fillStyle = alpha(color, 0.35 + 0.6 * k); g.ctx.fillRect(x + 0.6, y + 0.6, cell - 1.2, cell - 1.2); }
    else { g.ctx.fillStyle = alpha(color, 0.1); g.ctx.fillRect(x + 0.6, y + 0.6, cell - 1.2, cell - 1.2); g.ctx.strokeStyle = alpha(color, 0.55); g.ctx.lineWidth = 1; g.ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2); }
    if (o === highlight) { g.ctx.strokeStyle = C.white; g.ctx.lineWidth = 1.6; g.ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1); }
  }
  const s = r.steps[step];
  g.text(`${s.running} yêu cầu đang chạy · ${s.queued} đang chờ · ${vn(utilization(r, step) * 100, 1)}% bộ nhớ đã cấp chứa token thật`, rect.x + 14, oy + ROWS * cell + 16, { size: 11.5, color: C.ink });
  return { ox, oy, cell };
}
function legend(f: Frame, x: number, y: number) {
  const { g } = f;
  g.ctx.fillStyle = alpha(PALETTE[0], 0.9); g.ctx.fillRect(x, y - 5, 10, 10); g.text('chứa token', x + 15, y, { size: 11, color: C.muted });
  g.ctx.strokeStyle = alpha(PALETTE[0], 0.5); g.ctx.strokeRect(x + 92, y - 5, 10, 10); g.text('đã giữ nhưng trống', x + 107, y, { size: 11, color: C.muted });
  g.ctx.fillStyle = 'rgba(255,255,255,0.08)'; g.ctx.fillRect(x + 232, y - 5, 10, 10); g.text('trống', x + 247, y, { size: 11, color: C.muted });
}

// ---------- Chapter 1: KV cache size ----------
const REF = INFER.reference;
const chapter1: Scene = {
  title: 'KV cache lớn cỡ nào?',
  duration: 10,
  beats: () => [
    { at: 0, text: 'Mỗi token đã xử lý để lại một key và một value ở mỗi lớp, để các token sau không phải tính lại. Đó là KV cache.' },
    { at: 2.5, text: `Kích thước mỗi token = 2 × số lớp × số đầu KV × số chiều mỗi đầu × số byte. Đo thật trên Qwen2.5-0.5B (float32): ${vn(INFER.kvBytesPerTokenFp32, 0)} byte/token, khớp công thức.` },
    { at: 5.5, text: `Llama-2-13B không dùng GQA: ${mb(REF[3].kvBytesPerTokenFp16)} mỗi token ở fp16. Một cuộc hội thoại 2 048 token chiếm ${vn(REF[3].kvBytesPerTokenFp16 * 2048 / 1e9, 2)} GB.` },
    { at: 8, text: 'Cache lớn lên theo từng token sinh ra và độ dài cuối thì không biết trước. Cấp phát nó thế nào quyết định phục vụ được bao nhiêu người.' },
  ],
  formula: `byte/token = 2 (K, V) × L × H_kv × d_head × byte\nQwen2.5-0.5B fp32: 2 × 24 × 2 × 64 × 4 = ${vn(2 * 24 * 2 * 64 * 4, 0)} byte (đo được ${vn(INFER.kvBytesPerTokenFp32, 0)})`,
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.1, 1], narrow);
    const names = REF.map(r => r.name.replace('Qwen2.5-', 'Qwen ').replace('Llama-', 'Llama '));
    barChart(f, left, 'KV cache mỗi token (fp16)', names, REF.map(r => r.kvBytesPerTokenFp16 / 1000), { highlight: 3, reveal: seg(t, 2.5, 6), format: v => `${vn(v, 0)} KB` });
    barChart(f, right, 'Một chuỗi 2 048 token chiếm (fp16)', names, REF.map(r => r.kvBytesPerTokenFp16 * 2048 / 1e9), { highlight: 3, reveal: seg(t, 5.5, 8), format: v => `${vn(v, 2)} GB`, color: C.magenta, accent: C.magenta });
  },
  metrics: () => [{ label: 'Đo (Qwen 0,5B fp32)', value: `${vn(INFER.kvBytesPerTokenFp32 / 1000, 1)} KB/token`, tone: 'cyan' }, { label: 'Llama-2-13B fp16', value: `${vn(REF[3].kvBytesPerTokenFp16 / 1000, 0)} KB/token`, tone: 'amber' }],
  table: () => ({ headers: ['Mô hình', 'Lớp', 'Đầu KV', 'd_head', 'Byte/token (fp16)', '2 048 token'], rows: REF.map(r => [r.name, r.layers, r.kvHeads, r.headDim, vn(r.kvBytesPerTokenFp16, 0), `${vn(r.kvBytesPerTokenFp16 * 2048 / 1e9, 3)} GB`]) }),
};

// ---------- Chapter 2: contiguous reservation ----------
const chapter2: Scene = {
  title: 'Cấp phát liền mạch: lãng phí và phân mảnh',
  duration: 12,
  idle: false,
  beats: () => [
    { at: 0, text: `Cách cũ: mỗi yêu cầu giữ trước một vùng liền mạch đủ cho độ dài tối đa ${vn(MAX_LEN, 0)} token, vì không biết câu trả lời dài bao nhiêu.` },
    { at: 3, text: `Độ dài thật lấy từ Dolly-15k: trung vị câu hỏi ${INFER.workload.prompt['50']} token, câu trả lời ${INFER.workload.output['50']} token. Phần lớn vùng giữ trước để trống (ô viền).` },
    { at: 6.5, text: 'Ô bên phải biết trước đúng độ dài nên giữ vừa đủ, nhưng khi các yêu cầu xong ở thời điểm khác nhau, bộ nhớ trống bị cắt vụn thành các khe không đủ liền mạch.' },
    { at: 9.5, text: `Kết quả: chỉ ${vn(avgRunning('reserve'), 1)} yêu cầu chạy cùng lúc khi giữ 2 048 token, ${vn(avgRunning('exact'), 1)} khi biết trước độ dài.` },
  ],
  formula: 'lãng phí trong (internal) = giữ trước nhưng chưa dùng · lãng phí ngoài (external) = khe trống quá nhỏ để cấp',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, [1, 1], narrow);
    const step = stepNow(1, 11);
    memory(f, left, run('reserve'), step, `Giữ trước 2 048 token · bước ${step}`, C.red);
    memory(f, right, run('exact'), step, `Giữ đúng độ dài (biết trước) · bước ${step}`, C.amber);
    legend(f, left.x + 14, left.y + left.h - 14);
  },
  metrics: () => { const i = stepNow(1, 11); return [{ label: 'Bước giải mã', value: String(i), tone: 'violet' }, { label: 'Giữ 2 048: dùng thật', value: `${vn(utilization(run('reserve'), i) * 100, 0)}%`, tone: 'red' }, { label: 'Biết trước: dùng thật', value: `${vn(utilization(run('exact'), i) * 100, 0)}%`, tone: 'amber' }]; },
};
function avgRunning(s: Strategy) { const r = run(s); return r.steps.slice(100).reduce((a, x) => a + x.running, 0) / (r.steps.length - 100); }
function avgUtil(s: Strategy) { const r = run(s); let a = 0; for (let i = 100; i < r.steps.length; i++) a += utilization(r, i); return a / (r.steps.length - 100); }

// ---------- Chapter 3: paging ----------
const chapter3: Scene = {
  title: 'Chia KV cache thành trang',
  duration: 13,
  idle: false,
  beats: () => [
    { at: 0, text: `PagedAttention cắt KV cache của mỗi chuỗi thành các khối cố định ${BLOCK} token. Khối có thể nằm ở bất cứ đâu trong bộ nhớ GPU.` },
    { at: 3, text: 'Mỗi yêu cầu có một bảng khối (block table) ánh xạ khối logic 0, 1, 2… sang khối vật lý, giống bảng trang của hệ điều hành.' },
    { at: 6.5, text: 'Khối mới chỉ được cấp khi chuỗi đầy khối cũ. Lãng phí duy nhất là phần chưa đầy của khối cuối: dưới 16 token mỗi yêu cầu.' },
    { at: 9.5, text: `Kết quả trên cùng luồng yêu cầu: trung bình ${vn(avgRunning('paged'), 1)} yêu cầu chạy cùng lúc, ${vn(avgUtil('paged') * 100, 1)}% bộ nhớ đã cấp chứa token thật.` },
  ],
  formula: 'khối vật lý = block_table[vị trí token ÷ 16],  ô trong khối = vị trí token mod 16\nkernel attention đọc K, V theo bảng khối thay vì một mảng liền mạch',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.5, 1] : [1.75, 1], narrow);
    const step = stepNow(1, 12), r = run('paged'), table = r.tableAt(step);
    const m = memory(f, left, r, step, `PagedAttention · bước ${step}`, C.lime, table?.id ?? -1);
    legend(f, left.x + 14, left.y + left.h - 14);
    g.panel(right.x, right.y, right.w, right.h, table ? `Bảng khối của yêu cầu #${table.id}` : 'Bảng khối', C.cyan);
    if (!table) return;
    const req = r.reqs[table.id];
    g.text(`${req.prompt} token câu hỏi + ${Math.min(req.generated, req.output)}/${req.output} token trả lời`, right.x + 14, right.y + 50, { size: 11.5, color: C.muted });
    const rowH = Math.min(24, (right.h - 90) / Math.max(1, table.blocks.length));
    table.blocks.slice(0, Math.floor((right.h - 90) / rowH)).forEach((b, k) => {
      const y = right.y + 70 + k * rowH;
      g.text(`logic ${k}`, right.x + 16, y + rowH / 2, { size: 11.5, color: C.ink, mono: true });
      g.text(`→ vật lý ${b}`, right.x + 92, y + rowH / 2, { size: 11.5, color: PALETTE[table.id % PALETTE.length], mono: true });
      const filled = r.fill[step][b];
      g.ctx.fillStyle = 'rgba(255,255,255,0.08)'; g.ctx.fillRect(right.x + right.w - 86, y + 4, 70, rowH - 8);
      g.ctx.fillStyle = alpha(PALETTE[table.id % PALETTE.length], 0.85); g.ctx.fillRect(right.x + right.w - 86, y + 4, 70 * filled / BLOCK, rowH - 8);
      if (t > 3 && k < 4) { const x = m.ox + (b % COLS) * m.cell + m.cell / 2, yy = m.oy + Math.floor(b / COLS) * m.cell + m.cell / 2; g.line([[right.x + 2, y + rowH / 2], [x, yy]], alpha(C.white, 0.25 * seg(t, 3, 4)), { width: 1 }); }
    });
  },
  metrics: () => { const i = stepNow(1, 12), r = run('paged'); return [{ label: 'Đang chạy', value: String(r.steps[i].running), tone: 'lime' }, { label: 'Dùng thật', value: `${vn(utilization(r, i) * 100, 0)}%`, tone: 'cyan' }, { label: 'Bộ nhớ đã cấp', value: `${vn(r.steps[i].allocated / BLOCKS * 100, 0)}%`, tone: 'violet' }]; },
};

// ---------- Chapter 4: throughput ----------
const chapter4: Scene = {
  title: 'Nhiều người dùng hơn trên cùng GPU',
  duration: 12,
  beats: () => [
    { at: 0, text: `Cùng ${INFER.workload.sample.length} yêu cầu thật, cùng ${BLOCKS} khối bộ nhớ (${vn(BLOCKS * BLOCK, 0)} token), cùng tốc độ đến. Chỉ khác cách cấp phát.` },
    { at: 3, text: 'Số yêu cầu chạy song song quyết định thông lượng: mỗi bước giải mã sinh một token cho mỗi yêu cầu trong batch.' },
    { at: 7, text: `Sau ${STEPS} bước: ${vn(run('paged').steps.at(-1)!.generated / run('reserve').steps.at(-1)!.generated, 1)}× số token so với giữ 2 048, ${vn(run('paged').steps.at(-1)!.generated / run('exact').steps.at(-1)!.generated, 1)}× so với biết trước độ dài. Bài báo vLLM đo được 2–4× trên GPU thật.` },
  ],
  formula: 'thông lượng ≈ (số yêu cầu trong batch) × (token mỗi bước) / (thời gian mỗi bước)\nbước giải mã bị giới hạn bởi băng thông bộ nhớ, nên batch lớn hơn gần như “miễn phí”',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.2, 1], narrow);
    const cursor = Math.min(STEPS - 1, Math.floor(seg(t, 0.5, 6) * (STEPS - 1)));
    const colors = [C.red, C.amber, C.lime];
    lossChart(f, left, STRATEGIES.map((s, k) => ({ values: run(s.key).steps.map(x => x.running).slice(0, cursor + 1), color: colors[k] })), cursor, 100, null, 'Số yêu cầu chạy cùng lúc theo bước', 'bước giải mã', STEPS - 1);
    barChart(f, right, `Token sinh ra sau ${STEPS} bước`, ['giữ 2 048', 'biết trước', 'paged'], STRATEGIES.map(s => run(s.key).steps.at(-1)!.generated), { colors: colors, reveal: seg(t, 6, 8), format: v => vn(v, 0), accent: C.lime });
    f.g.text('đỏ: giữ 2 048 · cam: biết trước · xanh: paged', left.x + 14, left.y + left.h - 12, { size: 11, color: C.muted });
  },
  metrics: () => STRATEGIES.map((s, k) => ({ label: s.key === 'reserve' ? 'Giữ 2 048' : s.key === 'exact' ? 'Biết trước' : 'Paged', value: `${vn(avgRunning(s.key), 0)} yêu cầu`, tone: (['red', 'amber', 'lime'] as const)[k] })),
  table: () => ({ headers: ['Cách cấp phát', 'TB yêu cầu đồng thời', 'Bộ nhớ đã cấp chứa token', 'Token sinh ra', 'Yêu cầu xong', 'Bị ưu tiên lại'], rows: STRATEGIES.map(s => { const r = run(s.key), last = r.steps.at(-1)!; return [s.label, vn(avgRunning(s.key), 1), `${vn(avgUtil(s.key) * 100, 1)}%`, vn(last.generated, 0), last.finished, last.preempted]; }) }),
};

// ---------- Chapter 5: prefix sharing ----------
const PROMPT = 100, OUT = 60;
const chapter5: Scene = {
  title: 'Chia sẻ tiền tố và copy-on-write',
  duration: 11,
  beats: () => {
    const s = sharing(PROMPT, state.samples, OUT);
    return [
      { at: 0, text: `Sinh ${state.samples} câu trả lời cho cùng một câu hỏi ${PROMPT} token (parallel sampling, beam search).` },
      { at: 2.5, text: `${s.full} khối đầy của câu hỏi được dùng chung: nhiều bảng khối trỏ tới cùng khối vật lý, kèm bộ đếm tham chiếu.` },
      { at: 5.5, text: 'Khối cuối của câu hỏi còn chỗ trống. Khi một mẫu ghi token đầu tiên vào đó, khối được sao chép riêng (copy-on-write).' },
      { at: 8, text: `Tổng: ${s.shared} khối thay vì ${s.separate}, tiết kiệm ${vn(s.saving * 100, 0)}% bộ nhớ KV.` },
    ];
  },
  formula: () => { const s = sharing(PROMPT, state.samples, OUT); return `riêng: ${state.samples} × ⌈(${PROMPT} + ${OUT}) / 16⌉ = ${s.separate} khối\nchia sẻ: ${s.full} khối chung + ${state.samples} × ${s.perSample} khối riêng = ${s.shared} khối`; },
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const s = sharing(PROMPT, state.samples, OUT), n = state.samples;
    const area = f.safe;
    g.panel(area.x, area.y, area.w, area.h, `${n} mẫu dùng chung tiền tố ${PROMPT} token`, C.violet);
    const cell = Math.min(34, (area.w - 220) / (s.full + s.perSample + 2)), rowH = Math.min(60, (area.h - 120) / (n + 1));
    const x0 = area.x + (narrow ? 70 : 120), y0 = area.y + 60;
    const share = seg(t, 2.5, 4), cow = seg(t, 5.5, 7);
    for (let k = 0; k < n; k++) {
      const y = y0 + (k + 1) * rowH;
      g.text(`mẫu ${k + 1}`, x0 - 14, y + cell / 2, { size: 12, color: C.ink, align: 'right' });
      for (let b = 0; b < s.full + s.perSample; b++) {
        const shared = b < s.full, x = x0 + b * (cell + 6);
        const color = shared ? C.cyan : PALETTE[(k + 2) % PALETTE.length];
        const yy = shared ? y + (y0 - y) * share : y;
        if (shared && k > 0 && share > 0.98) continue;
        g.ctx.fillStyle = alpha(color, shared ? 0.75 : b === s.full ? 0.4 + 0.5 * cow : 0.7);
        g.ctx.fillRect(x, yy, cell, cell);
        if (b === s.full && cow > 0.5) g.text('CoW', x + cell / 2, yy + cell / 2, { size: 9, color: '#081022', align: 'center', weight: 700 });
        if (shared && share > 0.98) g.text(`×${n}`, x + cell / 2, yy + cell / 2, { size: 10, color: '#081022', align: 'center', weight: 700 });
      }
    }
    if (share > 0.98) g.text('khối chung (đếm tham chiếu)', x0, y0 - 10, { size: 11.5, color: C.cyan });
    const bar = { x: area.x + 20, y: area.y + area.h - 46, w: area.w - 40 };
    const reveal = seg(t, 8, 9.5);
    g.ctx.fillStyle = alpha(C.red, 0.7); g.ctx.fillRect(bar.x, bar.y, bar.w * reveal, 10);
    g.ctx.fillStyle = alpha(C.lime, 0.85); g.ctx.fillRect(bar.x, bar.y + 16, bar.w * s.shared / s.separate * reveal, 10);
    g.text(`riêng ${s.separate} khối`, bar.x + 4, bar.y - 8, { size: 11, color: C.ink });
    g.text(`chia sẻ ${s.shared} khối (−${vn(s.saving * 100, 0)}%)`, bar.x + bar.w * s.shared / s.separate * reveal + 8, bar.y + 22, { size: 11, color: C.lime });
  },
  metrics: () => { const s = sharing(PROMPT, state.samples, OUT); return [{ label: 'Không chia sẻ', value: `${s.separate} khối`, tone: 'red' }, { label: 'Chia sẻ', value: `${s.shared} khối`, tone: 'lime' }, { label: 'Tiết kiệm', value: `${vn(s.saving * 100, 0)}%`, tone: 'amber' }]; },
};

export const initPagedAttentionLab = () => glassLab('paged-attention', [chapter1, chapter2, chapter3, chapter4, chapter5], lab => {
  lab.range('step', value => String(value), value => { state.step = value; });
  lab.range('samples', value => String(value), value => { state.samples = value; });
  lab.on('[data-autoplay]', 'click', () => { state.step = null; lab.replay(); });
});
