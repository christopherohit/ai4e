// Inference engineering lab: prefill vs decode on a real model, what the KV cache saves, why decode is memory-bound
// (roofline), why batching and continuous batching raise throughput, and the learning roadmap.
import { glassLab, type Scene } from '../../_shared/glass-lab/engine';
import { C, Plot, alpha, seg, split, vn, type Point } from '../../_shared/glass-lab/gfx';
import { barChart, categoryChart, lossChart } from '../../_shared/glass-lab/views';
import { GPUS, M, OUTPUTS, PREFILL_RATE, ROADMAP, TPOT, TTFT_128, attainable, busy, continuousBatching, decodeIntensity, prefillIntensity, ridge, staticBatching } from './model';

export const explainerKind = 'measured-replay';

const state = { batch: 4, gpu: 1, slots: 6 };
const ms = (s: number) => `${vn(s * 1000, 0)} ms`;

// ---------- Chapter 1: prefill and decode ----------
const N_OUT = 40;
const chapter1: Scene = {
  title: 'Một lần gọi LLM: prefill rồi decode',
  duration: 12,
  beats: () => [
    { at: 0, text: `Gửi một prompt 128 token cho ${M.model.split('/')[1]} (đo thật trên CPU). Giai đoạn 1 là prefill: cả 128 token đi qua mô hình cùng lúc và xây KV cache.` },
    { at: 3, text: `Token đầu tiên xuất hiện sau ${ms(TTFT_128)}: đó là TTFT (time to first token).` },
    { at: 5, text: `Giai đoạn 2 là decode: mỗi token mới cần một lượt forward riêng, trung bình ${ms(TPOT)} một token (ITL / TPOT).` },
    { at: 9, text: `Độ trễ cho ${N_OUT} token = TTFT + (n − 1) × TPOT ≈ ${vn(TTFT_128 + (N_OUT - 1) * TPOT, 2)} s. Người dùng cảm nhận TTFT; thông lượng do decode quyết định.` },
  ],
  formula: `độ trễ = TTFT + (n − 1) · TPOT = ${ms(TTFT_128)} + ${N_OUT - 1} × ${ms(TPOT)}`,
  render(f) {
    const { g, t, narrow } = f;
    const area = f.safe;
    g.panel(area.x, area.y, area.w, area.h, 'Dòng thời gian của một yêu cầu (thời gian thật, CPU)', C.violet);
    const x0 = area.x + 20, w = area.w - 40, total = TTFT_128 + N_OUT * TPOT, px = (s: number) => x0 + s / total * w;
    const y = area.y + area.h * 0.42, h = 40, now = seg(t, 0.5, 10) * total;
    // prefill block
    g.ctx.fillStyle = alpha(C.amber, 0.2); g.ctx.fillRect(px(0), y, px(Math.min(now, TTFT_128)) - px(0), h);
    g.ctx.strokeStyle = alpha(C.amber, 0.8); g.ctx.strokeRect(px(0), y, px(TTFT_128) - px(0), h);
    g.text('Prefill · 128 token song song', px(TTFT_128 / 2), y + h / 2, { size: narrow ? 10.5 : 12.5, color: C.ink, align: 'center', weight: 650 });
    for (let i = 0; i < N_OUT; i++) {
      const a = TTFT_128 + i * TPOT, b = a + TPOT;
      if (now < a) break;
      g.ctx.fillStyle = alpha(C.cyan, 0.25 + 0.5 * Math.min(1, (now - a) / TPOT)); g.ctx.fillRect(px(a) + 1, y, px(Math.min(b, now)) - px(a) - 1, h);
    }
    g.text('Decode · từng token', px(TTFT_128 + N_OUT * TPOT / 2), y - 12, { size: 12, color: C.cyan, align: 'center' });
    // input tokens
    for (let i = 0; i < 32; i++) { g.ctx.fillStyle = alpha(C.white, 0.25); g.ctx.fillRect(px(0) + i * ((px(TTFT_128) - px(0)) / 32), y + h + 18, (px(TTFT_128) - px(0)) / 32 - 2, 10); }
    g.text('prompt', px(0), y + h + 44, { size: 11, color: C.muted });
    if (now > TTFT_128) {
      g.line([[px(TTFT_128), y - 30], [px(TTFT_128), y + h + 60]], C.lime, { width: 1.5, dash: [4, 4] });
      g.tag(`TTFT ${ms(TTFT_128)}`, px(TTFT_128), y - 40, C.lime);
    }
    const ticks = 6;
    for (let i = 0; i <= ticks; i++) { const s = total * i / ticks; g.text(`${vn(s, 1)} s`, px(s), area.y + area.h - 22, { size: 10.5, color: C.muted, align: 'center' }); }
    // Measured duration of each decode step (bar height ∝ milliseconds).
    const by = y + h + 110, bh = Math.min(90, area.y + area.h - 60 - by);
    if (bh > 30) {
      g.text('thời gian đo được của từng bước decode', px(TTFT_128), by - 12, { size: 11, color: C.muted });
      const peak = Math.max(...M.decodeWithCache.slice(0, N_OUT));
      for (let i = 0; i < N_OUT; i++) {
        const a = TTFT_128 + i * TPOT; if (now < a) break;
        const v = M.decodeWithCache[i], hh = bh * v / peak;
        g.ctx.fillStyle = alpha(C.cyan, 0.55); g.ctx.fillRect(px(a) + 1, by + bh - hh, px(a + TPOT) - px(a) - 2, hh);
      }
      g.text(`${ms(peak)} (cao nhất)`, px(TTFT_128) - 6, by + 6, { size: 10.5, color: C.muted, align: 'right' });
    }
  },
  metrics: () => [{ label: 'TTFT', value: ms(TTFT_128), tone: 'lime' }, { label: 'TPOT', value: ms(TPOT), tone: 'cyan' }, { label: 'Decode', value: `${vn(1 / TPOT, 1)} token/s`, tone: 'amber' }],
};

// ---------- Chapter 2: prefill is parallel, decode is not ----------
const chapter2: Scene = {
  title: 'Prefill nhanh theo token, decode thì không',
  duration: 10,
  beats: () => [
    { at: 0, text: 'Thời gian prefill tăng gần tuyến tính theo độ dài prompt (đo từ 16 tới 2 048 token).' },
    { at: 3, text: `Nhưng tính theo token, prefill xử lý khoảng ${vn(PREFILL_RATE, 0)} token/giây, trong khi decode chỉ ${vn(1 / TPOT, 0)} token/giây.` },
    { at: 6.5, text: 'Prefill là các phép nhân ma trận lớn (nhiều token cùng lúc): tận dụng tốt khả năng tính toán. Decode chỉ nhân ma trận với một vector: phần lớn thời gian là đọc trọng số từ bộ nhớ.' },
  ],
  formula: `prefill 2 048 token: ${vn(M.prefill.at(-1)!.seconds, 2)} s → ${vn(PREFILL_RATE, 0)} token/s\ndecode: ${ms(TPOT)}/token → ${vn(1 / TPOT, 1)} token/s`,
  render(f) {
    const { t, narrow } = f;
    const [left, right] = split(f.safe, [1, 1], narrow);
    lossChart(f, left, [{ values: M.prefill.map(p => p.seconds), color: C.amber, x: M.prefill.map(p => p.tokens) }], M.prefill.at(-1)!.tokens * seg(t, 0.5, 3), 4.6, null, 'Thời gian prefill theo độ dài prompt (s)', 'token trong prompt', 2048, v => vn(v, 0));
    barChart(f, right, 'Token xử lý mỗi giây (đo thật)', ['prefill', 'decode'], [PREFILL_RATE, 1 / TPOT], { colors: [C.amber, C.cyan], reveal: seg(t, 3, 5), format: v => `${vn(v, 0)} token/s`, accent: C.lime });
  },
  metrics: () => [{ label: 'Prefill', value: `${vn(PREFILL_RATE, 0)} token/s`, tone: 'amber' }, { label: 'Decode', value: `${vn(1 / TPOT, 1)} token/s`, tone: 'cyan' }, { label: 'Chênh lệch', value: `×${vn(PREFILL_RATE * TPOT, 0)}`, tone: 'lime' }],
  table: () => ({ headers: ['Token prompt', 'Prefill (s)', 'Token/s'], rows: M.prefill.map(p => [p.tokens, vn(p.seconds, 3), vn(p.tokens / p.seconds, 0)]) }),
};

// ---------- Chapter 3: KV cache ----------
const chapter3: Scene = {
  title: 'KV cache: không tính lại quá khứ',
  duration: 10,
  beats: () => [
    { at: 0, text: 'Không có KV cache, mỗi bước decode phải chạy lại toàn bộ chuỗi từ đầu: chi phí tăng dần theo độ dài.' },
    { at: 3.5, text: `Đo thật sau prompt 128 token: không cache, bước đầu ${ms(M.decodeWithoutCache[0])}, bước thứ 96 đã ${ms(M.decodeWithoutCache.at(-1)!)}.` },
    { at: 6.5, text: `Có cache, mỗi bước chỉ tính token mới và đọc K, V cũ: gần như không đổi, khoảng ${ms(TPOT)}. Đổi lại phải trả bằng bộ nhớ (${vn(M.kvBytesPerTokenFp32 / 1000, 1)} KB mỗi token ở fp32).` },
  ],
  formula: 'không cache: bước thứ n tính lại n token  →  tổng O(n²)\ncó cache: bước thứ n tính 1 token, đọc n cặp K, V  →  tổng O(n) phép tính',
  render(f) {
    const { t } = f;
    const cursor = Math.floor(seg(t, 0.5, 8) * 95);
    lossChart(f, f.safe, [{ values: M.decodeWithoutCache.slice(0, cursor + 1), color: C.red }, { values: M.decodeWithCache.slice(0, cursor + 1), color: C.lime }], cursor, 0.42, null, 'Thời gian mỗi bước decode (s): đỏ không cache, xanh có cache', 'bước decode', 95);
  },
  metrics: () => [{ label: 'Không cache, bước cuối', value: ms(M.decodeWithoutCache.at(-1)!), tone: 'red' }, { label: 'Có cache, trung bình', value: ms(TPOT), tone: 'lime' }, { label: 'Tổng 96 bước', value: `${vn(M.decodeWithoutCache.reduce((a, b) => a + b, 0), 1)} s so với ${vn(M.decodeWithCache.reduce((a, b) => a + b, 0), 1)} s`, tone: 'amber' }],
};

// ---------- Chapter 4: roofline and batching ----------
const P7B = 7.6e9, KV7B = 57344;
const chapter4: Scene = {
  title: 'Memory-bound hay compute-bound?',
  duration: 13,
  idle: false,
  beats: () => [
    { at: 0, text: 'Roofline: tốc độ tối đa = min(đỉnh tính toán, băng thông × cường độ số học). Cường độ = số FLOP trên mỗi byte đọc từ bộ nhớ.' },
    { at: 3, text: `Decode một chuỗi với mô hình 7,6B tham số: mỗi trọng số 2 byte chỉ dùng cho 2 FLOP, cường độ ≈ 1. Rất xa điểm gãy của ${GPUS[state.gpu].name} (≈ ${vn(ridge(GPUS[state.gpu]), 0)} FLOP/byte): GPU chủ yếu chờ bộ nhớ.` },
    { at: 7, text: 'Gom B chuỗi vào một batch: trọng số đọc một lần dùng cho B token. Cường độ tăng gần B lần, thông lượng tăng theo, tới khi chạm trần tính toán.' },
    { at: 10, text: `Đo thật trên CPU: batch 1 → ${vn(M.batch[0].tokensPerSecond, 0)} token/s, batch 64 → ${vn(M.batch.at(-1)!.tokensPerSecond, 0)} token/s, trong khi mỗi bước chỉ chậm hơn ${vn(M.batch.at(-1)!.stepSeconds / M.batch[0].stepSeconds, 1)} lần.` },
  ],
  formula: () => { const g = GPUS[state.gpu], ai = decodeIntensity(P7B, state.batch, 1024, KV7B); return `FLOP/s đạt được = min(${g.tflops} TFLOP/s, ${g.tbps} TB/s × cường độ)\ndecode 7,6B, batch ${state.batch}, ngữ cảnh 1 024: cường độ ≈ ${vn(ai, 1)} FLOP/byte → ${vn(attainable(g, ai) / 1e12, 1)} TFLOP/s`; },
  render(f) {
    const { g, t, narrow } = f;
    const [left, right] = split(f.safe, narrow ? [1.2, 1] : [1.3, 1], narrow);
    const gpu = GPUS[state.gpu];
    g.panel(left.x, left.y, left.w, left.h, `Roofline · ${gpu.name}`, C.violet);
    const plot = new Plot(g, left, [-0.5, 4], [11, 15.3], { l: 54, r: 16, t: 40, b: 36 });
    plot.axes({ xTicks: [0, 1, 2, 3, 4], yTicks: [11, 12, 13, 14, 15], xLabel: 'cường độ số học (FLOP/byte, log)', yLabel: 'FLOP/s (log)', fx: v => `10${['⁰', '¹', '²', '³', '⁴'][Math.round(v)] ?? ''}`, fy: v => `10${['¹¹', '¹²', '¹³', '¹⁴', '¹⁵'][Math.round(v) - 11] ?? ''}` });
    const roof: Point[] = [];
    for (let x = -0.5; x <= 4; x += 0.02) roof.push([x, Math.log10(attainable(gpu, 10 ** x))]);
    plot.line(roof, C.white, { width: 2.4, glow: 10 });
    g.tag(`điểm gãy ≈ ${vn(ridge(gpu), 0)}`, plot.px(Math.log10(ridge(gpu))), plot.py(Math.log10(gpu.tflops * 1e12)) - 16, C.white);
    const batches = [1, 4, 16, 64, 256];
    batches.forEach((b, k) => {
      if (t < 3 + k * 1.2 && b !== 1) return;
      const ai = decodeIntensity(P7B, b, 1024, KV7B), y = Math.log10(attainable(gpu, ai));
      g.circle(plot.px(Math.log10(ai)), plot.py(y), b === state.batch ? 8 : 5, b === state.batch ? C.amber : C.cyan, { glow: 10 });
      g.text(`B=${b}`, plot.px(Math.log10(ai)) + 8, plot.py(y) + 14, { size: 11, color: C.ink });
    });
    const sel = decodeIntensity(P7B, state.batch, 1024, KV7B);
    g.circle(plot.px(Math.log10(sel)), plot.py(Math.log10(attainable(gpu, sel))), 8, C.amber, { glow: 14 });
    const pre = prefillIntensity(P7B, 2048);
    g.circle(plot.px(Math.min(3.95, Math.log10(pre))), plot.py(Math.log10(attainable(gpu, pre))), 6, C.lime, { glow: 10 });
    g.text('prefill 2 048 token', plot.px(Math.min(3.95, Math.log10(pre))) - 6, plot.py(Math.log10(attainable(gpu, pre))) + 18, { size: 11, color: C.lime, align: 'right' });
    g.text('memory-bound', plot.px(0.2), plot.py(12.2), { size: 12, color: C.red });
    g.text('compute-bound', plot.px(3), plot.py(15.1), { size: 12, color: C.lime });
    categoryChart(f, right, 'Đo thật trên CPU: token/s theo batch', M.batch.map(b => String(b.batch)), [{ label: 'token/s', values: M.batch.map(b => b.tokensPerSecond), color: C.amber, format: v => vn(v, 0) }], [0, 360], M.batch.findIndex(b => b.batch >= state.batch), seg(t, 9.5, 11.5), v => vn(v, 0), 'batch');
  },
  metrics: () => { const g = GPUS[state.gpu], ai = decodeIntensity(P7B, state.batch, 1024, KV7B); return [{ label: 'Batch', value: String(state.batch), tone: 'amber' }, { label: 'Cường độ', value: `${vn(ai, 1)} FLOP/B`, tone: 'violet' }, { label: 'Tận dụng đỉnh', value: `${vn(attainable(g, ai) / (g.tflops * 1e12) * 100, 1)}%`, tone: ai < ridge(g) ? 'red' : 'lime' }]; },
  table: () => ({ headers: ['Batch (CPU)', 'Thời gian mỗi bước', 'Token/s'], rows: M.batch.map(b => [b.batch, ms(b.stepSeconds), vn(b.tokensPerSecond, 1)]) }),
};

// ---------- Chapter 5: continuous batching ----------
const chapter5: Scene = {
  title: 'Static batching và continuous batching',
  duration: 12,
  idle: false,
  beats: () => {
    const s = staticBatching(OUTPUTS, state.slots), c = continuousBatching(OUTPUTS, state.slots);
    return [
      { at: 0, text: `${OUTPUTS.length} yêu cầu với độ dài câu trả lời thật (Dolly-15k), batch ${state.slots} chỗ.` },
      { at: 2.5, text: 'Static batching: cả batch phải chờ câu trả lời dài nhất xong mới nhận nhóm mới. Các chỗ đã xong ngồi không (vùng tối).' },
      { at: 6, text: 'Continuous batching (Orca, vLLM): sau mỗi bước decode, chỗ nào xong được lấp ngay bằng yêu cầu đang chờ.' },
      { at: 9, text: `Cùng công việc: ${vn(s.makespan, 0)} bước so với ${vn(c.makespan, 0)} bước; tận dụng ${vn(busy(s, state.slots) * 100, 0)}% so với ${vn(busy(c, state.slots) * 100, 0)}%.` },
    ];
  },
  formula: () => { const s = staticBatching(OUTPUTS, state.slots), c = continuousBatching(OUTPUTS, state.slots); return `static: ${s.makespan} bước · continuous: ${c.makespan} bước (×${vn(s.makespan / c.makespan, 2)})`; },
  render(f) {
    const { g, t } = f;
    const [top, bottom] = split(f.safe, [1, 1], true, 12);
    const s = staticBatching(OUTPUTS, state.slots), c = continuousBatching(OUTPUTS, state.slots), max = Math.max(s.makespan, c.makespan);
    const draw = (rect: typeof top, r: typeof s, title: string, accent: string, from: number) => {
      g.panel(rect.x, rect.y, rect.w, rect.h, title, accent);
      const x0 = rect.x + 16, w = rect.w - 32, y0 = rect.y + 44, lh = (rect.h - 64) / state.slots, now = seg(t, from, from + 3.5) * max;
      g.ctx.fillStyle = 'rgba(255,255,255,0.04)'; g.ctx.fillRect(x0, y0, w * Math.min(now, r.makespan) / max, lh * state.slots);
      r.lanes.forEach(l => {
        if (l.start > now) return;
        const end = Math.min(l.end, now), color = ['#4fd6ff', '#7af0a0', '#ffb648', '#a98bff', '#ff5fa8', '#5b8cff'][l.id % 6];
        g.ctx.fillStyle = alpha(color, 0.75); g.ctx.fillRect(x0 + w * l.start / max + 0.5, y0 + l.slot * lh + 2, Math.max(1, w * (end - l.start) / max - 1), lh - 4);
      });
      g.text(`${r.makespan} bước`, x0 + w * r.makespan / max - 4, y0 - 8, { size: 11, color: C.ink, align: 'right' });
    };
    draw(top, s, 'Static batching', C.red, 1);
    draw(bottom, c, 'Continuous batching (iteration-level)', C.lime, 5);
  },
  metrics: () => { const s = staticBatching(OUTPUTS, state.slots), c = continuousBatching(OUTPUTS, state.slots); return [{ label: 'Static', value: `${s.makespan} bước`, tone: 'red' }, { label: 'Continuous', value: `${c.makespan} bước`, tone: 'lime' }, { label: 'Nhanh hơn', value: `×${vn(s.makespan / c.makespan, 2)}`, tone: 'amber' }]; },
};

// ---------- Chapter 6: roadmap ----------
const chapter6: Scene = {
  title: 'Lộ trình học Inference Engineering',
  duration: 8,
  beats: () => [
    { at: 0, text: 'Năm chặng: nền tảng suy luận, khái niệm Transformer, phần cứng GPU, kỹ thuật tối ưu và các engine phục vụ.' },
    { at: 3, text: 'Mới bắt đầu: tập trung chặng 1 tới 4. Kỹ sư nhiều kinh nghiệm: đi tiếp sang song song hóa và suy luận phân tán.' },
    { at: 5.5, text: 'Các bài viết khác trên trang đi sâu từng mục: KV cache và PagedAttention, quantization, MoE, giải mã ràng buộc kiểu Jev.' },
  ],
  metrics: () => [{ label: 'Chặng', value: String(ROADMAP.length), tone: 'violet' }, { label: 'Kỹ thuật tối ưu', value: String(ROADMAP[3].items.length), tone: 'cyan' }],
  render(f) {
    const { g, t, narrow } = f;
    const area = f.safe, colors = [C.magenta, C.violet, C.amber, C.cyan, C.lime];
    const n = ROADMAP.length, gap = 10, h = narrow ? (area.h - gap * (n - 1)) / n : area.h, w = narrow ? area.w : (area.w - gap * (n - 1)) / n;
    ROADMAP.forEach((stage, k) => {
      const on = seg(t, k * 0.6, k * 0.6 + 1);
      const x = narrow ? area.x : area.x + k * (w + gap), y = narrow ? area.y + k * (h + gap) : area.y;
      g.ctx.globalAlpha = on;
      g.panel(x, y, w, h, `${k + 1}. ${stage.title}`, colors[k]);
      stage.items.forEach((item, i) => {
        const yy = y + 52 + i * (narrow ? 18 : 30);
        if (yy > y + h - 10) return;
        g.circle(x + 18, yy, 3, colors[k]);
        g.text(item, x + 28, yy, { size: narrow ? 11 : 12.5, color: C.ink });
      });
      g.ctx.globalAlpha = 1;
    });
  },
};

export const initInferenceLab = () => glassLab('inference', [chapter1, chapter2, chapter3, chapter4, chapter5, chapter6], lab => {
  const batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
  lab.range('batch', value => String(batches[value]), value => { state.batch = batches[value]; });
  lab.range('slots', value => String(value), value => { state.slots = value; });
  lab.on('[data-gpu]', 'click', el => { state.gpu = Number(el.dataset.gpu); document.querySelectorAll<HTMLElement>('[data-gpu]').forEach(b => b.setAttribute('aria-pressed', String(b === el))); lab.invalidate(); });
});
