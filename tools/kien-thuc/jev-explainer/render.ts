// Jev-style parallel constrained decoding, replayed from real measurements (Qwen2.5-1.5B-Instruct on CPU): token-by-
// token JSON generation, one prefill into the KV cache, scoring one field through hidden state → LM head → candidate
// logits → softmax, and every field in one batched pass compared with autoregressive decoding.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, alpha, clamp, seg, split, vn } from '../../_shared/glass-lab/gfx';
import { FIELDS, FIELD_LABEL, JEV, invalidReason, kvBytes, parallelJson, type Doc } from './model';

export const explainerKind = 'measured-replay';

const state = { doc: 0, field: 0 };
let currentT = 0;
const doc = (): Doc => JEV.docs[state.doc];
const FIELD_COLOR = [C.amber, C.lime, C.magenta];
/** Show a token with its spaces visible: ‹␣true›. */
const tk = (t: string) => `‹${t.replace(/ /g, '␣')}›`;
const arValue = (d: Doc, field: string) => (d.autoregressive.text.match(new RegExp(`"${field}":\\s*"?([A-Za-z_0-9]+)`))?.[1] ?? '').toLowerCase();

/** Wrap text into lines that fit `width` at font `size`. */
function wrap(f: Frame, text: string, width: number, size: number) {
  const words = text.split(/\s+/), lines: string[] = [];
  let line = '';
  for (const w of words) { const next = line ? `${line} ${w}` : w; if (f.g.measure(next, size) > width && line) { lines.push(line); line = w; } else line = next; }
  if (line) lines.push(line);
  return lines;
}
function paragraph(f: Frame, text: string, x: number, y: number, width: number, size = 12.5, color: string = C.ink, maxLines = 99) {
  const lines = wrap(f, text, width, size).slice(0, maxLines);
  lines.forEach((l, i) => f.g.text(l, x, y + i * size * 1.45, { size, color }));
  return y + lines.length * size * 1.45;
}
function box(f: Frame, x: number, y: number, w: number, h: number, title: string, sub: string, color: string, glow = 0) {
  const { g } = f;
  g.ctx.save(); g.ctx.fillStyle = alpha(color, 0.12); g.ctx.strokeStyle = alpha(color, 0.7); g.ctx.lineWidth = 1.4;
  if (glow) { g.ctx.shadowColor = color; g.ctx.shadowBlur = glow; }
  g.ctx.beginPath(); g.ctx.roundRect(x, y, w, h, 10); g.ctx.fill(); g.ctx.stroke(); g.ctx.restore();
  g.text(title, x + w / 2, y + h / 2 - (sub ? 7 : 0), { size: 13, color: C.ink, align: 'center', weight: 650 });
  if (sub) g.text(sub, x + w / 2, y + h / 2 + 10, { size: 11, color: C.muted, align: 'center' });
}

// ---------- Chapter 1: autoregressive ----------
const chapter1: Scene = {
  title: 'Cách thường: sinh JSON từng token',
  duration: 14,
  beats: () => [
    { at: 0, text: 'Đầu vào: một báo cáo sự cố và schema JSON gồm 3 trường. Mô hình thường sẽ viết câu trả lời từng token một.' },
    { at: 2, text: 'Mỗi token mới cần một lượt forward qua toàn bộ 28 lớp Transformer, và lượt sau phải chờ lượt trước.' },
    { at: 8, text: `Ngay cả dấu ngoặc, dấu hai chấm và tên trường cũng tốn một lượt: ${doc().autoregressive.forwardPasses} lượt forward cho ba giá trị.` },
    { at: 11.5, text: doc().autoregressive.valid ? 'Lần này JSON hợp lệ, nhưng không có gì bảo đảm.' : `Kết quả thật không parse được: ${invalidReason(doc())}. Mô hình chép lại kiểu viết enum không ngoặc kép của schema.` },
  ],
  formula: () => `${doc().autoregressive.forwardPasses} lượt forward tuần tự · TTFT ${vn(doc().autoregressive.ttft, 2)} s + ${vn(doc().autoregressive.perToken * 1000, 0)} ms/token = ${vn(doc().autoregressive.seconds, 2)} s (CPU, ${JEV.model.split('/')[1]})`,
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const d = doc();
    const [left, right] = split(f.safe, narrow ? [1, 1.3] : [1, 1.4], narrow);
    g.panel(left.x, left.y, left.w, left.h, `Tài liệu · ${d.title}`, C.violet);
    let y = paragraph(f, d.text, left.x + 16, left.y + 52, left.w - 32, narrow ? 11.5 : 12.5, C.ink);
    g.text('Schema', left.x + 16, y + 18, { size: 11, color: C.muted, weight: 650 });
    paragraph(f, JEV.schemaText, left.x + 16, y + 38, left.w - 32, 11.5, C.amber, 4);
    // Right: token stream.
    g.panel(right.x, right.y, right.w, right.h, 'Sinh tự hồi quy: mỗi ô là một lượt forward', C.amber);
    const tokens = d.autoregressive.tokens, shown = Math.floor(seg(t, 2, 11) * tokens.length);
    const cell = { w: narrow ? 64 : 78, h: 34 }, cols = Math.max(1, Math.floor((right.w - 32) / (cell.w + 8)));
    tokens.slice(0, shown).forEach((tok, i) => {
      const x = right.x + 16 + (i % cols) * (cell.w + 8), yy = right.y + 54 + Math.floor(i / cols) * (cell.h + 10);
      const structural = /^[\s{}":,]*$/.test(tok);
      g.ctx.save(); g.ctx.fillStyle = alpha(structural ? C.white : C.amber, structural ? 0.06 : 0.16); g.ctx.strokeStyle = alpha(structural ? C.white : C.amber, 0.45);
      g.ctx.beginPath(); g.ctx.roundRect(x, yy, cell.w, cell.h, 7); g.ctx.fill(); g.ctx.stroke(); g.ctx.restore();
      g.text(JSON.stringify(tok).slice(1, -1).replace(/\\"/g, '"') || '␣', x + cell.w / 2, yy + cell.h / 2, { size: 12, color: structural ? C.muted : C.ink, align: 'center', mono: true });
      g.text(String(i + 1), x + 4, yy + 8, { size: 8.5, color: C.muted });
    });
    const rows = Math.ceil(tokens.length / cols), by = right.y + 54 + rows * (cell.h + 10) + 14;
    if (t > 11.5) {
      g.text('{' + d.autoregressive.text.slice(1), right.x + 16, by, { size: 12.5, color: C.ink, mono: true });
      g.tag(d.autoregressive.valid ? 'JSON hợp lệ' : 'JSON.parse lỗi', right.x + 16, by + 30, d.autoregressive.valid ? C.lime : C.red, 'left');
    }
  },
  metrics: () => { const a = doc().autoregressive; const n = Math.floor(seg(currentT, 2, 11) * a.tokens.length); return [{ label: 'Lượt forward', value: `${Math.max(n, currentT > 2 ? 1 : 0)} / ${a.forwardPasses}`, tone: 'amber' }, { label: 'Thời gian (CPU)', value: `${vn(a.seconds, 2)} s`, tone: 'violet' }, { label: 'JSON', value: a.valid ? 'hợp lệ' : 'không hợp lệ', tone: a.valid ? 'lime' : 'red' }]; },
  table: () => ({ headers: ['Tài liệu', 'Token sinh ra', 'Lượt forward', 'Thời gian', 'JSON hợp lệ', 'Kết quả'], rows: JEV.docs.map(d => [d.title, d.autoregressive.tokens.length, d.autoregressive.forwardPasses, `${vn(d.autoregressive.seconds, 2)} s`, d.autoregressive.valid ? 'có' : 'không', d.autoregressive.text]) }),
};

// ---------- Chapter 2: one prefill into the KV cache ----------
const chapter2: Scene = {
  title: 'Prefill một lần vào KV cache',
  duration: 11,
  beats: () => [
    { at: 0, text: `Ý tưởng của Jev: đưa ngữ cảnh + schema (${doc().parallel.promptTokens} token) qua Transformer đúng một lần.` },
    { at: 2, text: 'Ở giai đoạn prefill mọi token được xử lý song song: cả ma trận token đi qua từng lớp cùng lúc.' },
    { at: 5, text: `Mỗi lớp ghi key và value của mọi token vào KV cache: ${JEV.layers} lớp × ${JEV.kvHeads} đầu KV × ${JEV.headDim} chiều.` },
    { at: 8, text: `Cache này dùng lại được cho mọi trường của schema, không phải tính lại ngữ cảnh. Prefill đo được: ${vn(doc().parallel.prefillSeconds, 2)} s trên CPU.` },
  ],
  formula: () => `KV cache = 2 × ${JEV.layers} lớp × ${JEV.kvHeads} đầu × ${JEV.headDim} chiều × ${doc().parallel.promptTokens} token × 2 byte (fp16) = ${vn(kvBytes(doc().parallel.promptTokens) / 1e6, 2)} MB`,
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const d = doc(), n = d.parallel.promptTokens, L = JEV.layers;
    const area = f.safe;
    g.panel(area.x, area.y, area.w, area.h, `Prefill: ${n} token × ${L} lớp`, C.violet);
    const left = area.x + 20, top = area.y + 56, w = area.w - 40, h = area.h - 110;
    const colW = w / n, rowH = h / L;
    const pass = seg(t, 2, 7.5) * (L + 2);
    for (let l = 0; l < L; l++) {
      const done = clamp(pass - l), y = top + (L - 1 - l) * rowH;
      g.ctx.fillStyle = alpha(C.violet, 0.08); g.ctx.fillRect(left, y + 1, w, rowH - 2);
      if (done > 0) {
        g.ctx.fillStyle = alpha(l % 2 ? C.cyan : C.amber, 0.25 + 0.45 * done);
        g.ctx.fillRect(left, y + 1, w * done, rowH - 2);
      }
      if (!narrow && l % 3 === 0) g.text(`lớp ${l + 1}`, left - 4, y + rowH / 2, { size: 9, color: C.muted, align: 'right' });
    }
    // token strip
    for (let i = 0; i < n; i++) { g.ctx.fillStyle = alpha(C.white, 0.18 + 0.25 * ((i * 37) % 5) / 5); g.ctx.fillRect(left + i * colW, top + h + 10, Math.max(1, colW - 0.6), 12); }
    g.text(`${n} token ngữ cảnh + schema + “{” (xử lý song song)`, left, top + h + 38, { size: 11.5, color: C.muted });
    if (t > 5) g.tag('K, V của mọi lớp → KV cache', area.x + area.w - 20, top + h + 38, C.lime, 'right');
  },
  metrics: () => [{ label: 'Token ngữ cảnh', value: String(doc().parallel.promptTokens), tone: 'violet' }, { label: 'Prefill (CPU)', value: `${vn(doc().parallel.prefillSeconds, 2)} s`, tone: 'amber' }, { label: 'KV cache (fp16)', value: `${vn(kvBytes(doc().parallel.promptTokens) / 1e6, 2)} MB`, tone: 'lime' }],
};

// ---------- Chapter 3: scoring one field ----------
const chapter3: Scene = {
  title: 'Chấm điểm một trường',
  duration: 15,
  beats: () => {
    const f = doc().parallel.fields[FIELDS[state.field]];
    return [
      { at: 0, text: `Nối KV cache với hậu tố của trường: ${f.suffixTokens.map(tk).join(' ')}.` },
      { at: 2.5, text: `Token cuối cho ra hidden state ${JEV.hidden} chiều (ô lưới bên trái là vector thật, vàng dương, xanh âm).` },
      { at: 5, text: `LM head (một lớp tuyến tính chung) biến nó thành ${vn(JEV.vocab, 0)} logit, một cho mỗi token của từ vựng.` },
      { at: 8, text: 'Chỉ giữ các hàng ứng với giá trị hợp lệ. Giá trị nhiều token (M·EDIUM, T·IER·_·1) được cộng log-xác suất của mọi token.' },
      { at: 11, text: `Softmax trên các lựa chọn hợp lệ: ${f.candidates.map(c => `${c.value} ${vn(c.p * 100, 1)}%`).join(', ')}. Không thể ra giá trị ngoài schema.` },
    ];
  },
  formula: () => `h = Transformer(cache, hậu tố)[-1] ∈ ℝ^${JEV.hidden},  z = W_lm · h ∈ ℝ^${JEV.vocab}\nscore(c) = Σ log p(token của c),  p(c) = softmax(score) chỉ trên các c hợp lệ`,
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const field = FIELDS[state.field], data = doc().parallel.fields[field], color = FIELD_COLOR[state.field];
    const [a, b, c] = split(f.safe, narrow ? [1, 0.7, 1.3] : [1, 0.75, 1.35], narrow, 12);
    // a: hidden state grid
    g.panel(a.x, a.y, a.w, a.h, `Hidden state · ${JEV.hidden} chiều`, C.violet);
    const cols = 48, rows = Math.ceil(data.hidden.length / cols), cs = Math.min((a.w - 32) / cols, (a.h - 120) / rows);
    const sorted = data.hidden.map(Math.abs).sort((x, y) => x - y), max = sorted[Math.floor(sorted.length * 0.95)], reveal = seg(t, 2.5, 4.5);
    data.hidden.forEach((v, i) => {
      if (i / data.hidden.length > reveal) return;
      const k = Math.min(1, Math.abs(v) / max);
      g.ctx.fillStyle = v >= 0 ? `rgba(255,182,72,${k})` : `rgba(79,214,255,${k})`;
      g.ctx.fillRect(a.x + 16 + (i % cols) * cs, a.y + 50 + Math.floor(i / cols) * cs, cs - 0.3, cs - 0.3);
    });
    paragraph(f, `cache + ${data.suffixTokens.map(tk).join(' ')}`, a.x + 16, a.y + 60 + rows * cs, a.w - 32, 11.5, color, 2);
    // b: vocab strip with candidate rows
    g.panel(b.x, b.y, b.w, b.h, `${vn(JEV.vocab, 0)} logit`, C.amber);
    const sx = b.x + 18, sy = b.y + 50, sw = 26, sh = b.h - 90, show = seg(t, 5, 7);
    for (let i = 0; i < 120; i++) { g.ctx.fillStyle = alpha(C.white, (0.05 + 0.08 * ((i * 53) % 7) / 7) * show); g.ctx.fillRect(sx, sy + i * sh / 120, sw, sh / 120 - 0.5); }
    const pick = seg(t, 8, 10);
    data.firstTokenLogits.forEach((cand, k) => {
      const y = sy + cand.id / JEV.vocab * sh;
      g.ctx.fillStyle = alpha(color, show); g.ctx.fillRect(sx - 2, y - 2, sw + 4, 4);
      if (pick > 0) {
        const ly = sy + 20 + k * 30;
        g.line([[sx + sw + 2, y], [sx + sw + 24, ly]], alpha(color, 0.6 * pick), { width: 1 });
        g.text(`${tk(cand.token)} · ${vn(cand.logit, 1)}`, sx + sw + 28, ly, { size: 11, color: C.ink, mono: true, alpha: pick });
      }
    });
    g.text(`max ${vn(data.vocabLogitStats.max, 1)} · TB ${vn(data.vocabLogitStats.mean, 1)}`, sx, b.y + b.h - 22, { size: 10.5, color: C.muted });
    // c: candidate probabilities
    g.panel(c.x, c.y, c.w, c.h, `${FIELD_LABEL[field]}: softmax trên giá trị hợp lệ`, C.lime);
    const bar = seg(t, 11, 13);
    const bx = c.x + 18, bw = c.w - 120, by0 = c.y + 56, bh = Math.min(42, (c.h - 120) / data.candidates.length);
    data.candidates.forEach((cand, k) => {
      const y = by0 + k * (bh + 10), best = cand.value === data.answer;
      g.text(cand.value, bx, y + bh / 2 - 6, { size: 13, color: best ? C.lime : C.ink, weight: 650, mono: true });
      g.text(`${cand.tokens.map(tk).join('')}  log p = ${vn(cand.logprob, 2)}`, bx, y + bh / 2 + 10, { size: 10.5, color: C.muted, mono: true });
      const x0 = bx + Math.min(200, bw * 0.5), w = (c.x + c.w - 80 - x0) * cand.p * bar;
      g.ctx.save(); g.ctx.fillStyle = best ? C.lime : alpha(color, 0.75); if (best) { g.ctx.shadowColor = C.lime; g.ctx.shadowBlur = 12; }
      g.ctx.fillRect(x0, y + 4, Math.max(1, w), bh - 8); g.ctx.restore();
      g.text(`${vn(cand.p * 100, cand.p < 0.01 ? 2 : 1)}%`, x0 + w + 8, y + bh / 2, { size: 12, color: C.ink, weight: 650 });
    });
    const freeY = by0 + data.candidates.length * (bh + 10) + 16;
    if (t > 11 && freeY + 40 < c.y + c.h) {
      g.text('Nếu để mô hình tự do, token tiếp theo có thể là:', bx, freeY, { size: 11, color: C.muted });
      g.text(data.top.slice(0, 4).map(x => `${tk(x.token)} ${vn(x.p * 100, 1)}%`).join('   '), bx, freeY + 20, { size: 11, color: C.ink, mono: true });
    }
  },
  metrics: () => { const d = doc().parallel.fields[FIELDS[state.field]]; const best = d.candidates.find(c => c.value === d.answer)!; return [{ label: 'Trường', value: FIELDS[state.field], tone: 'violet' }, { label: 'Đáp án', value: d.answer, tone: 'lime' }, { label: 'Xác suất', value: `${vn(best.p * 100, 1)}%`, tone: 'amber' }]; },
  table: () => ({ headers: ['Tài liệu', 'Trường', 'Giá trị', 'Token', 'Σ log p', 'Xác suất'], rows: JEV.docs.flatMap(d => FIELDS.flatMap(fd => d.parallel.fields[fd].candidates.map(c => [d.title, fd, c.value, c.tokens.join(' · '), vn(c.logprob, 3), `${vn(c.p * 100, 2)}%`]))) }),
};

// ---------- Chapter 4: all fields in one pass, comparison ----------
const chapter4: Scene = {
  title: 'Mọi trường trong một lượt, so sánh',
  duration: 13,
  beats: () => {
    const d = doc(), same = FIELDS.every(fd => arValue(d, fd) === d.parallel.fields[fd].answer.toLowerCase());
    return [
      { at: 0, text: `Cache được nhân bản cho ${d.parallel.rows} hàng (mỗi trường × mỗi giá trị) và chấm điểm trong một lượt forward theo batch.` },
      { at: 3, text: 'JSON được ghép bằng code từ các giá trị đã chọn: luôn đúng cú pháp, luôn nằm trong schema.' },
      { at: 6, text: `Tổng cộng 2 lượt forward thay vì ${d.autoregressive.forwardPasses}: ${vn(d.parallel.seconds, 2)} s so với ${vn(d.autoregressive.seconds, 2)} s trên CPU (×${vn(d.autoregressive.seconds / d.parallel.seconds, 1)}).` },
      { at: 9.5, text: same ? 'Ở tài liệu này hai cách cho cùng đáp án.' : 'Lưu ý: ở tài liệu này action_tier khác nhau. Cách song song chấm từng trường độc lập, không thấy các giá trị đã chọn của trường khác.' },
    ];
  },
  formula: () => { const d = doc(); return `song song: prefill ${vn(d.parallel.prefillSeconds, 2)} s + 1 lượt batch ${d.parallel.rows} hàng ${vn(d.parallel.scoreSeconds, 2)} s = ${vn(d.parallel.seconds, 2)} s\ntự hồi quy: ${d.autoregressive.forwardPasses} lượt = ${vn(d.autoregressive.seconds, 2)} s`; },
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const d = doc();
    const [top, bottom] = split(f.safe, narrow ? [1.3, 1] : [1.2, 1], true, 12);
    g.panel(top.x, top.y, top.w, top.h, 'Prefill một lần → chấm mọi trường cùng lúc', C.violet);
    const bw = narrow ? 92 : 130, bh = 50;
    const px = top.x + 20, py = top.y + top.h / 2 - bh / 2;
    box(f, px, py, bw, bh, 'Prefill', `${d.parallel.promptTokens} token`, C.violet, 8);
    box(f, px + bw + 26, py, bw, bh, 'KV cache', `×${d.parallel.rows} hàng`, C.cyan, 8);
    g.arrow(px + bw + 2, py + bh / 2, px + bw + 24, py + bh / 2, C.muted, { width: 1.5 });
    const fx = px + 2 * bw + 70, fw = narrow ? 120 : 170, gap = (top.h - 70 - 3 * bh) / 2;
    FIELDS.forEach((fd, k) => {
      const y = top.y + 46 + k * (bh + gap), on = seg(t, 0.5 + k * 0.4, 1.5 + k * 0.4);
      g.ctx.globalAlpha = on;
      g.arrow(px + 2 * bw + 28, py + bh / 2, fx - 4, y + bh / 2, alpha(FIELD_COLOR[k], 0.7), { width: 1.4 });
      box(f, fx, y, fw, bh, fd, `${JEV.schema[fd].length} lựa chọn`, FIELD_COLOR[k]);
      const ans = d.parallel.fields[fd], best = ans.candidates.find(c => c.value === ans.answer)!;
      g.arrow(fx + fw + 4, y + bh / 2, fx + fw + 30, y + bh / 2, alpha(FIELD_COLOR[k], 0.7), { width: 1.4 });
      box(f, fx + fw + 34, y, narrow ? 96 : 130, bh, ans.answer, `p = ${vn(best.p, 2)}`, C.amber);
      g.ctx.globalAlpha = 1;
    });
    if (t > 3) { g.text(parallelJson(d), top.x + top.w - 20, top.y + top.h - 18, { size: narrow ? 10.5 : 12, color: C.lime, align: 'right', mono: true }); }
    // bottom: comparison bars
    g.panel(bottom.x, bottom.y, bottom.w, bottom.h, 'So sánh trên cùng máy (CPU, float32)', C.amber);
    const rows: [string, number, number, string, string][] = [
      ['Lượt forward', d.autoregressive.forwardPasses, d.parallel.forwardPasses, String(d.autoregressive.forwardPasses), '2'],
      ['Thời gian', d.autoregressive.seconds, d.parallel.seconds, `${vn(d.autoregressive.seconds, 2)} s`, `${vn(d.parallel.seconds, 2)} s`],
    ];
    const reveal = seg(t, 6, 8), lx = bottom.x + 18, x0 = bottom.x + (narrow ? 110 : 150), maxW = bottom.w - (x0 - bottom.x) - 90;
    rows.forEach(([label, a, b, sa, sb], k) => {
      const y = bottom.y + 50 + k * 60, m = Math.max(a, b);
      g.text(label, lx, y + 12, { size: 12, color: C.ink, weight: 650 });
      g.ctx.fillStyle = alpha(C.red, 0.8); g.ctx.fillRect(x0, y, maxW * a / m * reveal, 16);
      g.text(`tự hồi quy ${sa}`, x0 + maxW * a / m * reveal + 6, y + 8, { size: 11, color: C.ink });
      g.ctx.fillStyle = alpha(C.lime, 0.85); g.ctx.fillRect(x0, y + 22, maxW * b / m * reveal, 16);
      g.text(`song song ${sb}`, x0 + maxW * b / m * reveal + 6, y + 30, { size: 11, color: C.ink });
    });
    const yy = bottom.y + 50 + 2 * 60;
    if (yy + 20 < bottom.y + bottom.h) g.text(`JSON hợp lệ: tự hồi quy ${d.autoregressive.valid ? 'có' : 'không'} · song song luôn có`, lx, yy + 6, { size: 12, color: d.autoregressive.valid ? C.ink : C.red });
  },
  metrics: () => { const d = doc(); return [{ label: 'Lượt forward', value: `${d.parallel.forwardPasses} thay vì ${d.autoregressive.forwardPasses}`, tone: 'lime' }, { label: 'Nhanh hơn (CPU)', value: `×${vn(d.autoregressive.seconds / d.parallel.seconds, 1)}`, tone: 'amber' }, { label: 'JSON hợp lệ', value: '100%', tone: 'lime' }]; },
  table: () => ({ headers: ['Tài liệu', 'Tự hồi quy', 'Song song (Jev)', 'Thời gian TH / SS'], rows: JEV.docs.map(d => [d.title, d.autoregressive.text, parallelJson(d), `${vn(d.autoregressive.seconds, 2)} s / ${vn(d.parallel.seconds, 2)} s`]) }),
};

export const initJevLab = () => glassLab('jev', [chapter1, chapter2, chapter3, chapter4], lab => {
  lab.on('[data-doc]', 'click', el => { state.doc = Number(el.dataset.doc); document.querySelectorAll<HTMLElement>('[data-doc]').forEach(b => b.setAttribute('aria-pressed', String(b === el))); lab.invalidate(); });
  lab.on('[data-field]', 'click', el => { state.field = Number(el.dataset.field); document.querySelectorAll<HTMLElement>('[data-field]').forEach(b => b.setAttribute('aria-pressed', String(b === el))); lab.replay(); });
});

