// Naive Bayes lab on the SMS Spam Collection: count words in 4 457 training messages, map every word by its two
// likelihoods, add up the evidence of a message token by token, then check the real test-set results and calibration.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, Plot, alpha, clamp, ease, seg, split, vn, type Rect } from '../../_shared/glass-lab/gfx';
import { CALIBRATION, EXAMPLES, EXTREME, METRICS, PRIOR_SPAM, TEST, TOTAL, TRAIN, V, VOCAB, evidence, likelihood, logRatio, strongest } from './model';

export const explainerKind = 'educational-simulation';

const ALPHAS = METRICS.map(m => m.alpha);
const state = { alphaIndex: 4, text: EXAMPLES[0].text, focus: 'claim', hover: '' };
const alphaNow = () => ALPHAS[state.alphaIndex];
const nf = (v: number) => Math.round(v).toLocaleString('vi-VN');
let currentT = 0;

/** Probabilities that underflow a fixed format are written in scientific notation. */
function prob(p: number) {
  if (p >= 0.001 && p <= 0.999) return `${vn(p * 100, 2)}%`;
  if (p > 0.999) return 1 - p < 1e-15 ? '> 1 − 10⁻¹⁵' : `1 − ${sci(1 - p)}`;
  return p < 1e-300 ? '< 10⁻³⁰⁰' : sci(p);
}
const sup = (n: number) => String(n).replace('-', '⁻').replace(/\d/g, d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(d)]);
function sci(p: number) {
  const e = Math.floor(Math.log10(p)), m = p / 10 ** e;
  return `${vn(m, 1)} × 10${sup(e)}`;
}

// ---------- Chapter 1: counting ----------
const SPAM_WORDS = strongest(1, 'spam', 9), HAM_WORDS = strongest(1, 'ham', 9);

const chapter1: Scene = {
  title: 'Huấn luyện là đếm',
  duration: 11,
  beats: () => [
    { at: 0, text: `4 457 tin nhắn huấn luyện đã gán nhãn: ${nf(TRAIN.spam)} spam và ${nf(TRAIN.ham)} tin thường. Prior P(spam) = ${vn(PRIOR_SPAM * 100, 1)}%.` },
    { at: 3, text: `Đếm số lần mỗi từ xuất hiện trong từng lớp: ${nf(TOTAL.spam)} từ trong spam, ${nf(TOTAL.ham)} từ trong tin thường, ${nf(V)} từ khác nhau.` },
    { at: 6, text: '“claim” xuất hiện 94 lần trong spam và chưa lần nào trong tin thường; “lor”, “later” thì ngược lại.' },
    { at: 8.5, text: 'Toàn bộ “mô hình” chỉ là bảng đếm này. Không có vòng lặp tối ưu nào.' },
  ],
  formula: 'P(spam) = 598 / 4 457\nP(từ | lớp) ≈ số lần từ xuất hiện trong lớp / tổng số từ của lớp',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 2.4] : [0.8, 1.7], narrow);
    g.panel(left.x, left.y, left.w, left.h, 'Prior: tỉ lệ hai lớp', C.magenta);
    const grow = seg(t, 0.3, 2.5);
    const bar = { x: left.x + 24, y: left.y + 46, w: left.w - 48, h: Math.min(46, left.h * 0.16) };
    const spamW = bar.w * PRIOR_SPAM * grow;
    const { ctx } = g;
    ctx.save(); ctx.beginPath(); ctx.roundRect(bar.x, bar.y, bar.w, bar.h, 12); ctx.clip();
    ctx.fillStyle = alpha(C.cyan, 0.22); ctx.fillRect(bar.x, bar.y, bar.w * grow, bar.h);
    ctx.fillStyle = C.magenta; ctx.shadowColor = C.magenta; ctx.shadowBlur = 16; ctx.fillRect(bar.x, bar.y, spamW, bar.h);
    ctx.restore();
    g.text(`spam ${nf(TRAIN.spam * grow)}`, bar.x, bar.y + bar.h + 18, { size: 12.5, color: C.magenta, weight: 650 });
    g.text(`thường ${nf(TRAIN.ham * grow)}`, bar.x + bar.w, bar.y + bar.h + 18, { size: 12.5, color: C.cyan, weight: 650, align: 'right' });
    // Big number under the bar; on short panels it shrinks so it never leaves the card.
    const room = left.y + left.h - (bar.y + bar.h + 30), big = clamp(room * 0.42, 18, 40);
    g.text(vn(PRIOR_SPAM * 100 * grow, 1) + '%', left.x + left.w / 2, bar.y + bar.h + 30 + big * 0.7, { size: big, weight: 700, color: C.ink, align: 'center', mono: true });
    if (room > 90) g.text('P(spam) trước khi đọc nội dung', left.x + left.w / 2, bar.y + bar.h + 30 + big * 1.5, { size: 12, color: C.muted, align: 'center' });
    if (left.h > 300) g.text(`kiểm tra: ${nf(TEST.spam)} spam · ${nf(TEST.ham)} thường (giữ riêng)`, left.x + left.w / 2, left.y + left.h - 22, { size: 11.5, color: C.muted, align: 'center' });

    g.panel(right.x, right.y, right.w, right.h, 'Từ phân biệt rõ nhất (số lần xuất hiện: spam / thường)', C.amber);
    const columns = split({ x: right.x + 16, y: right.y + 44, w: right.w - 32, h: right.h - 56 }, [1, 1], false, 24);
    [[SPAM_WORDS, C.magenta, 'nghiêng hẳn về spam'], [HAM_WORDS, C.cyan, 'nghiêng hẳn về tin thường']].forEach(([words, color, label], c) => {
      const col = columns[c], list = words as typeof SPAM_WORDS, row = Math.min(30, (col.h - 24) / list.length);
      g.text(label as string, col.x, col.y + 6, { size: 11.5, color: color as string, weight: 650 });
      const max = Math.max(...list.map(w => Math.max(w.spam, w.ham)));
      list.forEach((w, i) => {
        const p = seg(t, 3 + i * 0.25 + c * 0.15, 4.2 + i * 0.25 + c * 0.15), y = col.y + 28 + i * row;
        if (p <= 0) return;
        g.text(w.word, col.x, y, { size: 13, weight: 650, color: C.ink, alpha: p, mono: true });
        const bx = col.x + 92, bw = col.w - 160;
        if (bw > 40) {
        g.ctx.save(); g.ctx.globalAlpha = p;
        g.ctx.fillStyle = C.magenta; g.ctx.fillRect(bx, y - 7, bw * w.spam / max * p, 6);
        g.ctx.fillStyle = C.cyan; g.ctx.fillRect(bx, y + 1, bw * w.ham / max * p, 6);
        g.ctx.restore();
        }
        g.text(`${w.spam} / ${w.ham}`, col.x + col.w, y, { size: 12, color: C.muted, align: 'right', mono: true, alpha: p });
      });
    });
  },
  metrics: () => [{ label: 'Tin huấn luyện', value: nf(TRAIN.spam + TRAIN.ham), tone: 'violet' }, { label: 'Từ vựng', value: nf(V), tone: 'amber' }, { label: 'P(spam)', value: `${vn(PRIOR_SPAM * 100, 1)}%`, tone: 'magenta' }],
  table: () => ({ headers: ['Từ', 'Spam', 'Thường', 'log P(w|spam) − log P(w|thường)'], rows: [...SPAM_WORDS, ...HAM_WORDS].map(w => [w.word, w.spam, w.ham, vn(w.ratio, 2)]) }),
};

// ---------- Chapter 2: the likelihood map ----------
const LABELLED = ['claim', 'prize', 'free', 'txt', 'call', 'win', 'ringtone', 'ok', 'lor', 'love', 'home', 'later', 'you', 'the', 'i', 'sorry'];
let wordPlot: Plot | null = null;

const chapter2: Scene = {
  title: 'Likelihood và làm trơn',
  duration: 11,
  beats: () => [
    { at: 0, text: `Mỗi chấm là một trong ${nf(V)} từ. Trục ngang: P(từ | thường), trục dọc: P(từ | spam), thang log.` },
    { at: 3, text: 'Trên đường chéo, từ xuất hiện ở hai lớp như nhau và không giúp phân loại. Càng xa đường chéo, từ càng có sức nặng.' },
    { at: 6, text: 'Từ chưa từng gặp trong một lớp vẫn được cộng α vào số đếm. Không có α, xác suất đó bằng 0 và một từ duy nhất quyết định toàn bộ kết luận.' },
    { at: 9, text: 'Kéo α về 0,01: các từ có số đếm 0 bị đẩy ra rìa, cách đường chéo hàng triệu lần.' },
  ],
  formula: () => { const [s, h] = likelihood('claim', alphaNow()); return `P(claim | spam) = (94 + α) / (${nf(TOTAL.spam)} + α·${nf(V)}) = ${prob(s)}\nP(claim | thường) = (0 + α) / (${nf(TOTAL.ham)} + α·${nf(V)}) = ${prob(h)},  α = ${vn(alphaNow(), 2)}`; },
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.6, 1] : [1.7, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, 'Bản đồ từ: xác suất trong tin thường và trong spam', C.violet);
    const a = alphaNow();
    // The smallest probability is α / (total + α|V|): the axes follow it so zero-count words sit at the edge.
    const low = Math.floor((Math.log10(a / (TOTAL.ham + a * V)) - 0.4) * 2) / 2;
    const plot = new Plot(g, left, [low, -1], [low, -1], { l: 48, r: 16, t: 40, b: 36 });
    wordPlot = plot;
    const lbl = (v: number) => `10${sup(v)}`;
    const marks = Array.from({ length: Math.floor(-1 - low) + 1 }, (_, i) => -1 - i).filter(v => v >= low);
    plot.axes({ xTicks: marks, yTicks: marks, xLabel: 'P(từ | thường)', yLabel: 'P(từ | spam)', fx: lbl, fy: lbl });
    plot.clip(() => {
      g.polygon([plot.p([low, low]), plot.p([-1, -1]), plot.p([low, -1])], alpha(C.magenta, 0.06));
      g.polygon([plot.p([low, low]), plot.p([-1, -1]), plot.p([-1, low])], alpha(C.cyan, 0.05));
      plot.line([[low, low], [-1, -1]], alpha(C.white, 0.4), { width: 1.2, dash: [5, 5] });
      const reveal = seg(t, 0.2, 3);
      VOCAB.forEach(([word], i) => {
        if (i / VOCAB.length > reveal) return;
        const [s, h] = likelihood(word, a), r = Math.log(s / h);
        const color = r > 0 ? C.magenta : C.cyan;
        g.circle(plot.px(Math.log10(h)), plot.py(Math.log10(s)), state.hover === word || state.focus === word ? 5 : 2, color, { alpha: clamp(0.25 + Math.abs(r) / 6, 0.25, 0.9) });
      });
      if (t > 3) for (const word of LABELLED) {
        const [s, h] = likelihood(word, a), x = plot.px(Math.log10(h)), y = plot.py(Math.log10(s));
        g.circle(x, y, 4, C.white, { glow: 10, alpha: seg(t, 3, 4) });
        g.text(word, x + 7, y - 7, { size: 12, weight: 650, color: Math.log(s / h) > 0 ? C.magenta : C.cyan, alpha: seg(t, 3, 4) });
      }
    });
    g.text('vùng spam', plot.px(low + 0.3), plot.py(-1.5), { size: 12, color: C.magenta, weight: 650 });
    g.text('vùng tin thường', plot.px(-1.3), plot.py(low + 0.4), { size: 12, color: C.cyan, weight: 650, align: 'right' });
    detail(f, right, state.hover || state.focus);
  },
  metrics: () => [{ label: 'α (làm trơn)', value: vn(alphaNow(), 2), tone: 'amber' }, { label: 'log tỉ số của “claim”', value: vn(logRatio('claim', alphaNow()), 2), tone: 'magenta' }, { label: 'log tỉ số của “lor”', value: vn(logRatio('lor', alphaNow()), 2), tone: 'cyan' }],
  pointer(e) {
    if (!wordPlot || (e.type !== 'move' && e.type !== 'click')) { if (e.type === 'leave' && state.hover) { state.hover = ''; return true; } return false; }
    let best = '', distance = 10;
    for (const [word] of VOCAB) {
      const [s, h] = likelihood(word, alphaNow()), d = Math.hypot(wordPlot.px(Math.log10(h)) - e.x, wordPlot.py(Math.log10(s)) - e.y);
      if (d < distance) { distance = d; best = word; }
    }
    if (e.type === 'click' && best) { state.focus = best; return true; }
    if (best !== state.hover) { state.hover = best; return true; }
    return false;
  },
  table: () => ({ headers: ['Từ', 'Spam', 'Thường', 'P(w|spam)', 'P(w|thường)', 'log tỉ số'], rows: LABELLED.map(w => { const i = VOCAB.findIndex(v => v[0] === w); const [s, h] = likelihood(w, alphaNow()); return [w, VOCAB[i]?.[1] ?? 0, VOCAB[i]?.[2] ?? 0, prob(s), prob(h), vn(Math.log(s / h), 2)]; }) }),
};

function detail(f: Frame, rect: Rect, word: string) {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, `Chi tiết: “${word}”`, C.amber);
  const row = VOCAB.find(v => v[0] === word);
  if (!row) return;
  const [s, h] = likelihood(word, alphaNow()), r = Math.log(s / h);
  const x = rect.x + 20, lines: [string, string, string][] = [
    ['trong spam', `${nf(row[1])} lần`, C.magenta], ['trong tin thường', `${nf(row[2])} lần`, C.cyan],
    ['P(từ | spam)', prob(s), C.magenta], ['P(từ | thường)', prob(h), C.cyan], ['mỗi lần xuất hiện cộng', `${r >= 0 ? '+' : ''}${vn(r, 2)} vào log-odds`, r >= 0 ? C.magenta : C.cyan],
  ];
  const step = Math.min(42, (rect.h - 110) / lines.length);
  lines.forEach(([label, value, color], i) => {
    g.text(label, x, rect.y + 56 + i * step, { size: 11.5, color: C.muted });
    g.text(value, x, rect.y + 56 + i * step + 17, { size: 15, weight: 650, color, mono: true });
  });
  g.text(`= nhân P(spam)/P(thường) với ${Math.exp(r) >= 1 ? vn(Math.exp(r), Math.exp(r) > 100 ? 0 : 1) : '1/' + vn(1 / Math.exp(r), Math.exp(-r) > 100 ? 0 : 1)}`, x, rect.y + rect.h - 22, { size: 12, color: C.ink });
}

// ---------- Chapter 3: classify a message ----------
const chapter3: Scene = {
  title: 'Cộng bằng chứng từng từ',
  duration: 12,
  beats: () => {
    const r = evidence(state.text, alphaNow());
    return [
      { at: 0, text: `Bắt đầu từ prior: log-odds = log(${vn(PRIOR_SPAM, 3)} / ${vn(1 - PRIOR_SPAM, 3)}) = ${vn(r.prior, 2)}.` },
      { at: 2, text: 'Mỗi từ cộng thêm log P(từ | spam) − log P(từ | thường): cột hồng đẩy lên phía spam, cột xanh kéo xuống phía tin thường.' },
      { at: 7, text: 'Từ không có trong bộ từ vựng huấn luyện bị bỏ qua (cột xám). Thứ tự từ không quan trọng: chỉ có tổng.' },
      { at: 10, text: `Tổng log-odds ${vn(r.logOdds, 2)} → P(spam) = ${prob(r.pSpam)}. ${r.pSpam >= 0.5 ? 'Kết luận: spam.' : 'Kết luận: tin thường.'}` },
    ];
  },
  formula: 'log P(spam|tin)/P(thường|tin) = log P(spam)/P(thường) + Σ_từ [log P(từ|spam) − log P(từ|thường)]',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.5, 1] : [1.9, 1], narrow);
    const r = evidence(state.text, alphaNow());
    const shown = Math.floor(seg(t, 1.5, 9.5) * r.steps.length + 1e-9);
    g.panel(left.x, left.y, left.w, left.h, 'Biểu đồ thác nước: log-odds sau mỗi từ', C.violet);
    const values = [r.prior, ...r.steps.map(s => s.total)];
    const lo = Math.min(-6, ...values) - 1, hi = Math.max(6, ...values) + 1;
    const plot = new Plot(g, left, [0, r.steps.length + 1], [lo, hi], { l: 46, r: 14, t: 40, b: 58 });
    plot.axes({ yTicks: [Math.ceil(lo / 10) * 10, 0, Math.floor(hi / 10) * 10].filter((v, i, a) => a.indexOf(v) === i), fy: v => vn(v, 0), yLabel: 'log-odds (dương = spam)' });
    plot.line([[0, 0], [r.steps.length + 1, 0]], alpha(C.white, 0.5), { width: 1.2 });
    g.text('↑ spam', plot.box.x + plot.box.w - 6, plot.py(0) - 12, { size: 11, color: C.magenta, align: 'right' });
    g.text('↓ thường', plot.box.x + plot.box.w - 6, plot.py(0) + 12, { size: 11, color: C.cyan, align: 'right' });
    const width = Math.min(34, plot.box.w / (r.steps.length + 1) * 0.7);
    const column = (i: number, from: number, to: number, color: string, label: string, appear: number) => {
      const x = plot.px(i + 0.5), y0 = plot.py(from), y1 = plot.py(from + (to - from) * ease(appear));
      g.ctx.save(); g.ctx.shadowColor = color; g.ctx.shadowBlur = 12; g.ctx.fillStyle = color;
      g.ctx.beginPath(); g.ctx.roundRect(x - width / 2, Math.min(y0, y1), width, Math.max(2, Math.abs(y1 - y0)), 4); g.ctx.fill(); g.ctx.restore();
      g.ctx.save(); g.ctx.translate(x, plot.box.y + plot.box.h + 8); g.ctx.rotate(-Math.PI / 4);
      g.text(label, 0, 0, { size: 11.5, color: C.ink, align: 'right', mono: true, alpha: appear }); g.ctx.restore();
    };
    column(0, 0, r.prior, C.violet, 'prior', seg(t, 0.2, 1.2));
    r.steps.forEach((s, i) => {
      if (i >= shown) return;
      const from = i === 0 ? r.prior : r.steps[i - 1].total;
      const p = clamp((seg(t, 1.5, 9.5) * r.steps.length - i) * 1.5);
      column(i + 1, from, s.total, s.known ? (s.delta >= 0 ? C.magenta : C.cyan) : alpha(C.white, 0.25), s.word, p);
      plot.line([[i + 0.5, from], [i + 1.5, from]], alpha(C.white, 0.25), { width: 1, dash: [2, 3] });
    });
    gauge(f, right, shown >= r.steps.length && t > 9.5 ? r.pSpam : 1 / (1 + Math.exp(-(shown ? r.steps[shown - 1].total : r.prior))));
  },
  metrics() {
    const r = evidence(state.text, alphaNow());
    return [{ label: 'Từ dùng được', value: `${r.steps.filter(s => s.known).length} / ${r.steps.length}`, tone: 'violet' }, { label: 'Tổng log-odds', value: vn(r.logOdds, 2), tone: r.logOdds >= 0 ? 'magenta' : 'cyan' }, { label: 'P(spam)', value: prob(r.pSpam), tone: r.pSpam >= 0.5 ? 'magenta' : 'cyan' }];
  },
  table: () => { const r = evidence(state.text, alphaNow()); return { headers: ['Từ', 'Có trong từ vựng', 'Đóng góp', 'Log-odds tích lũy'], rows: [['prior', 'có', vn(r.prior, 3), vn(r.prior, 3)], ...r.steps.map(s => [s.word, s.known ? 'có' : 'không', vn(s.delta, 3), vn(s.total, 3)])] }; },
};

/** Half-ring gauge for P(spam). */
function gauge(f: Frame, rect: Rect, p: number) {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, 'Xác suất hậu nghiệm', C.magenta);
  const cx = rect.x + rect.w / 2, cy = rect.y + rect.h * 0.62, radius = Math.min(rect.w * 0.38, rect.h * 0.42);
  const { ctx } = g;
  ctx.save(); ctx.lineCap = 'round'; ctx.lineWidth = 16;
  ctx.strokeStyle = alpha(C.cyan, 0.25); ctx.beginPath(); ctx.arc(cx, cy, radius, Math.PI, 2 * Math.PI); ctx.stroke();
  ctx.strokeStyle = p >= 0.5 ? C.magenta : C.cyan; ctx.shadowColor = ctx.strokeStyle as string; ctx.shadowBlur = 18;
  ctx.beginPath(); ctx.arc(cx, cy, radius, Math.PI, Math.PI + Math.PI * clamp(p)); ctx.stroke();
  ctx.restore();
  g.text(prob(p), cx, cy - radius * 0.28, { size: Math.min(30, rect.w / 9), weight: 700, align: 'center', mono: true, color: p >= 0.5 ? C.magenta : C.cyan });
  g.text('P(spam | tin nhắn)', cx, cy + 4, { size: 12, color: C.muted, align: 'center' });
  g.text('thường', cx - radius, cy + 26, { size: 11.5, color: C.cyan, align: 'center' });
  g.text('spam', cx + radius, cy + 26, { size: 11.5, color: C.magenta, align: 'center' });
}

// ---------- Chapter 4: real results ----------
const chapter4: Scene = {
  title: 'Kết quả thật trên 1 115 tin',
  duration: 11,
  beats: () => {
    const m = METRICS[state.alphaIndex];
    return [
      { at: 0, text: `Chạy mô hình trên 1 115 tin chưa từng thấy khi huấn luyện (α = ${vn(m.alpha, 2)}).` },
      { at: 2.5, text: `Đúng ${vn(m.accuracy * 100, 1)}%: bắt được ${m.tp}/${m.tp + m.fn} tin spam, chỉ chặn nhầm ${m.fp}/${m.fp + m.tn} tin thường.` },
      { at: 6, text: `Nhưng xác suất thì quá tự tin: ${vn(EXTREME * 100, 0)}% số tin nhận P(spam) dưới 0,1% hoặc trên 99,9%.` },
      { at: 8.5, text: `Có tin spam bị chấm P(spam) < 10⁻⁶. Giả định các từ độc lập cộng dồn bằng chứng trùng lặp, đẩy xác suất ra hai cực.` },
    ];
  },
  formula: 'precision = TP / (TP + FP)    recall = TP / (TP + FN)',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1, 1.3], narrow);
    const m = METRICS[state.alphaIndex], p = seg(t, 0.5, 2.5);
    g.panel(left.x, left.y, left.w, left.h, `Ma trận nhầm lẫn · α = ${vn(m.alpha, 2)}`, C.cyan);
    const size = Math.min((left.w - 120) / 2, (left.h - 150) / 2), x0 = left.x + 90, y0 = left.y + 70;
    const cells: [number, string, string][] = [[m.tn, 'thường → thường', C.cyan], [m.fp, 'thường → spam', C.red], [m.fn, 'spam → thường', C.amber], [m.tp, 'spam → spam', C.magenta]];
    cells.forEach(([value, label, color], i) => {
      const cx = x0 + (i % 2) * (size + 8), cy = y0 + Math.floor(i / 2) * (size + 8);
      const strength = i === 0 || i === 3 ? 0.28 : 0.12 + Math.min(0.3, value / 60);
      g.ctx.save(); g.ctx.fillStyle = alpha(color, strength * p); g.ctx.strokeStyle = alpha(color, 0.7); g.ctx.beginPath(); g.ctx.roundRect(cx, cy, size, size, 12); g.ctx.fill(); g.ctx.stroke(); g.ctx.restore();
      g.text(nf(value * p), cx + size / 2, cy + size / 2 - 6, { size: Math.min(34, size / 3), weight: 700, align: 'center', mono: true, color });
      g.text(label, cx + size / 2, cy + size / 2 + size / 5, { size: 11, align: 'center', color: C.muted });
    });
    g.text('thật: thường', x0 - 10, y0 + size / 2, { size: 11, color: C.muted, align: 'right' });
    g.text('thật: spam', x0 - 10, y0 + size * 1.5 + 8, { size: 11, color: C.muted, align: 'right' });
    g.text('dự đoán: thường', x0 + size / 2, y0 - 14, { size: 11, color: C.muted, align: 'center' });
    g.text('dự đoán: spam', x0 + size * 1.5 + 8, y0 - 14, { size: 11, color: C.muted, align: 'center' });
    const stats: [string, number, string][] = [['độ chính xác', m.accuracy, C.lime], ['precision', m.precision, C.magenta], ['recall', m.recall, C.amber]];
    stats.forEach(([label, value, color], i) => g.text(`${label} ${vn(value * 100, 1)}%`, left.x + 20 + i * (left.w - 40) / 3, left.y + left.h - 24, { size: 12.5, weight: 650, color }));
    calibration(f, right, seg(t, 5.5, 8));
  },
  metrics: () => { const m = METRICS[state.alphaIndex]; return [{ label: 'Độ chính xác', value: `${vn(m.accuracy * 100, 1)}%`, tone: 'lime' }, { label: 'Precision', value: `${vn(m.precision * 100, 1)}%`, tone: 'magenta' }, { label: 'Recall', value: `${vn(m.recall * 100, 1)}%`, tone: 'amber' }]; },
  table: () => ({ headers: ['α', 'Đúng', 'Precision', 'Recall', 'TP', 'FP', 'FN', 'TN'], rows: METRICS.map(m => [vn(m.alpha, 2), `${vn(m.accuracy * 100, 2)}%`, `${vn(m.precision * 100, 2)}%`, `${vn(m.recall * 100, 2)}%`, m.tp, m.fp, m.fn, m.tn]), caption: 'Tập kiểm tra 1 115 tin (149 spam, 966 thường), chia 80/20 phân tầng với hạt giống 42.' }),
};

/** Histogram of P(spam) on the test set (α = 1), stacked by true class, with uneven bins that expose the extremes. */
function calibration(f: Frame, rect: Rect, p: number) {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, 'P(spam) mà mô hình gán cho 1 115 tin kiểm tra (α = 1)', C.amber);
  const box = { x: rect.x + 48, y: rect.y + 76, w: rect.w - 64, h: rect.h - 136 };
  const max = Math.max(...CALIBRATION.map(b => b.spam + b.ham));
  const labels = ['< 10⁻⁶', '10⁻⁶–0,1%', '0,1–10%', '10–50%', '50–90%', '90–99,9%', '99,9%–1−10⁻⁶', '> 1−10⁻⁶'];
  const width = box.w / CALIBRATION.length;
  CALIBRATION.forEach((bin, i) => {
    const x = box.x + i * width + width * 0.15, w = width * 0.7;
    const hh = box.h * bin.ham / max * p, hs = box.h * bin.spam / max * p;
    const extreme = i === 0 || i === CALIBRATION.length - 1;
    g.ctx.save(); g.ctx.fillStyle = C.cyan; if (extreme) { g.ctx.shadowColor = C.cyan; g.ctx.shadowBlur = 12; }
    g.ctx.fillRect(x, box.y + box.h - hh, w, hh); g.ctx.fillStyle = C.magenta; g.ctx.fillRect(x, box.y + box.h - hh - hs, w, hs); g.ctx.restore();
    if (bin.spam + bin.ham) g.text(`${bin.ham ? bin.ham : ''}${bin.ham && bin.spam ? ' + ' : ''}${bin.spam ? bin.spam : ''}`, x + w / 2, box.y + box.h - hh - hs - 10, { size: 10.5, align: 'center', color: C.ink, mono: true });
    g.ctx.save(); g.ctx.translate(x + w / 2, box.y + box.h + 10); g.ctx.rotate(-Math.PI / 6);
    g.text(labels[i], 0, 0, { size: 10.5, color: extreme ? C.amber : C.muted, align: 'right' }); g.ctx.restore();
  });
  g.line([[box.x, box.y + box.h], [box.x + box.w, box.y + box.h]], C.faint, { width: 1 });
  g.text('xanh: tin thường thật · hồng: spam thật', rect.x + rect.w - 16, rect.y + 17, { size: 11, color: C.muted, align: 'right' });
  const wrong = CALIBRATION[0].spam;
  if (p > 0.6) g.tag(`${wrong} tin spam bị chấm P(spam) < 10⁻⁶`, box.x + width * 1.2, box.y + box.h * 0.32, C.amber, 'left');
}

export const initNaiveBayesLab = () => glassLab('naive-bayes', [chapter1, chapter2, chapter3, chapter4], lab => {
  lab.range('alpha', value => vn(ALPHAS[value], 2), value => { state.alphaIndex = value; });
  const box = lab.root.querySelector<HTMLTextAreaElement>('[data-message]')!;
  box.value = state.text;
  box.addEventListener('input', () => { state.text = box.value; lab.replay(); });
  lab.on('[data-example]', 'click', element => { state.text = EXAMPLES[Number(element.dataset.example)].text; box.value = state.text; lab.replay(); });
});
