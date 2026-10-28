// Hypothesis-testing lab: H0/H1 on Student's sleep data, the exact randomisation distribution, the t distribution with
// one- and two-sided rejection regions and the p-value area, the common tests on real data, type I/II errors and power
// by simulation, and the classic misreadings (p-hacking, significance vs effect size).
import { glassLab, type Scene } from '../../_shared/glass-lab/engine';
import { C, Plot, alpha, seg, split, vn, type Point } from '../../_shared/glass-lab/gfx';
import { barChart, categoryChart } from '../../_shared/glass-lab/views';
import { DIFF, OBS, S, criticalT, mean, normPdf, pairedT, power, signFlipMeans, simulateP, tPdf, tSf, variance } from './model';

export const explainerKind = 'educational-simulation';

const state = { alpha: 0.05, side: 'two' as 'two' | 'right' | 'left', n: 20, d: 0.5, test: null as number | null };
let currentT = 0;
const PT = pairedT(DIFF);
const FLIPS = signFlipMeans();
const EXACT_P = FLIPS.filter(m => Math.abs(m) >= Math.abs(OBS) - 1e-9).length / FLIPS.length;
const pFor = (t: number, df: number, side = state.side) => (side === 'two' ? 2 * tSf(Math.abs(t), df) : side === 'right' ? tSf(t, df) : 1 - tSf(t, df));
const SUP: Record<string, string> = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const fmtP = (p: number) => { if (p >= 1e-4) return vn(p, 4); const e = Math.floor(Math.log10(p)); return `${vn(p / 10 ** e, 2)}·10${String(e).split('').map(c => SUP[c]).join('')}`; };

// ---------- Chapter 1: H0 and H1 ----------
const chapter1: Scene = {
  title: 'Giả thuyết không và giả thuyết đối',
  duration: 11,
  beats: () => [
    { at: 0, text: '1908, “Student” (William Gosset) dùng dữ liệu này để giới thiệu kiểm định t: số giờ ngủ thêm của 10 bệnh nhân khi dùng hai loại thuốc.' },
    { at: 3, text: `Mỗi bệnh nhân thử cả hai thuốc (thiết kế cặp). Chênh lệch thuốc 2 − thuốc 1 trung bình ${vn(OBS, 2)} giờ.` },
    { at: 6, text: 'H₀ (giả thuyết không): hai thuốc như nhau, chênh lệch trung bình bằng 0. H₁ (giả thuyết đối): chênh lệch khác 0.' },
    { at: 8.5, text: 'Câu hỏi của kiểm định: nếu H₀ đúng, thấy chênh lệch lớn như 1,58 giờ (hoặc hơn) chỉ do ngẫu nhiên thì hiếm đến mức nào?' },
  ],
  formula: `H₀: μ_d = 0   H₁: μ_d ≠ 0\nd̄ = ${vn(OBS, 2)} giờ, s_d = ${vn(Math.sqrt(variance(DIFF)), 3)}, n = ${DIFF.length}`,
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.2, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, 'Số giờ ngủ thêm của từng bệnh nhân', C.violet);
    const plot = new Plot(g, left, [-0.5, 9.5], [-2.2, 6], { l: 44, r: 16, t: 40, b: 30 });
    plot.axes({ yTicks: [-2, 0, 2, 4, 6], xTicks: [], yLabel: 'giờ' });
    const show = seg(t, 0.5, 3);
    S.sleep.drug1.forEach((v, i) => {
      if (i / 10 > show) return;
      const w = S.sleep.drug2[i];
      plot.line([[i, v], [i, w]], alpha(C.white, 0.35), { width: 1.5 });
      g.circle(plot.px(i), plot.py(v), 6, C.cyan, { glow: 8 });
      g.circle(plot.px(i), plot.py(w), 6, C.amber, { glow: 8 });
      g.text(String(i + 1), plot.px(i), plot.box.y + plot.box.h + 14, { size: 10.5, color: C.muted, align: 'center' });
    });
    g.text('● thuốc 1   ● thuốc 2', left.x + left.w - 16, left.y + 50, { size: 11.5, color: C.ink, align: 'right' });
    const dp = new Plot(g, right, [-1.5, 5], [0, 1], { l: 20, r: 20, t: 40, b: 36 });
    g.panel(right.x, right.y, right.w, right.h, 'Chênh lệch d = thuốc 2 − thuốc 1', C.amber);
    dp.axes({ xTicks: [-1, 0, 1, 2, 3, 4, 5], yTicks: [], xLabel: 'giờ' });
    const reveal = seg(t, 3, 5);
    DIFF.forEach((v, i) => { if (i / 10 > reveal) return; const stack = DIFF.slice(0, i).filter(u => Math.abs(u - v) < 0.25).length; g.circle(dp.px(v), dp.py(0.12 + stack * 0.12), 7, C.amber, { glow: 8 }); });
    if (t > 6) { dp.line([[0, 0], [0, 0.9]], C.white, { width: 1.5, dash: [5, 4] }); g.tag('H₀: μ_d = 0', dp.px(0), dp.py(0.9) - 10, C.white); }
    if (t > 3.5) { dp.line([[OBS, 0], [OBS, 0.75]], C.lime, { width: 2.4, glow: 10 }); g.tag(`d̄ = ${vn(OBS, 2)}`, dp.px(OBS), dp.py(0.75) - 10, C.lime); }
  },
  metrics: () => [{ label: 'n', value: '10 cặp', tone: 'violet' }, { label: 'd̄', value: `${vn(OBS, 2)} giờ`, tone: 'lime' }],
  table: () => ({ headers: ['Bệnh nhân', 'Thuốc 1', 'Thuốc 2', 'd'], rows: DIFF.map((d, i) => [i + 1, vn(S.sleep.drug1[i], 1), vn(S.sleep.drug2[i], 1), vn(d, 1)]) }),
};

// ---------- Chapter 2: the null distribution ----------
const chapter2: Scene = {
  title: 'Nếu H₀ đúng thì kết quả trông thế nào?',
  duration: 12,
  beats: () => [
    { at: 0, text: 'Nếu hai thuốc như nhau thì nhãn “thuốc 1/thuốc 2” của mỗi bệnh nhân hoán đổi được: dấu của mỗi chênh lệch là may rủi.' },
    { at: 3, text: 'Đổi dấu mọi cách có thể: 2¹⁰ = 1 024 tổ hợp, mỗi tổ hợp cho một d̄. Đó là phân phối chính xác của d̄ dưới H₀.' },
    { at: 7, text: `Chỉ ${FLIPS.filter(m => Math.abs(m) >= Math.abs(OBS) - 1e-9).length} trên 1 024 tổ hợp cho |d̄| ≥ ${vn(OBS, 2)}: p-value = ${vn(EXACT_P, 4)}.` },
    { at: 9.5, text: `Kiểm định t dùng công thức thay vì đếm: t = ${vn(PT.t, 2)}, p = ${vn(S.sleep.p, 4)}. Hai cách cho cùng kết luận.` },
  ],
  formula: `p-value = P(|d̄*| ≥ |d̄_obs| | H₀) = ${FLIPS.filter(m => Math.abs(m) >= Math.abs(OBS) - 1e-9).length} / 1 024 = ${vn(EXACT_P, 4)}`,
  render(f) {
    const { g, t } = f; currentT = t;
    const area = f.safe;
    g.panel(area.x, area.y, area.w, area.h, 'Phân phối của d̄ khi đổi dấu ngẫu nhiên (1 024 tổ hợp)', C.violet);
    const plot = new Plot(g, area, [-2, 2], [0, 1], { l: 30, r: 20, t: 40, b: 34 });
    plot.axes({ xTicks: [-2, -1.58, -1, 0, 1, 1.58, 2], yTicks: [], fx: v => vn(v, 2), xLabel: 'd̄ (giờ)' });
    const bins = 60, counts = new Array(bins).fill(0), shown = Math.floor(seg(t, 3, 7) * FLIPS.length);
    FLIPS.slice(0, shown).forEach(m => { counts[Math.min(bins - 1, Math.max(0, Math.floor((m + 2) / 4 * bins)))]++; });
    const peak = 75;
    counts.forEach((c, i) => {
      const x0 = -2 + 4 * i / bins, x1 = x0 + 4 / bins, extreme = Math.abs((x0 + x1) / 2) >= OBS - 0.02;
      g.ctx.fillStyle = alpha(extreme && t > 7 ? C.red : C.cyan, 0.75);
      const h = (plot.py(0) - plot.py(0.9)) * Math.min(1, c / peak);
      g.ctx.fillRect(plot.px(x0) + 0.5, plot.py(0) - h, plot.px(x1) - plot.px(x0) - 1, h);
    });
    [-OBS, OBS].forEach(v => plot.line([[v, 0], [v, 0.95]], C.lime, { width: 2, glow: 8, dash: v < 0 ? [5, 4] : undefined }));
    g.tag(`d̄ quan sát = ${vn(OBS, 2)}`, plot.px(OBS), plot.py(0.95) - 10, C.lime);
    if (t > 7) g.tag(`p = ${vn(EXACT_P, 4)}`, plot.px(0), plot.py(0.95) - 10, C.red);
  },
  metrics: () => [{ label: 'Tổ hợp', value: '1 024', tone: 'violet' }, { label: 'p (chính xác)', value: vn(EXACT_P, 4), tone: 'red' }, { label: 'p (t-test)', value: vn(S.sleep.p, 4), tone: 'amber' }],
};

// ---------- Chapter 3: t distribution, alpha and p ----------
const chapter3: Scene = {
  title: 'p-value, mức ý nghĩa α và vùng bác bỏ',
  duration: 12,
  idle: false,
  beats: () => {
    const p = pFor(PT.t, PT.df);
    return [
      { at: 0, text: `Dưới H₀, thống kê t = d̄ / (s_d/√n) theo phân phối Student với ${PT.df} bậc tự do.` },
      { at: 3, text: `Chọn trước mức ý nghĩa α = ${vn(state.alpha, 2)}: xác suất chấp nhận bác bỏ H₀ khi nó đúng (sai lầm loại I). Vùng đỏ là vùng bác bỏ.` },
      { at: 6.5, text: `t quan sát = ${vn(PT.t, 2)}. p-value là diện tích phần đuôi từ t trở ra (${state.side === 'two' ? 'cả hai phía' : 'một phía'}): p = ${vn(p, 4)}.` },
      { at: 9.5, text: p <= state.alpha ? `p ≤ α nên t nằm trong vùng bác bỏ: bác bỏ H₀. Có bằng chứng thuốc 2 làm ngủ thêm khác thuốc 1.` : 'p > α: không bác bỏ H₀ (không có nghĩa H₀ đúng).' },
    ];
  },
  formula: () => `t = d̄ / (s_d / √n) = ${vn(OBS, 2)} / ${vn(PT.se, 3)} = ${vn(PT.t, 3)}\np = ${state.side === 'two' ? 'P(|T| ≥ |t|)' : state.side === 'right' ? 'P(T ≥ t)' : 'P(T ≤ t)'} = ${vn(pFor(PT.t, PT.df), 5)},  α = ${vn(state.alpha, 2)}`,
  render(f) {
    const { g, t } = f; currentT = t;
    const area = f.safe;
    g.panel(area.x, area.y, area.w, area.h, `Phân phối t với ${PT.df} bậc tự do`, C.violet);
    const plot = new Plot(g, area, [-5, 5], [0, 0.42], { l: 40, r: 20, t: 40, b: 34 });
    plot.axes({ xTicks: [-4, -2, 0, 2, 4], yTicks: [0, 0.2, 0.4], xLabel: 't' });
    const tail = state.side === 'two' ? state.alpha / 2 : state.alpha, crit = criticalT(tail, PT.df);
    const fill = (from: number, to: number, color: string) => { const pts: Point[] = [[from, 0]]; for (let x = from; x <= to; x += 0.02) pts.push([x, tPdf(x, PT.df)]); pts.push([to, 0]); g.polygon(pts.map(p => [plot.px(p[0]), plot.py(p[1])] as Point), alpha(color, 0.45)); };
    if (t > 3) { if (state.side !== 'left') fill(crit, 5, C.red); if (state.side !== 'right') fill(-5, -crit, C.red); }
    if (t > 6.5) { const o = PT.t; if (state.side !== 'left') fill(Math.abs(o), 5, C.amber); if (state.side === 'two') fill(-5, -Math.abs(o), C.amber); if (state.side === 'left') fill(-5, o, C.amber); }
    plot.curve(x => tPdf(x, PT.df), C.cyan, { width: 2.4, glow: 10 });
    plot.curve(x => normPdf(x), alpha(C.white, 0.35), { width: 1.2, dash: [4, 4] });
    if (t > 3) { g.tag(`±${vn(crit, 3)}`, plot.px(crit), plot.py(0.08) - 8, C.red); }
    if (t > 6.5) { plot.line([[PT.t, 0], [PT.t, 0.3]], C.lime, { width: 2.4, glow: 10 }); g.tag(`t = ${vn(PT.t, 2)}`, plot.px(PT.t), plot.py(0.3) - 10, C.lime); }
    g.text('nét đứt: phân phối chuẩn (t tiến tới nó khi n lớn)', area.x + area.w - 20, area.y + 52, { size: 11, color: C.muted, align: 'right' });
  },
  metrics: () => { const p = pFor(PT.t, PT.df); return [{ label: 'α', value: vn(state.alpha, 2), tone: 'red' }, { label: 'p-value', value: vn(p, 4), tone: 'amber' }, { label: 'Kết luận', value: p <= state.alpha ? 'bác bỏ H₀' : 'không bác bỏ H₀', tone: p <= state.alpha ? 'lime' : 'violet' }]; },
};

// ---------- Chapter 4: common tests on real data ----------
const jm = mean(S.mpg.japan), em = mean(S.mpg.europe);
const fr = S.titanic.female[0] / S.titanic.female[1], mr = S.titanic.male[0] / S.titanic.male[1];
const TESTS = [
  { name: 't-test cặp', data: 'Giấc ngủ (Student 1908)', stat: `t = ${vn(S.sleep.t, 2)}`, p: S.sleep.p, note: `d̄ = ${vn(OBS, 2)} giờ, 10 cặp` },
  { name: 't-test độc lập (Welch)', data: 'Auto MPG: xe Nhật vs xe châu Âu', stat: `t = ${vn(S.mpg.welch.t, 2)}`, p: S.mpg.welch.p, note: `${vn(jm, 1)} vs ${vn(em, 1)} dặm/gallon (n = ${S.mpg.japan.length}, ${S.mpg.europe.length})` },
  { name: 'z cho 2 tỷ lệ', data: 'Titanic: nữ vs nam', stat: `z = ${vn(S.titanic.z, 1)}`, p: S.titanic.pz, note: `sống sót ${vn(fr * 100, 1)}% vs ${vn(mr * 100, 1)}%` },
  { name: 'Chi-square bảng chéo', data: 'Titanic: hạng vé × sống sót', stat: `χ² = ${vn(S.titanic.chi2, 1)}`, p: S.titanic.pChi2, note: `${S.titanic.dof} bậc tự do, 891 hành khách` },
];
const testNow = () => state.test ?? Math.min(3, Math.floor(seg(currentT, 0.5, 11) * 4));
const chapter4: Scene = {
  title: 'Các phép kiểm định thường dùng',
  duration: 12,
  idle: false,
  beats: () => [
    { at: 0, text: 'Thiết kế cặp (cùng người, hai điều kiện): t-test cặp trên các chênh lệch.' },
    { at: 3, text: `Hai nhóm độc lập, dữ liệu định lượng: t-test độc lập. Xe Nhật trung bình ${vn(jm, 1)}, xe châu Âu ${vn(em, 1)} dặm/gallon: p = ${vn(S.mpg.welch.p, 4)}.` },
    { at: 6, text: `Hai tỷ lệ: kiểm định z. Trên Titanic, ${vn(fr * 100, 0)}% phụ nữ sống sót so với ${vn(mr * 100, 0)}% đàn ông: z = ${vn(S.titanic.z, 1)}, p cỡ 10⁻⁵⁹.` },
    { at: 9, text: 'Bảng chéo nhiều nhóm: chi-square so sánh số đếm quan sát O với số kỳ vọng E nếu hai biến độc lập.' },
  ],
  formula: () => ['t = d̄ / SE_d', 't = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂)', 'z = (p̂₁ − p̂₂) / √(p̂(1 − p̂)(1/n₁ + 1/n₂))', 'χ² = Σ (O − E)² / E'][testNow()],
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.1, 1], narrow);
    const k = testNow(), test = TESTS[k];
    g.panel(left.x, left.y, left.w, left.h, `${test.name} · ${test.data}`, C.violet);
    if (k === 0 || k === 1) {
      const plot = new Plot(g, left, k === 0 ? [-2, 6] : [15, 48], [0, 1], { l: 20, r: 20, t: 40, b: 34 });
      plot.axes({ xTicks: k === 0 ? [-2, 0, 2, 4, 6] : [20, 30, 40], yTicks: [], xLabel: k === 0 ? 'giờ' : 'dặm/gallon' });
      const groups = k === 0 ? [[S.sleep.drug1, C.cyan], [S.sleep.drug2, C.amber]] as const : [[S.mpg.europe, C.cyan], [S.mpg.japan, C.amber]] as const;
      groups.forEach(([vals, color], gi) => { vals.forEach((v, i) => g.circle(plot.px(v), plot.py(0.25 + gi * 0.4 + ((i * 37) % 11) * 0.012), 3.6, color, { alpha: 0.8 })); const m = mean([...vals]); plot.line([[m, 0.18 + gi * 0.4], [m, 0.42 + gi * 0.4]], color, { width: 3, glow: 10 }); });
      g.text(k === 0 ? 'thuốc 1 (dưới) · thuốc 2 (trên)' : 'châu Âu (dưới) · Nhật (trên)', left.x + 16, left.y + left.h - 14, { size: 11, color: C.muted });
    } else if (k === 2) {
      barChart(f, { x: left.x + 4, y: left.y + 30, w: left.w - 8, h: left.h - 34 }, 'Tỷ lệ sống sót', ['nữ', 'nam'], [fr * 100, mr * 100], { colors: [C.amber, C.cyan], max: 100, format: v => `${vn(v, 1)}%` });
    } else {
      const tbl = S.titanic.classTable, cw = (left.w - 80) / 2, ch = Math.min(60, (left.h - 110) / 3);
      ['chết', 'sống'].forEach((h, j) => g.text(h, left.x + 70 + j * cw + cw / 2, left.y + 56, { size: 12, color: C.muted, align: 'center' }));
      tbl.forEach((row, i) => { g.text(`hạng ${i + 1}`, left.x + 60, left.y + 76 + i * ch + ch / 2, { size: 12, color: C.ink, align: 'right' }); row.forEach((o, j) => { const e = S.titanic.expected[i][j], x = left.x + 70 + j * cw, y = left.y + 70 + i * ch; g.ctx.fillStyle = alpha(o > e ? C.amber : C.cyan, Math.min(0.85, Math.abs(o - e) / 80 + 0.1)); g.ctx.fillRect(x + 3, y + 3, cw - 6, ch - 6); g.text(`O = ${o}`, x + cw / 2, y + ch / 2 - 7, { size: 12.5, color: C.ink, align: 'center', weight: 650 }); g.text(`E = ${vn(e, 1)}`, x + cw / 2, y + ch / 2 + 9, { size: 11.5, color: C.ink, align: 'center', shadow: true }); }); });
    }
    g.panel(right.x, right.y, right.w, right.h, 'Kết quả (scipy.stats)', C.amber);
    TESTS.forEach((ts, i) => {
      const y = right.y + 56 + i * Math.min(72, (right.h - 70) / 4), on = i === k;
      g.text(ts.name, right.x + 16, y, { size: 13, color: on ? C.amber : C.ink, weight: 650 });
      g.text(`${ts.stat} · p = ${fmtP(ts.p)}`, right.x + 16, y + 18, { size: 12, color: on ? C.ink : C.muted, mono: true });
      g.text(ts.note, right.x + 16, y + 35, { size: 11, color: C.muted });
    });
  },
  metrics: () => { const ts = TESTS[testNow()]; return [{ label: 'Phép kiểm định', value: ts.name, tone: 'violet' }, { label: 'p-value', value: fmtP(ts.p), tone: ts.p < 0.05 ? 'lime' : 'red' }]; },
  table: () => ({ headers: ['Mục tiêu', 'Dữ liệu', 'Phép kiểm định', 'Thống kê', 'p-value'], rows: TESTS.map(ts => [ts.note, ts.data, ts.name, ts.stat, fmtP(ts.p)]) }),
};

// ---------- Chapter 5: errors and power ----------
const RUNS = 2000;
let cache: { key: string; h0: number[]; h1: number[] } | null = null;
const sims = () => { const key = `${state.n}-${state.d}`; if (cache?.key !== key) cache = { key, h0: simulateP(state.n, 0, RUNS, 7), h1: simulateP(state.n, state.d, RUNS, 11) }; return cache; };
const NS = [5, 10, 20, 30, 50, 80, 120, 200];
const chapter5: Scene = {
  title: 'Sai lầm loại I, loại II và lực kiểm định',
  duration: 12,
  idle: false,
  beats: () => {
    const s = sims(), fp = s.h0.filter(p => p <= state.alpha).length / RUNS, pw = s.h1.filter(p => p <= state.alpha).length / RUNS;
    return [
      { at: 0, text: `Mô phỏng ${RUNS} thí nghiệm khi H₀ đúng (không có hiệu ứng): p-value phân bố đều từ 0 tới 1.` },
      { at: 3, text: `Vì vậy khoảng α = ${vn(state.alpha * 100, 0)}% số thí nghiệm vẫn cho p ≤ α: đó là sai lầm loại I (dương tính giả). Đo được ${vn(fp * 100, 1)}%.` },
      { at: 6, text: `Khi có hiệu ứng thật (d = ${vn(state.d, 1)}, n = ${state.n} mỗi nhóm), p dồn về gần 0. Tỷ lệ bác bỏ đúng là lực kiểm định (power): ${vn(pw * 100, 1)}%.` },
      { at: 9, text: 'Phần còn lại là sai lầm loại II (bỏ sót hiệu ứng). Muốn lực 80% với d = 0,5 cần khoảng 64 người mỗi nhóm.' },
    ];
  },
  formula: () => `α = P(bác bỏ H₀ | H₀ đúng),  β = P(không bác bỏ | H₁ đúng),  power = 1 − β\nn = ${state.n}, d = ${vn(state.d, 1)}: power ≈ ${vn(power(state.n, state.d, state.alpha) * 100, 1)}% (xấp xỉ chuẩn)`,
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.2, 1], narrow);
    const [a, b] = split(left, [1, 1], true, 12);
    const s = sims(), hist = (ps: number[]) => { const c = new Array(20).fill(0); ps.forEach(p => c[Math.min(19, Math.floor(p * 20))]++); return c.map(v => v / RUNS * 100); };
    const labels = Array.from({ length: 20 }, (_, i) => (i % 5 === 0 ? vn(i / 20, 2) : ''));
    barChart(f, a, `H₀ đúng: p-value của ${RUNS} thí nghiệm (%)`, labels, hist(s.h0), { max: 12, reveal: seg(t, 0.5, 3), colors: hist(s.h0).map((_, i) => (i / 20 < state.alpha ? C.red : C.cyan)), format: () => '', accent: C.red });
    barChart(f, b, `H₁ đúng (d = ${vn(state.d, 1)}): p-value (%)`, labels, hist(s.h1), { max: 100, reveal: seg(t, 6, 8), colors: hist(s.h1).map((_, i) => (i / 20 < state.alpha ? C.lime : C.cyan)), format: () => '', accent: C.lime });
    categoryChart(f, right, `Lực kiểm định theo cỡ mẫu (α = ${vn(state.alpha, 2)})`, NS.map(String), [
      { label: 'd = 0,2', values: NS.map(n => power(n, 0.2, state.alpha)), color: C.cyan, format: v => `${vn(v * 100, 0)}%` },
      { label: 'd = 0,5', values: NS.map(n => power(n, 0.5, state.alpha)), color: C.amber, format: v => `${vn(v * 100, 0)}%` },
      { label: 'd = 0,8', values: NS.map(n => power(n, 0.8, state.alpha)), color: C.lime, format: v => `${vn(v * 100, 0)}%` },
    ], [0, 1], NS.indexOf(state.n), seg(t, 9, 11), v => `${vn(v * 100, 0)}%`, 'n mỗi nhóm');
  },
  metrics: () => { const s = sims(); return [{ label: 'Loại I (đo)', value: `${vn(s.h0.filter(p => p <= state.alpha).length / RUNS * 100, 1)}%`, tone: 'red' }, { label: 'Power (đo)', value: `${vn(s.h1.filter(p => p <= state.alpha).length / RUNS * 100, 1)}%`, tone: 'lime' }]; },
};

// ---------- Chapter 6: misreadings ----------
const NOISE = simulateP(30, 0, 20, 1);
const chapter6: Scene = {
  title: 'Những cách hiểu sai phổ biến',
  duration: 12,
  beats: () => [
    { at: 0, text: 'p-value KHÔNG phải xác suất H₀ đúng. Nó là xác suất thấy dữ liệu (hoặc cực đoan hơn) khi giả định H₀ đúng.' },
    { at: 3, text: `Chạy 20 kiểm định trên dữ liệu hoàn toàn ngẫu nhiên: xác suất ít nhất một p < 0,05 là 1 − 0,95²⁰ ≈ 64%. Ở đây ${NOISE.filter(p => p < 0.05).length} kiểm định “có ý nghĩa” dù không có hiệu ứng nào.` },
    { at: 7, text: 'Thử nhiều cách rồi chỉ báo cáo kết quả p < 0,05 là p-hacking. Cần hiệu chỉnh (Bonferroni: α/m) hoặc đăng ký trước giả thuyết.' },
    { at: 9.5, text: 'Ý nghĩa thống kê ≠ ý nghĩa thực tiễn: với mẫu cực lớn, chênh lệch rất nhỏ cũng cho p rất nhỏ. Luôn báo cáo kích thước hiệu ứng và khoảng tin cậy.' },
  ],
  formula: 'P(ít nhất 1 dương tính giả trong m kiểm định) = 1 − (1 − α)^m;  Bonferroni: dùng α/m',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1.3, 1], narrow);
    barChart(f, left, '20 kiểm định trên nhiễu thuần túy: p-value', NOISE.map((_, i) => String(i + 1)), NOISE, { max: 1.05, reveal: seg(t, 3, 5.5), colors: NOISE.map(p => (p < 0.05 / (t > 7 ? 20 : 1) ? C.red : p < 0.05 ? C.amber : C.cyan)), format: p => (p < 0.05 ? vn(p, 3) : '') });
    g.panel(right.x, right.y, right.w, right.h, 'Hiểu sai → hiểu đúng', C.amber);
    const rows = [['p là xác suất H₀ đúng', 'p là xác suất dữ liệu (hoặc cực đoan hơn) khi H₀ đúng'], ['p < 0,05 ⇒ quan trọng thực tế', 'ý nghĩa thống kê ≠ ý nghĩa thực tiễn'], ['p lớn ⇒ H₀ đúng', 'chỉ là chưa đủ bằng chứng để bác bỏ H₀'], ['chỉ cần nhìn p', 'xem thêm kích thước hiệu ứng, khoảng tin cậy, cỡ mẫu, thiết kế']];
    rows.forEach(([bad, good], i) => {
      const y = right.y + 56 + i * Math.min(76, (right.h - 70) / 4), on = seg(t, i * 1.5, i * 1.5 + 1.5);
      g.text(`✗ ${bad}`, right.x + 16, y, { size: 12, color: C.red, alpha: on });
      g.text(`✓ ${good}`, right.x + 16, y + 20, { size: 12, color: C.lime, alpha: on });
    });
    if (t > 7) g.tag(`Bonferroni: α/20 = 0,0025`, left.x + left.w - 16, left.y + 52, C.red, 'right');
  },
  metrics: () => [{ label: 'p < 0,05 (trên nhiễu)', value: `${NOISE.filter(p => p < 0.05).length}/20`, tone: 'amber' }, { label: 'Sau Bonferroni', value: `${NOISE.filter(p => p < 0.0025).length}/20`, tone: 'lime' }],
};

export const initHypothesisLab = () => glassLab('hypothesis', [chapter1, chapter2, chapter3, chapter4, chapter5, chapter6], lab => {
  const alphas = [0.01, 0.05, 0.1];
  lab.range('alpha', value => vn(alphas[value], 2), value => { state.alpha = alphas[value]; });
  lab.on('input[name="side"]', 'change', el => { state.side = (el as HTMLInputElement).value as typeof state.side; lab.invalidate(); });
  lab.range('n', value => String(NS[value]), value => { state.n = NS[value]; });
  lab.range('d', value => vn(value / 10, 1), value => { state.d = value / 10; });
  lab.range('test', value => TESTS[value].name, value => { state.test = value; });
});
