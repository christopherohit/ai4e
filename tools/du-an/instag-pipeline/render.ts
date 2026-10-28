// InsTaG mechanics lab: how one pixel is splatted from Gaussians, why a motion prior learned from other people lets a
// few seconds of video suffice, what the trained motion field actually does (real Δμ statistics), why the inside of the
// mouth is hooked to the face branch, and the held-out numbers of the two trained models.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, Plot, alpha, seg, split, vn, type Point, type Rect } from '../../_shared/glass-lab/gfx';
import { barChart, categoryChart } from '../../_shared/glass-lab/views';
import { COUNTS, CURVE, IDENTITIES, LIP_OPEN, META, MOTION, MOTION_CORR, MOUTH_FREE, MOUTH_HOOK, N, PRIOR, RESULTS, SPLATS, TARGET, compositeAt, fitAligned, fitScratch, mismatch, samples, testError } from './model';

export const explainerKind = 'educational-simulation';

const state = { n: 10 };
let currentT = 0;
const ID_COLORS = [C.cyan, C.violet, C.amber, C.magenta, '#5b8cff'];

// ---------- Chapter 1: splatting one pixel ----------
function splats(f: Frame, rect: Rect, px: number, py: number, show: number) {
  const { g } = f, s = Math.min(rect.w, rect.h - 50), ox = rect.x + (rect.w - s) / 2, oy = rect.y + 40;
  const order = [...SPLATS].sort((a, b) => b.depth - a.depth);  // draw far to near
  for (const sp of order) {
    g.ctx.save(); g.ctx.translate(ox + sp.x * s, oy + sp.y * s); g.ctx.rotate(sp.rot);
    const grad = g.ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
    grad.addColorStop(0, alpha(sp.color, sp.opacity * 0.85)); grad.addColorStop(1, alpha(sp.color, 0));
    g.ctx.scale(sp.sx * s * 2, sp.sy * s * 2); g.ctx.fillStyle = grad; g.ctx.beginPath(); g.ctx.arc(0, 0, 1, 0, Math.PI * 2); g.ctx.fill(); g.ctx.restore();
  }
  const x = ox + px * s, y = oy + py * s;
  g.ctx.save(); g.ctx.globalAlpha = show; g.ctx.strokeStyle = C.white; g.ctx.lineWidth = 1.5; g.ctx.strokeRect(x - 5, y - 5, 10, 10); g.ctx.restore();
  g.line([[x - 14, y], [x - 6, y]], C.white, { width: 1.2, alpha: show }); g.line([[x + 6, y], [x + 14, y]], C.white, { width: 1.2, alpha: show });
  g.text('pixel', x + 10, y - 12, { size: 11, color: C.ink, alpha: show });
}
const pixelAt = (time: number): [number, number] => [0.5 + 0.12 * Math.cos(time * 0.5), 0.52 + 0.08 * Math.sin(time * 0.7)];
const chapter1: Scene = {
  title: 'Gaussian splatting: một pixel được tô thế nào',
  duration: 11,
  idle: true,
  beats: () => [
    { at: 0, text: `InsTaG biểu diễn khuôn mặt bằng hàng nghìn Gaussian 3D: mô hình đã huấn luyện có ${META.faceCount.toLocaleString('vi-VN')} Gaussian cho mặt và ${META.mouthCount} cho khoang miệng.` },
    { at: 2.5, text: 'Mỗi Gaussian có tâm μ, kích thước s, hướng q (quaternion), độ đục α và màu. Chiếu lên ảnh, nó thành một vết elip mờ.' },
    { at: 5, text: 'Để tô một pixel: xếp các vết theo độ sâu từ gần tới xa, mỗi vết che một phần αᵢ, phần còn lại Tᵢ cho các vết phía sau.' },
    { at: 8, text: 'Không có tia, không lấy mẫu dọc tia như NeRF: chỉ là xếp lớp các vết đã chiếu. Vì vậy 3DGS render nhanh hơn NeRF nhiều lần.' },
  ],
  formula: 'αᵢ = oᵢ · exp(−½ dᵀ Σᵢ′⁻¹ d),   C = Σᵢ Tᵢ αᵢ cᵢ,   Tᵢ = Πⱼ<ᵢ (1 − αⱼ)\nΣᵢ′ = J W Σᵢ Wᵀ Jᵀ (hiệp phương sai 3D chiếu lên ảnh)',
  render(f) {
    const { t, time, narrow } = f; currentT = t;
    const [px, py] = pixelAt(t < 5 ? 0 : time);
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1, 1.1], narrow);
    f.g.panel(left.x, left.y, left.w, left.h, 'Sáu Gaussian đã chiếu lên ảnh · mô phỏng', C.amber);
    splats(f, left, px, py, seg(t, 4.5, 5.5));
    const rows = compositeAt(px, py);
    barChart(f, right, 'Đóng góp vào pixel, gần → xa (w = T·α)', rows.map(r => `z=${vn(r.depth, 2)}`), rows.map(r => r.w), { max: 1, reveal: seg(t, 5, 7.5), format: v => vn(v, 2), colors: rows.map(r => r.color), accent: C.amber });
  },
  metrics: () => { const rows = compositeAt(...pixelAt(currentT < 5 ? 0 : performance.now() / 1000)); const T = rows.reduce((s, r) => s * (1 - r.alpha), 1); return [{ label: 'Tổng trọng số', value: vn(1 - T, 2), tone: 'amber' }, { label: 'Gaussian mặt + miệng', value: `${META.faceCount} + ${META.mouthCount}`, tone: 'cyan' }]; },
};

// ---------- Chapter 2: identity-free prior and few-shot adaptation ----------
const chapter2: Scene = {
  title: 'Prior chuyển động: vì sao 10 giây là đủ',
  duration: 13,
  beats: () => {
    const c = CURVE.find(x => x.n >= state.n) ?? CURVE.at(-1)!;
    return [
      { at: 0, text: 'Mô phỏng: trục ngang là một đặc trưng âm thanh (âm mở miệng tới đâu), trục dọc là độ mở miệng. Năm người dùng khi huấn luyện trước (đường mảnh) có cùng dáng nhưng biên độ và độ lệch khác nhau.' },
      { at: 3, text: 'Universal Motion Field học phần chung của họ (đường trắng). Phần riêng của từng người đi vào một trường cá nhân nhỏ; loss tương phản âm ngăn phần riêng “ăn” mất phần chung.' },
      { at: 6, text: `Người mới (đường xanh lá đứt) chỉ có ${state.n} frame, và video ngắn chỉ chứa một khoảng âm thanh (vùng sáng). Học từ đầu (đỏ) khớp các điểm nhưng đoán bừa ở âm chưa gặp.` },
      { at: 9, text: `Có prior, chỉ cần học độ lệch δ (hỏi prior ở đâu) và hệ số τ (chuyển động lớn bao nhiêu): đó là Motion Aligner. Trung vị sai số trên 101 đoạn video ngắn: ${vn(c.aligned, 3)} so với ${vn(c.scratch, 3)} khi học từ đầu.` },
    ];
  },
  formula: () => { const a = fitAligned(samples(state.n)); return `học từ đầu: đa thức bậc 5 trên ${state.n} điểm\nprior + căn chỉnh: y ≈ τ · U(a − δ),  τ = ${vn(a.tau, 2)},  δ = ${vn(a.delta, 3)}`; },
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.3, 1] : [1.6, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, 'Độ mở miệng theo đặc trưng âm thanh · mô phỏng', C.lime);
    const plot = new Plot(g, left, [0, 1], [-0.2, 1.6], { l: 38, r: 14, t: 40, b: 34 });
    plot.axes({ xTicks: [0, 0.25, 0.5, 0.75, 1], yTicks: [0, 0.5, 1, 1.5], xLabel: 'đặc trưng âm thanh a', fx: v => vn(v, 2), fy: v => vn(v, 1) });
    const pts = samples(state.n), show = seg(t, 6, 7.5);
    g.ctx.save(); g.ctx.globalAlpha = 0.08 * show; g.ctx.fillStyle = C.white; g.ctx.fillRect(plot.px(0.15), plot.box.y, plot.px(0.65) - plot.px(0.15), plot.box.h); g.ctx.restore();
    plot.clip(() => {
      IDENTITIES.forEach((fn, i) => plot.curve(fn, ID_COLORS[i], { width: 1.2, alpha: 0.55 * seg(t, 0, 2) }));
      plot.curve(PRIOR, C.white, { width: 3, glow: 10, alpha: seg(t, 3, 4.5) });
      plot.curve(TARGET, C.lime, { width: 1.6, dash: [6, 5], alpha: show });
      if (t > 7) plot.curve(fitScratch(pts), C.red, { width: 2.2, glow: 6, alpha: seg(t, 7, 8.5) });
      if (t > 9) plot.curve(fitAligned(pts), C.lime, { width: 2.6, glow: 10, alpha: seg(t, 9, 10.5) });
    });
    pts.forEach(([a, y]) => g.circle(plot.px(a), plot.py(y), 3.2, C.ink, { alpha: show }));
    categoryChart(f, right, 'Trung vị sai số trên cả dải âm thanh', COUNTS.map(String), [
      { label: 'học từ đầu', values: CURVE.map(c => c.scratch), color: C.red, format: v => vn(v, 2) },
      { label: 'prior + căn chỉnh', values: CURVE.map(c => c.aligned), color: C.lime, format: v => vn(v, 2) },
    ], [0, 0.8], COUNTS.findIndex(n => n >= state.n), seg(t, 9, 12), v => vn(v, 1), 'số frame của người mới');
  },
  metrics: () => { const pts = samples(state.n); return [{ label: 'Học từ đầu', value: vn(testError(fitScratch(pts)), 3), tone: 'red' }, { label: 'Prior + căn chỉnh', value: vn(testError(fitAligned(pts)), 3), tone: 'lime' }]; },
  table: () => ({ headers: ['Số frame', 'Học từ đầu', 'Prior + căn chỉnh'], rows: CURVE.map(c => [c.n, vn(c.scratch, 3), vn(c.aligned, 3)]), caption: 'Trung vị sai số RMS trên toàn dải a ∈ [0, 1], 101 đoạn ngắn ngẫu nhiên mỗi cỡ.' }),
};

// ---------- Chapter 3: the real motion field ----------
const chapter3: Scene = {
  title: 'Motion field thật của mô hình đã huấn luyện',
  duration: 10,
  beats: () => {
    const face = MOTION.reduce((s, m) => s + m.face, 0) / MOTION.length, mouth = MOTION.reduce((s, m) => s + m.mouth, 0) / MOTION.length;
    return [
      { at: 0, text: 'Số đo thật: ở 25 frame thử nghiệm, mạng chuyển động của InsTaG dự đoán độ dịch Δμ cho từng Gaussian. Đây là độ dịch trung bình của nhánh mặt (vàng) và nhánh miệng (hồng).' },
      { at: 3, text: `Trung bình nhánh mặt ${vn(face * 1000, 1)}·10⁻³, nhánh miệng ${vn(mouth * 1000, 1)}·10⁻³ (đơn vị không gian chuẩn hóa của đầu). Nhánh miệng nằm sâu bên trong, dịch ít hơn.` },
      { at: 6, text: `Cả hai đi cùng độ mở môi AU25 đo bằng OpenFace (trắng): tương quan ${vn(MOTION_CORR.faceAu, 2)} (mặt) và ${vn(MOTION_CORR.mouthAu, 2)} (miệng) trên 25 frame. Trung bình trên hàng nghìn Gaussian gồm cả trán, má, nên tương quan không gần 1.` },
    ];
  },
  formula: 'δ = MLP( H(μ) ⊕ a ⊕ e ) = { Δμ, Δs, Δq },   θ̃ = θ + δ',
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.3, 1] : [1.7, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, 'Độ dịch Δμ trung bình theo frame (chuẩn hóa) và AU25', C.amber);
    const plot = new Plot(g, left, [0, N - 1], [0, 1.1], { l: 30, r: 14, t: 40, b: 34 });
    plot.axes({ xTicks: [0, 50, 100, 150, 200, 249], yTicks: [], xLabel: 'frame thử nghiệm', fx: v => String(Math.round(v)) });
    const nz = (k: 'face' | 'mouth' | 'au25') => { const v = MOTION.map(m => m[k]); const lo = Math.min(...v), hi = Math.max(...v); return MOTION.map((m, i) => [m.frame, (v[i] - lo) / (hi - lo)] as Point); };
    const draw = (pts: Point[], color: string, reveal: number) => { const n = Math.ceil(pts.length * reveal); plot.line(pts.slice(0, n), color, { width: 2, glow: 6 }); pts.slice(0, n).forEach(p => g.circle(plot.px(p[0]), plot.py(p[1]), 3, color)); };
    draw(nz('face'), C.amber, seg(t, 0, 2.5));
    draw(nz('mouth'), C.magenta, seg(t, 0.5, 3));
    if (t > 6) draw(nz('au25'), C.white, seg(t, 6, 8));
    barChart(f, right, 'Độ dịch trung bình |Δμ| (×10⁻³)', [`mặt · ${META.faceCount} Gaussian`, `miệng · ${META.mouthCount} Gaussian`], [MOTION.reduce((s, m) => s + m.face, 0) / MOTION.length * 1e3, MOTION.reduce((s, m) => s + m.mouth, 0) / MOTION.length * 1e3], { reveal: seg(t, 3, 5), format: v => vn(v, 1), colors: [C.amber, C.magenta], accent: C.amber });
  },
  metrics: () => [{ label: 'ρ(Δμ mặt, AU25)', value: vn(MOTION_CORR.faceAu, 2), tone: 'amber' }, { label: 'ρ(Δμ miệng, AU25)', value: vn(MOTION_CORR.mouthAu, 2), tone: 'magenta' }],
  table: () => ({ headers: ['Frame', 'Δμ mặt', 'Δμ miệng', 'AU25'], rows: MOTION.map(m => [m.frame, vn(m.face, 5), vn(m.mouth, 5), vn(m.au25, 2)]) }),
};

// ---------- Chapter 4: face–mouth hook ----------
function mouthView(f: Frame, cx: number, cy: number, w: number, lip: number, inner: number, label: string, color: string) {
  const { g } = f, gap = 6 + lip * w * 0.32, inGap = 6 + inner * w * 0.32;
  g.ctx.save(); g.ctx.fillStyle = 'rgba(60,10,25,0.9)'; g.ctx.beginPath(); g.ctx.ellipse(cx, cy, w * 0.42, inGap / 2 + 2, 0, 0, Math.PI * 2); g.ctx.fill();
  g.ctx.fillStyle = '#f2efe6'; g.ctx.fillRect(cx - w * 0.28, cy - inGap / 2, w * 0.56, Math.min(10, inGap * 0.35)); g.ctx.fillRect(cx - w * 0.26, cy + inGap / 2 - Math.min(9, inGap * 0.3), w * 0.52, Math.min(9, inGap * 0.3));
  g.ctx.fillStyle = '#d9727f'; g.ctx.strokeStyle = '#ffb3be'; g.ctx.lineWidth = 1.5;
  g.ctx.beginPath(); g.ctx.ellipse(cx, cy - gap / 2 - 7, w * 0.46, 9, 0, 0, Math.PI * 2); g.ctx.fill(); g.ctx.stroke();
  g.ctx.beginPath(); g.ctx.ellipse(cx, cy + gap / 2 + 8, w * 0.44, 10, 0, 0, Math.PI * 2); g.ctx.fill(); g.ctx.stroke(); g.ctx.restore();
  g.text(label, cx, cy + w * 0.5 + 12, { size: 12, color, align: 'center', weight: 650 });
  if (Math.abs(inner - lip) > 0.15) g.text(inner > lip ? 'răng lộ qua môi' : 'môi mở, trong tối', cx, cy - w * 0.5 - 4, { size: 11, color: C.red, align: 'center' });
}
const chapter4: Scene = {
  title: 'Face–mouth hook: môi và khoang miệng cùng nhịp',
  duration: 12,
  idle: true,
  beats: () => [
    { at: 0, text: 'InsTaG tách hai nhánh Gaussian: mặt (gồm môi) và khoang miệng (răng, lưỡi). Mô phỏng này dùng độ mở môi thật AU25 làm chuyển động của nhánh mặt.' },
    { at: 3, text: `Nếu nhánh miệng tự đoán độ mở từ âm thanh với vài giây dữ liệu (đỏ), nó lệch nhịp với môi: sai lệch ${vn(mismatch(MOUTH_FREE), 3)}. Răng lộ qua môi đang khép, hoặc môi mở mà bên trong tối.` },
    { at: 7, text: `Hook: nhánh miệng đọc biên độ biến dạng của nhánh mặt (μ_dist = Δμ_max − Δμ_min) và dùng nó để co giãn chuyển động của mình (xanh lá): sai lệch ${vn(mismatch(MOUTH_HOOK), 3)}.` },
  ],
  formula: 'μ_dist = Δμ_max − Δμ_min (từ nhánh mặt),   Δμᴹ = Uᴹ(μ, {a, φ}),   μᴹ_hook = Δμᴹ ⊙ τ_M,   τ_M = MLP(Hᴹ(μ) ⊕ μ_dist)',
  render(f) {
    const { g, t, time, narrow } = f; currentT = t;
    const frame = Math.floor(time * 25) % N;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1, 1.3], narrow);
    g.panel(left.x, left.y, left.w, left.h, `Miệng ở frame ${frame}`, C.magenta);
    const w = Math.min(left.w * 0.38, left.h * 0.45), cy = left.y + left.h * 0.5;
    mouthView(f, left.x + left.w * 0.27, cy, w, LIP_OPEN[frame], MOUTH_FREE[frame], 'không hook', C.red);
    mouthView(f, left.x + left.w * 0.73, cy, w, LIP_OPEN[frame], t > 7 ? MOUTH_HOOK[frame] : LIP_OPEN[frame], 'có hook', C.lime);
    g.panel(right.x, right.y, right.w, right.h, 'Độ mở: môi (vàng) và khoang miệng', C.amber);
    const plot = new Plot(g, right, [0, N - 1], [0, 1.05], { l: 16, r: 12, t: 38, b: 30 });
    plot.axes({ xTicks: [0, 125, 249], yTicks: [], xLabel: 'frame', fx: v => String(Math.round(v)) });
    const line = (a: readonly number[], c: string, reveal: number) => plot.clip(() => plot.line(a.slice(0, Math.ceil(N * reveal)).map((v, i) => [i, v] as Point), c, { width: 1.8, glow: 5 }));
    line(LIP_OPEN, C.amber, 1);
    line(MOUTH_FREE, C.red, seg(t, 3, 5));
    if (t > 7) line(MOUTH_HOOK, C.lime, seg(t, 7, 9));
    plot.line([[frame, 0], [frame, 1.05]], C.white, { width: 1.2, alpha: 0.6 });
  },
  metrics: () => [{ label: 'Lệch không hook', value: vn(mismatch(MOUTH_FREE), 3), tone: 'red' }, { label: 'Lệch có hook', value: vn(mismatch(MOUTH_HOOK), 3), tone: 'lime' }],
};

// ---------- Chapter 5: held-out results ----------
const chapter5: Scene = {
  title: 'Kết quả thật: 10 giây so với vài phút',
  duration: 10,
  beats: () => [
    { at: 0, text: `Hai mô hình huấn luyện thật bằng code gốc trên cùng người (Obama) và đo trên cùng đoạn thử nghiệm (${RESULTS[0].test} và ${RESULTS[1].test} frame).` },
    { at: 2.5, text: `SyncTalk học từ đầu trên ${RESULTS[0].frames.toLocaleString('vi-VN')} frame (gần 5 phút) của người cần dựng. InsTaG dùng prior huấn luyện trước trên 5 người khác, rồi chỉ thích nghi với ${RESULTS[1].frames} frame (10 giây): ít hơn 29 lần.` },
    { at: 5.5, text: `Chất lượng: PSNR ${vn(RESULTS[1].psnr, 2)} so với ${vn(RESULTS[0].psnr, 2)} dB, LMD (lệch landmark môi) ${vn(RESULTS[1].lmd, 2)} so với ${vn(RESULTS[0].lmd, 2)}. Kém hơn một chút, đổi lại dữ liệu ít hơn hẳn.` },
    { at: 8, text: 'Hai phương pháp khác cả biểu diễn (NeRF và Gaussian) lẫn cách học, nên đây là so sánh hai hệ thống hoàn chỉnh, không phải chỉ riêng yếu tố dữ liệu.' },
  ],
  formula: 'PSNR = 10 log₁₀(1 / MSE) dB ↑,   LPIPS ↓,   LMD = khoảng cách landmark môi trung bình (pixel) ↓',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const parts = split(f.safe, [1, 1, 1], narrow);
    barChart(f, parts[0], 'Frame của người cần dựng', RESULTS.map(r => r.name), RESULTS.map(r => r.frames), { reveal: seg(t, 2.5, 4.5), format: v => v.toLocaleString('vi-VN'), colors: [C.violet, C.lime], accent: C.violet });
    barChart(f, parts[1], 'PSNR (dB) ↑', RESULTS.map(r => r.name), RESULTS.map(r => r.psnr), { max: 40, reveal: seg(t, 5.5, 7), format: v => vn(v, 2), colors: [C.violet, C.lime], accent: C.cyan });
    barChart(f, parts[2], 'LMD môi ↓', RESULTS.map(r => r.name), RESULTS.map(r => r.lmd), { max: 4, reveal: seg(t, 6, 7.5), format: v => vn(v, 2), colors: [C.violet, C.lime], accent: C.amber });
  },
  metrics: () => [{ label: 'Dữ liệu', value: '250 / 7 272 frame', tone: 'lime' }, { label: 'PSNR', value: `${vn(RESULTS[1].psnr, 1)} / ${vn(RESULTS[0].psnr, 1)} dB`, tone: 'violet' }],
  table: () => ({ headers: ['Mô hình', 'Frame người cần dựng', 'Frame thử nghiệm', 'PSNR', 'LPIPS', 'LMD', 'Ghi chú'], rows: RESULTS.map(r => [r.name, r.frames, r.test, vn(r.psnr, 2), vn(r.lpips, 4), vn(r.lmd, 2), r.note]) }),
};

export const initInstagPipelineLab = () => glassLab('instag-pipeline', [chapter1, chapter2, chapter3, chapter4, chapter5], lab => {
  lab.range('frames', v => `${v} frame`, v => { state.n = v; lab.invalidate(); });
});
