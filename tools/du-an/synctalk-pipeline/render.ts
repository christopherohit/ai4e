// SyncTalk mechanics lab: why audio must be read in windows (real AU25 vs audio RMS), why lips and upper face get
// separate controls (real AU45), how landmark + optical-flow tracking removes head jitter, why a tri-plane hash grid is
// small, and how a ray is rendered and the head is blended back into the original frame.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, Plot, alpha, seg, split, vn, type Point, type Rect } from '../../_shared/glass-lab/gfx';
import { barChart } from '../../_shared/glass-lab/views';
import { BEST_LAG, BLINKS, FLOW_SIGMA, FUSED, LAGS, LAG_CORR, LANDMARK_SIGMA, N, PAIRS, RAW, RAY, SIG, XS, YAW, composite, jitter, mask, memory, movingAverage, rmse, seam } from './model';

export const explainerKind = 'educational-simulation';

const state = { lag: null as number | null, window: 9, res: 512, blur: 0.08 };
let currentT = 0;
const MB = (b: number) => b / 2 ** 20;
const fmtMB = (v: number) => (v >= 1024 ? `${vn(v / 1024, 1)} GB` : `${vn(v, v < 10 ? 1 : 0)} MB`);
const lagNow = () => state.lag ?? (currentT < 2 ? 0 : Math.round(Math.min(1, (currentT - 2) / 6) * BEST_LAG));
const norm = (a: readonly number[]) => { const m = Math.max(...a); return a.map(v => v / m); };
const MOUTH = norm(SIG.mouth), RMS = norm(SIG.rms), EYE = SIG.eye.map(v => Math.min(1, v / 2));

function series(plot: Plot, values: readonly number[], color: string, shift = 0, reveal = 1, width = 1.8) {
  const pts: Point[] = [];
  const end = Math.floor(values.length * reveal);
  for (let i = 0; i < end; i++) { const x = i - shift; if (x >= 0 && x < N) pts.push([x, values[i]]); }
  plot.clip(() => plot.line(pts, color, { width, glow: 6 }));
}

// ---------- Chapter 1: audio ↔ lips needs a window ----------
const chapter1: Scene = {
  title: 'Âm thanh và môi không khớp từng frame',
  duration: 11,
  beats: () => [
    { at: 0, text: 'Hai tín hiệu thật đo trên 10 giây video thử nghiệm (25 frame/giây): độ mở môi AU25 từ OpenFace (vàng) và độ to của âm thanh RMS (xanh).' },
    { at: 2.5, text: `Đặt cạnh nhau ở cùng frame, tương quan gần như bằng 0 (${vn(PAIRS[1].value, 2)}). Nhìn kỹ: môi mở trước, tiếng to lên sau.` },
    { at: 5.5, text: `Dịch âm thanh sớm lên ${BEST_LAG} frame (${BEST_LAG * 40} ms) thì hai đường gần trùng nhau: tương quan ${vn(PAIRS[0].value, 2)}. Môi chuẩn bị hình dạng trước khi âm phát ra (đồng phát âm), và RMS chỉ đo độ to.` },
    { at: 8.5, text: 'Bài học cho mô hình: hình môi ở một frame phụ thuộc vào cả một cửa sổ âm thanh trước và sau. Bộ mã hóa audio–visual của SyncTalk học trên cửa sổ, không gán một frame âm cho một frame ảnh.' },
  ],
  formula: () => `ρ(k) = corr( AU25[t],  RMS[t + k] ),   Δt = k / 25 fps\nk = ${lagNow()}: Δt = ${lagNow() * 40} ms,  ρ = ${vn(LAG_CORR[LAGS.indexOf(lagNow())], 3)}`,
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const k = lagNow();
    const [left, right] = split(f.safe, narrow ? [1.2, 1] : [1.6, 1], narrow);
    g.panel(left.x, left.y, left.w, left.h, 'Tín hiệu thật: AU25 (môi) và RMS âm thanh', C.amber);
    const plot = new Plot(g, left, [0, N - 1], [0, 1.05], { l: 30, r: 14, t: 40, b: 34 });
    plot.axes({ xTicks: [0, 50, 100, 150, 200, 249], yTicks: [], xLabel: 'frame (25 fps)', fx: v => String(Math.round(v)) });
    series(plot, MOUTH, C.amber, 0, seg(t, 0, 2));
    series(plot, RMS, C.cyan, k, seg(t, 0.5, 2.5));
    g.text('AU25 môi mở', plot.box.x + 6, plot.box.y + 12, { size: 11.5, color: C.amber });
    g.text(k ? `RMS dịch sớm ${k} frame` : 'RMS âm thanh', plot.box.x + 6, plot.box.y + 28, { size: 11.5, color: C.cyan });
    barChart(f, right, 'Tương quan theo độ lệch k (frame)', LAGS.map(v => (v % 5 === 0 ? String(v) : '')), LAG_CORR.map(v => Math.max(0, v)), { max: 1, highlight: LAGS.indexOf(k), reveal: seg(t, 2.5, 5.5), format: v => (v > 0.7 ? vn(v, 2) : ''), color: C.cyan, accent: C.amber });
  },
  metrics: () => [{ label: 'Độ lệch', value: `${lagNow()} frame`, tone: 'cyan' }, { label: 'Tương quan', value: vn(LAG_CORR[LAGS.indexOf(lagNow())], 2), tone: 'amber' }],
  table: () => ({ headers: ['k (frame)', 'Δt (ms)', 'ρ'], rows: LAGS.map((k, i) => [k, k * 40, vn(LAG_CORR[i], 3)]) }),
};

// ---------- Chapter 2: separate lip and expression controls ----------
function face(f: Frame, rect: Rect, frame: number, show: number) {
  const { g } = f, cx = rect.x + rect.w / 2, cy = rect.y + rect.h * 0.55, rx = Math.min(rect.w * 0.26, rect.h * 0.3), ry = rx * 1.3;
  g.ctx.save(); g.ctx.fillStyle = 'rgba(255,214,180,0.10)'; g.ctx.strokeStyle = 'rgba(255,255,255,0.35)'; g.ctx.lineWidth = 1.5;
  g.ctx.beginPath(); g.ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2); g.ctx.fill(); g.ctx.stroke(); g.ctx.restore();
  // masks
  g.ctx.save(); g.ctx.globalAlpha = 0.18 * show;
  g.ctx.fillStyle = C.violet; g.ctx.fillRect(cx - rx * 0.85, cy - ry * 0.62, rx * 1.7, ry * 0.5);
  g.ctx.fillStyle = C.cyan; g.ctx.fillRect(cx - rx * 0.6, cy + ry * 0.22, rx * 1.2, ry * 0.42);
  g.ctx.restore();
  const eye = SIG.eye[frame], open = Math.max(0.08, 1 - eye / 2), brow = SIG.brow[frame] / 5, mouth = SIG.mouth[frame] / 2.2;
  for (const s of [-1, 1]) {
    const ex = cx + s * rx * 0.38, ey = cy - ry * 0.28;
    g.ctx.save(); g.ctx.fillStyle = C.ink; g.ctx.beginPath(); g.ctx.ellipse(ex, ey, rx * 0.14, rx * 0.08 * open, 0, 0, Math.PI * 2); g.ctx.fill(); g.ctx.restore();
    g.line([[ex - rx * 0.17, ey - rx * (0.2 + brow * 0.3)], [ex + rx * 0.17, ey - rx * (0.22 + brow * 0.3)]], C.ink, { width: 2.4 });
  }
  const my = cy + ry * 0.42;
  g.ctx.save(); g.ctx.fillStyle = 'rgba(120,30,50,0.9)'; g.ctx.strokeStyle = '#ff9eb0'; g.ctx.lineWidth = 2;
  g.ctx.beginPath(); g.ctx.ellipse(cx, my, rx * 0.32, Math.max(1.5, rx * 0.16 * mouth), 0, 0, Math.PI * 2); g.ctx.fill(); g.ctx.stroke(); g.ctx.restore();
  g.text('M_exp: mắt, lông mày ← AU45', cx, cy - ry * 0.7, { size: 11, color: C.violet, align: 'center', alpha: show });
  g.text('M_lip: môi, cằm ← âm thanh', cx, cy + ry * 0.74, { size: 11, color: C.cyan, align: 'center', alpha: show });
}
const chapter2: Scene = {
  title: 'Môi theo âm thanh, mắt theo biểu cảm',
  duration: 12,
  idle: true,
  beats: () => [
    { at: 0, text: 'Mặt bên trái chuyển động theo đúng tín hiệu đo từ video thật: độ mở môi AU25, chớp mắt AU45, nâng mày AU01.' },
    { at: 3, text: `Trong 10 giây có ${BLINKS} lần chớp mắt. Tương quan giữa chớp mắt và âm thanh chỉ ${vn(PAIRS[2].value, 2)}, giữa chớp mắt và môi ${vn(PAIRS[4].value, 2)}: âm thanh không đoán được lúc nào ta chớp mắt.` },
    { at: 6.5, text: 'Nếu bắt một đặc trưng audio điều khiển cả khuôn mặt, mô hình sẽ học chớp mắt “theo âm tiết”, trông rất giả. SyncTalk tách hai nguồn: đặc trưng môi f_l từ audio, đặc trưng biểu cảm f_e từ ảnh chụp khuôn mặt.' },
    { at: 9.5, text: 'Mặt nạ attention theo vùng (M_lip, M_exp) giới hạn mỗi đặc trưng chỉ tác động lên vùng của nó.' },
  ],
  formula: 'f_l = f_lip ⊙ (V ⊙ M_lip),   f_e = f_exp ⊙ (V ⊙ M_exp)\nV: attention theo vùng, ⊙: nhân từng phần tử',
  render(f) {
    const { g, t, time, narrow } = f; currentT = t;
    const frame = Math.floor(time * 25) % N;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1, 1.2], narrow);
    g.panel(left.x, left.y, left.w, left.h, `Khuôn mặt theo tín hiệu thật · frame ${frame}`, C.violet);
    face(f, left, frame, seg(t, 8.5, 10));
    const [top, bottom] = split(right, [1, 1], true, 10);
    g.panel(top.x, top.y, top.w, top.h, 'AU45 chớp mắt (tím) và RMS âm thanh (xanh)', C.violet);
    const plot = new Plot(g, top, [0, N - 1], [0, 1.05], { l: 16, r: 12, t: 36, b: 14 });
    series(plot, RMS, alpha(C.cyan, 0.8), 0, 1, 1.4);
    series(plot, EYE, C.violet, 0, seg(t, 1, 3.5), 2);
    plot.line([[frame, 0], [frame, 1.05]], C.white, { width: 1.2, alpha: 0.6 });
    barChart(f, bottom, 'Tương quan (giá trị tuyệt đối)', PAIRS.map(p => p.label), PAIRS.map(p => Math.abs(p.value)), { max: 1, reveal: seg(t, 3, 6), format: v => vn(v, 2), colors: [C.cyan, C.cyan, C.violet, C.violet, C.violet], accent: C.violet });
  },
  metrics: () => [{ label: 'Số lần chớp', value: String(BLINKS), tone: 'violet' }, { label: 'ρ(mắt, âm)', value: vn(PAIRS[2].value, 2), tone: 'cyan' }],
};

// ---------- Chapter 3: head pose without jitter ----------
const chapter3: Scene = {
  title: 'Ổn định tư thế đầu',
  duration: 12,
  beats: () => {
    const ma = movingAverage(RAW, state.window);
    return [
      { at: 0, text: 'Mô phỏng: góc quay đầu thật (trắng) gồm một chuyển động chậm và một cú quay nhanh quanh frame 140.' },
      { at: 2, text: `Ước lượng pose từng frame từ landmark bị nhiễu (đỏ, σ = ${vn(LANDMARK_SIGMA, 1)}°): đầu render ra sẽ rung liên tục.` },
      { at: 4.5, text: `Làm mượt bằng trung bình trượt ${state.window} frame (vàng) giảm rung nhưng làm tròn cú quay đầu: sai số còn ${vn(rmse(ma), 2)}°.` },
      { at: 7.5, text: `SyncTalk theo dõi điểm đặc trưng bằng optical flow giữa các frame (chuyển động tương đối rất chính xác, σ = ${vn(FLOW_SIGMA, 2)}°) rồi tối ưu pose cùng landmark. Kết hợp hai nguồn (xanh lá): sai số ${vn(rmse(FUSED), 2)}°, giữ được cú quay.` },
      { at: 10, text: 'Kéo cửa sổ trung bình trượt: cửa sổ nhỏ còn rung, cửa sổ lớn xóa chuyển động thật. Kết hợp landmark + flow không phải đánh đổi như vậy.' },
    ];
  },
  formula: 'p* = argmin Σₜ (pₜ − landmarkₜ)² + λ Σₜ (pₜ₊₁ − pₜ − flowₜ)²,   λ = (σ_landmark / σ_flow)²',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const ma = movingAverage(RAW, state.window);
    const [left, right] = split(f.safe, narrow ? [1.3, 1] : [1.8, 1], narrow);
    f.g.panel(left.x, left.y, left.w, left.h, 'Góc quay đầu (độ) theo frame · mô phỏng', C.lime);
    const plot = new Plot(f.g, left, [0, N - 1], [-8, 16], { l: 38, r: 14, t: 40, b: 34 });
    plot.axes({ xTicks: [0, 50, 100, 150, 200, 249], yTicks: [-5, 0, 5, 10, 15], xLabel: 'frame', fx: v => String(Math.round(v)), fy: v => `${v}°` });
    const pts = (a: readonly number[]) => a.map((v, i) => [i, v] as Point);
    plot.clip(() => {
      const reveal = Math.floor(N * seg(t, 2, 4));
      for (let i = 0; i < reveal; i++) f.g.circle(plot.px(i), plot.py(RAW[i]), 1.8, alpha(C.red, 0.7));
      plot.line(pts(YAW), C.white, { width: 2 });
      if (t > 4.5) plot.line(pts(ma).slice(0, Math.floor(N * seg(t, 4.5, 6.5))), C.amber, { width: 2, glow: 6 });
      if (t > 7.5) plot.line(pts(FUSED).slice(0, Math.floor(N * seg(t, 7.5, 9.5))), C.lime, { width: 2.2, glow: 8 });
    });
    barChart(f, right, 'Sai số so với pose thật (độ)', ['landmark', `TB trượt ${state.window}`, 'landmark + flow'], [rmse(RAW), rmse(ma), rmse(FUSED)], { max: 2, reveal: seg(t, 4.5, 9.5), format: v => vn(v, 2), colors: [C.red, C.amber, C.lime], accent: C.lime });
  },
  metrics: () => { const ma = movingAverage(RAW, state.window); return [{ label: 'Rung (TB trượt)', value: `${vn(jitter(ma), 2)}°/frame`, tone: 'amber' }, { label: 'Rung (landmark + flow)', value: `${vn(jitter(FUSED), 2)}°/frame`, tone: 'lime' }]; },
};

// ---------- Chapter 4: tri-plane hash grid ----------
function triplane(f: Frame, rect: Rect, t: number) {
  const { g } = f, s = Math.min(rect.w * 0.3, rect.h * 0.34), ox = rect.x + rect.w * 0.12, oy = rect.y + rect.h * 0.78;
  // oblique projection of the unit cube: x → right, y → up, z → back
  const P = (x: number, y: number, z: number): Point => [ox + x * s + z * s * 0.55, oy - y * s - z * s * 0.4];
  const plane = (pts: Point[], color: string, label: string, lp: Point) => {
    g.polygon(pts, alpha(color, 0.12), alpha(color, 0.7), 1.2);
    for (let i = 1; i < 6; i++) {
      const a = i / 6;
      g.line([[pts[0][0] + (pts[1][0] - pts[0][0]) * a, pts[0][1] + (pts[1][1] - pts[0][1]) * a], [pts[3][0] + (pts[2][0] - pts[3][0]) * a, pts[3][1] + (pts[2][1] - pts[3][1]) * a]], alpha(color, 0.25), { width: 0.8 });
      g.line([[pts[0][0] + (pts[3][0] - pts[0][0]) * a, pts[0][1] + (pts[3][1] - pts[0][1]) * a], [pts[1][0] + (pts[2][0] - pts[1][0]) * a, pts[1][1] + (pts[2][1] - pts[1][1]) * a]], alpha(color, 0.25), { width: 0.8 });
    }
    g.text(label, lp[0], lp[1], { size: 11.5, color, weight: 650 });
  };
  plane([P(0, 0, 1), P(1, 0, 1), P(1, 1, 1), P(0, 1, 1)], C.cyan, 'XY', P(1.02, 1, 1));
  plane([P(0, 0, 0), P(0, 0, 1), P(0, 1, 1), P(0, 1, 0)], C.amber, 'YZ', P(-0.16, 1.05, 0));
  plane([P(0, 0, 0), P(1, 0, 0), P(1, 0, 1), P(0, 0, 1)], C.magenta, 'XZ', P(1.05, -0.08, 0));
  const ang = f.time * 0.6, x = 0.5 + 0.25 * Math.cos(ang), y = 0.55 + 0.2 * Math.sin(ang * 1.3), z = 0.5 + 0.25 * Math.sin(ang);
  const show = seg(t, 2, 4);
  const p = P(x, y, z);
  for (const [q, c] of [[P(x, y, 1), C.cyan], [P(0, y, z), C.amber], [P(x, 0, z), C.magenta]] as [Point, string][]) {
    g.line([p, q], c, { width: 1.2, dash: [4, 4], alpha: show });
    g.circle(q[0], q[1], 4, c, { glow: 10, alpha: show });
  }
  g.circle(p[0], p[1], 6, C.white, { glow: 14 });
  g.text('x = (x, y, z)', p[0] + 9, p[1] - 8, { size: 11.5, color: C.ink });
  // features → MLP → σ, c
  const bx = rect.x + rect.w * 0.68, by = rect.y + rect.h * 0.22, show2 = seg(t, 4.5, 6.5);
  [C.cyan, C.amber, C.magenta].forEach((c, i) => { for (let j = 0; j < 4; j++) f.g.square(bx + j * 13, by + i * 22, 10, alpha(c, 0.25 + 0.6 * ((j * 7 + i * 3) % 5) / 5), { alpha: show2 }); });
  g.text('f_XY ⊕ f_YZ ⊕ f_XZ', bx + 26, by - 12, { size: 11, color: C.muted, align: 'center', alpha: show2 });
  const show3 = seg(t, 6.5, 8.5);
  g.arrow(bx + 26, by + 64, bx + 26, by + 92, C.muted, { width: 1.3, alpha: show3 });
  g.ctx.save(); g.ctx.globalAlpha = show3; g.ctx.strokeStyle = C.violet; g.ctx.fillStyle = alpha(C.violet, 0.15); g.ctx.beginPath(); g.ctx.roundRect(bx - 20, by + 96, 92, 40, 8); g.ctx.fill(); g.ctx.stroke(); g.ctx.restore();
  g.text('MLP + f_l, f_e', bx + 26, by + 116, { size: 11.5, color: C.ink, align: 'center', alpha: show3 });
  g.arrow(bx + 26, by + 138, bx + 26, by + 164, C.muted, { width: 1.3, alpha: show3 });
  g.text('σ (mật độ), c (màu)', bx + 26, by + 178, { size: 12, color: C.lime, align: 'center', weight: 650, alpha: show3 });
}
const chapter4: Scene = {
  title: 'Tri-plane hash: không gian 3D bằng ba mặt phẳng',
  duration: 11,
  idle: true,
  beats: () => {
    const m = memory(state.res);
    return [
      { at: 0, text: 'NeRF cần hỏi “tại điểm 3D này có gì?”. Lưu một lưới 3D đặc trưng ở độ phân giải cao thì bộ nhớ tăng theo lập phương.' },
      { at: 2, text: 'Tri-plane chiếu điểm x lên ba mặt phẳng XY, YZ, XZ, đọc đặc trưng tại ba hình chiếu (nội suy song tuyến tính trên lưới nhiều độ phân giải).' },
      { at: 4.5, text: 'Ba vector đặc trưng được nối lại, cùng đặc trưng môi f_l và biểu cảm f_e đi vào một MLP nhỏ để ra mật độ σ và màu c.' },
      { at: 7.5, text: `Ở độ phân giải ${state.res}: lưới 3D nhiều mức cần ${fmtMB(MB(m.dense3d))}, ba mặt phẳng chỉ ${fmtMB(MB(m.triplane))}; bảng băm 2D (giới hạn 2¹⁹ ô mỗi mức) ${fmtMB(MB(m.triHash))}. Mặt người gần như là một bề mặt, nên ba hình chiếu 2D đủ chi tiết.` },
    ];
  },
  formula: 'f(x) = H_XY(x, y) ⊕ H_YZ(y, z) ⊕ H_XZ(x, z),   (σ, c) = MLP(f(x), d, f_l, f_e)',
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.2, 1] : [1.5, 1], narrow);
    f.g.panel(left.x, left.y, left.w, left.h, 'Tra cứu một điểm 3D', C.cyan);
    triplane(f, left, t);
    const m = memory(state.res);
    barChart(f, right, `Bộ nhớ đặc trưng · độ phân giải ${state.res} (log₁₀ MB)`, ['lưới 3D', 'băm 3D', 'tri-plane', 'tri-plane băm'], [m.dense3d, m.hash3d, m.triplane, m.triHash].map(b => Math.max(0, Math.log10(MB(b)) + 1)), { max: 5.5, reveal: seg(t, 7.5, 9.5), format: v => fmtMB(10 ** (v - 1)), colors: [C.red, C.amber, C.cyan, C.lime], accent: C.lime });
  },
  metrics: () => { const m = memory(state.res); return [{ label: 'Lưới 3D', value: fmtMB(MB(m.dense3d)), tone: 'red' }, { label: 'Tri-plane băm', value: fmtMB(MB(m.triHash)), tone: 'lime' }]; },
  table: () => ({ headers: ['Độ phân giải', 'Lưới 3D', 'Băm 3D', 'Tri-plane', 'Tri-plane băm'], rows: [64, 128, 256, 512, 1024, 2048].map(r => { const m = memory(r); return [r, fmtMB(MB(m.dense3d)), fmtMB(MB(m.hash3d)), fmtMB(MB(m.triplane)), fmtMB(MB(m.triHash))]; }), caption: '16 mức, 2 đặc trưng float16 mỗi ô, bảng băm tối đa 2¹⁹ ô mỗi mức (cấu hình kiểu Instant-NGP).' }),
};

// ---------- Chapter 5: render a ray, blend into the frame ----------
const chapter5: Scene = {
  title: 'Render một tia và ghép đầu vào khung hình',
  duration: 12,
  beats: () => {
    const peak = RAY.reduce((a, b) => (b.w > a.w ? b : a));
    return [
      { at: 0, text: 'Mô phỏng một tia đi qua mặt: mật độ σ (tím) gần 0 trong không khí, vọt lên ở bề mặt da.' },
      { at: 2.5, text: `Độ trong suốt T (xanh) giảm về 0 ngay sau bề mặt; trọng số w = T·α (vàng) dồn vào vài mẫu quanh da (mẫu lớn nhất w = ${vn(peak.w, 2)}). Màu pixel là tổng màu có trọng số.` },
      { at: 5.5, text: 'NeRF render đầu rất tốt nhưng tóc, cổ áo và nền thật khó tái tạo sắc nét. Portrait-Sync Generator ghép mặt render vào khung hình gốc.' },
      { at: 8, text: `Ghép bằng mặt nạ cứng tạo đường nối (bước nhảy ${vn(seam(composite(mask(0))), 3)}); làm mờ mặt nạ (Gaussian blur) xóa đường nối: ${vn(seam(composite(mask(state.blur))), 3)}. Kéo độ mờ để thử. Khe tối giữa đầu và thân được tô bằng màu cổ trung bình.` },
    ];
  },
  formula: () => `C(r) = Σᵢ Tᵢ αᵢ cᵢ,   αᵢ = 1 − e^(−σᵢ δᵢ),   Tᵢ = Πⱼ<ᵢ (1 − αⱼ)\nI = M_blur · F_render + (1 − M_blur) · F_gốc,   bước nhảy lớn nhất = ${vn(seam(composite(mask(state.blur))), 3)}`,
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, [1, 1], narrow);
    f.g.panel(left.x, left.y, left.w, left.h, 'Một tia qua khuôn mặt · mô phỏng', C.violet);
    const plot = new Plot(f.g, left, [0.2, 1.2], [0, 1.05], { l: 34, r: 14, t: 40, b: 34 });
    plot.axes({ xTicks: [0.2, 0.4, 0.6, 0.8, 1, 1.2], yTicks: [0, 0.5, 1], xLabel: 'khoảng cách dọc tia t', fx: v => vn(v, 1), fy: v => vn(v, 1) });
    const maxS = Math.max(...RAY.map(s => s.sigma)), n = Math.floor(RAY.length * seg(t, 0, 2.5));
    plot.line(RAY.slice(0, n).map(s => [s.t, s.sigma / maxS] as Point), C.violet, { width: 2, glow: 6 });
    if (t > 2.5) {
      const m = Math.floor(RAY.length * seg(t, 2.5, 5));
      plot.line(RAY.slice(0, m).map(s => [s.t, s.T] as Point), C.cyan, { width: 2, glow: 6 });
      RAY.slice(0, m).forEach(s => { if (s.w > 0.005) f.g.line([[plot.px(s.t), plot.py(0)], [plot.px(s.t), plot.py(s.w)]], C.amber, { width: 4, glow: 6 }); });
    }
    f.g.text('σ / max', plot.box.x + plot.box.w - 6, plot.box.y + 12, { size: 11, align: 'right', color: C.violet });
    f.g.text('T', plot.box.x + plot.box.w - 6, plot.box.y + 27, { size: 11, align: 'right', color: C.cyan });
    f.g.text('w = T·α', plot.box.x + plot.box.w - 6, plot.box.y + 42, { size: 11, align: 'right', color: C.amber });
    f.g.panel(right.x, right.y, right.w, right.h, 'Lát cắt qua đường ghép cằm → cổ · mô phỏng', C.amber);
    const p2 = new Plot(f.g, right, [0, 1], [0.4, 1.05], { l: 34, r: 14, t: 40, b: 34 });
    p2.axes({ xTicks: [0, 0.5, 1], yTicks: [0.5, 0.75, 1], xLabel: 'vị trí ←mặt render | khung gốc→', fx: v => vn(v, 1), fy: v => vn(v, 2) });
    const show = seg(t, 5.5, 7.5), blur = t < 8 ? 0 : state.blur;
    const mk = mask(blur), comp = composite(mk);
    p2.line(XS.map((x, i) => [x, mk[i] * 0.55 + 0.45] as Point), C.muted, { width: 1.2, dash: [4, 4], alpha: show });
    p2.line(XS.map((x, i) => [x, comp[i]] as Point), C.lime, { width: 2.6, glow: 10, alpha: show });
    f.g.text(blur ? `mặt nạ mờ (${vn(blur, 2)})` : 'mặt nạ cứng', p2.box.x + 6, p2.box.y + 12, { size: 11, color: C.muted, alpha: show });
    f.g.text('ảnh ghép', p2.box.x + 6, p2.box.y + 27, { size: 11, color: C.lime, alpha: show });
  },
  metrics: () => [{ label: 'T cuối tia', value: vn(RAY.at(-1)!.T, 4), tone: 'cyan' }, { label: 'Bước nhảy ở đường ghép', value: vn(seam(composite(mask(state.blur))), 3), tone: 'lime' }],
};

export const initSyncTalkPipelineLab = () => glassLab('synctalk-pipeline', [chapter1, chapter2, chapter3, chapter4, chapter5], lab => {
  lab.range('lag', v => `${v} frame (${v * 40} ms)`, v => { state.lag = v; lab.invalidate(); });
  lab.range('window', v => `${v} frame`, v => { state.window = v; lab.invalidate(); });
  lab.range('res', v => String(2 ** v), v => { state.res = 2 ** v; lab.invalidate(); });
  lab.range('blur', v => vn(v, 2), v => { state.blur = v; lab.invalidate(); });
});
