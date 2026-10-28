// PCA lab: centre and scale real data, sweep a direction and watch the variance of the projections, find that the
// best direction is the eigenvector of the covariance matrix (S u = λ u), project onto it, then run PCA on 64-pixel
// handwritten digits: their 2-D map, the "eigen-digits" and reconstructions from k components.
import { glassLab, type Frame, type Scene } from '../../_shared/glass-lab/engine';
import { C, Plot, alpha, fitEqual, lerp, seg, split, vn, type Point, type Rect } from '../../_shared/glass-lab/gfx';
import { categoryChart } from '../../_shared/glass-lab/views';
import { COV, DIGITS, EIG, MPG, NAMES, PC1_ANGLE, RAW, STD, cumulative, dot, mse, reconstruct, times, unit, varianceAlong, type Vec } from './model';

export const explainerKind = 'educational-simulation';

const DIGIT_COLORS = [C.cyan, C.amber, C.magenta, C.lime, C.violet, C.red, '#ffffff', '#5b8cff', '#ff9e6b', '#c8f06b'];
const state = { angle: null as number | null, k: null as number | null, sample: 3 };
let currentT = 0;
const PAD = { l: 44, r: 16, t: 40, b: 34 };

function stdPlot(f: Frame, rect: Rect, title: string, accent: string = C.violet, extent = 3.2) {
  f.g.panel(rect.x, rect.y, rect.w, rect.h, title, accent);
  const [x, y] = fitEqual(rect, PAD, [-extent, extent], [-extent * 0.8, extent * 0.8]);
  const plot = new Plot(f.g, rect, x, y, PAD);
  plot.axes({ xTicks: [-2, -1, 0, 1, 2], yTicks: [-2, -1, 0, 1, 2], xLabel: 'khối lượng (chuẩn hóa)', yLabel: 'mã lực (chuẩn hóa)', fx: v => vn(v, 0), fy: v => vn(v, 0) });
  return plot;
}
function scatter(plot: Plot, points: Vec[], color: string = C.cyan, r = 2.6, a = 0.85) {
  plot.clip(() => points.forEach(([x, y]) => plot.g.circle(plot.px(x), plot.py(y), r, color, { glow: 5, alpha: a })));
}
function axisLine(plot: Plot, u: Vec, color: string, width = 2.4, label?: string) {
  const L = 6;
  plot.clip(() => plot.line([[-u[0] * L, -u[1] * L], [u[0] * L, u[1] * L]], color, { width, glow: 12 }));
  if (label) plot.g.tag(label, plot.px(u[0] * 2.3), plot.py(u[1] * 2.3) - 14, color);
}
/** Ellipse of the covariance S at `k` standard deviations. */
function ellipse(plot: Plot, k: number, color: string, width = 1.6) {
  const pts: Point[] = [];
  for (let i = 0; i <= 96; i++) {
    const a = i / 96 * Math.PI * 2, c = Math.cos(a) * Math.sqrt(EIG.values[0]) * k, s = Math.sin(a) * Math.sqrt(EIG.values[1]) * k;
    const [u, v] = EIG.vectors;
    pts.push([c * u[0] + s * v[0], c * u[1] + s * v[1]]);
  }
  plot.clip(() => plot.line(pts, color, { width, glow: 8 }));
}

// ---------- Chapter 1: centring and scaling ----------
const chapter1: Scene = {
  title: 'Đưa dữ liệu về tâm và cùng thang đo',
  duration: 10,
  beats: () => [
    { at: 0, text: `392 chiếc xe thật (Auto MPG): khối lượng từ ${vn(Math.min(...RAW.map(p => p[0])), 1)} tới ${vn(Math.max(...RAW.map(p => p[0])), 1)} nghìn pound, công suất từ ${Math.min(...RAW.map(p => p[1]))} tới ${Math.max(...RAW.map(p => p[1]))} mã lực.` },
    { at: 2.5, text: 'Hai cột có đơn vị khác nhau. Nếu để nguyên, cột có số lớn (mã lực) sẽ lấn át mọi phương sai.' },
    { at: 4.5, text: 'Trừ trung bình để đám mây về tâm (0, 0), rồi chia độ lệch chuẩn để mỗi trục có phương sai 1.' },
    { at: 7.5, text: `Ma trận hiệp phương sai sau chuẩn hóa: đường chéo bằng 1, ngoài đường chéo ${vn(COV[0][1], 3)}. Hai biến tương quan rất mạnh.` },
  ],
  formula: `zᵢ = (xᵢ − μ) / σ,  μ = (${vn(MPG.mean[0], 2)}; ${vn(MPG.mean[1], 1)}),  σ = (${vn(MPG.std[0], 2)}; ${vn(MPG.std[1], 1)})\nS = (1/n) Σ zᵢzᵢᵀ = [[${vn(COV[0][0], 3)}; ${vn(COV[0][1], 3)}], [${vn(COV[1][0], 3)}; ${vn(COV[1][1], 3)}]]`,
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.4, 1] : [1.5, 1], narrow);
    const c = seg(t, 4.5, 6), s = seg(t, 6, 7.5);
    // Interpolate raw → centred → scaled, expressed in standard units so all three fit one view.
    const pts: Vec[] = RAW.map(([w, h], i) => {
      const raw: Vec = [(w - 1.5) / 0.85, (h - 40) / 40];
      const centred: Vec = [(w - MPG.mean[0]) / 0.85, (h - MPG.mean[1]) / 40];
      const z = STD[i];
      const a: Vec = [lerp(raw[0], centred[0], c), lerp(raw[1], centred[1], c)];
      return [lerp(a[0], z[0], s), lerp(a[1], z[1], s)];
    });
    g.panel(left.x, left.y, left.w, left.h, s > 0.98 ? 'Sau chuẩn hóa: tâm (0, 0), phương sai 1 mỗi trục' : c > 0.98 ? 'Đã trừ trung bình' : 'Dữ liệu gốc', C.violet);
    const [x, y] = fitEqual(left, PAD, [-3.5, 4.5], [-2.8, 4.2]);
    const plot = new Plot(g, left, x, y, PAD);
    plot.axes({ xTicks: [-2, 0, 2, 4], yTicks: [-2, 0, 2, 4], fx: v => vn(v, 0), fy: v => vn(v, 0) });
    scatter(plot, pts);
    const m = pts.reduce((acc, p) => [acc[0] + p[0] / pts.length, acc[1] + p[1] / pts.length], [0, 0]);
    g.circle(plot.px(m[0]), plot.py(m[1]), 6, C.amber, { glow: 14 });
    g.text('trung bình', plot.px(m[0]) + 10, plot.py(m[1]) - 10, { size: 11.5, color: C.amber });
    // Right: the covariance matrix as a 2×2 heat table.
    g.panel(right.x, right.y, right.w, right.h, 'Ma trận hiệp phương sai S', C.amber);
    const cell = Math.min((right.w - 60) / 2, (right.h - 110) / 2), ox = right.x + (right.w - cell * 2) / 2, oy = right.y + 60;
    const labels = ['khối lượng', 'mã lực'];
    const reveal = seg(t, 7.5, 8.5);
    COV.forEach((row, i) => row.forEach((v, j) => {
      g.ctx.fillStyle = alpha(i === j ? C.cyan : C.amber, 0.12 + 0.5 * v * reveal);
      g.ctx.fillRect(ox + j * cell + 3, oy + i * cell + 3, cell - 6, cell - 6);
      g.text(reveal > 0 ? vn(v, 3) : '?', ox + j * cell + cell / 2, oy + i * cell + cell / 2, { size: Math.min(26, cell / 3), color: C.ink, align: 'center', weight: 650, mono: true });
    }));
    labels.forEach((l, i) => { g.text(l, ox + i * cell + cell / 2, oy - 12, { size: 11.5, color: C.muted, align: 'center' }); g.text(l, ox - 8, oy + i * cell + cell / 2, { size: 11.5, color: C.muted, align: 'right' }); });
  },
  metrics: () => [{ label: 'Số xe', value: String(RAW.length), tone: 'cyan' }, { label: 'Tương quan', value: vn(COV[0][1], 3), tone: 'amber' }],
  table: () => ({ headers: ['Xe', 'Khối lượng (nghìn lb)', 'Mã lực', 'z khối lượng', 'z mã lực'], rows: RAW.slice(0, 15).map((p, i) => [NAMES[i], vn(p[0], 3), p[1], vn(STD[i][0], 2), vn(STD[i][1], 2)]), caption: '15 xe đầu tiên trong 392.' }),
};

// ---------- Chapter 2: sweep a direction ----------
const ANGLES = Array.from({ length: 73 }, (_, i) => -90 + i * 2.5);
const VARS = ANGLES.map(a => varianceAlong(unit(a)));
const angleNow = () => state.angle ?? (currentT < 8 ? lerp(-90, 90, seg(currentT, 0.5, 8)) : lerp(90, PC1_ANGLE, seg(currentT, 8, 10)));
const chapter2: Scene = {
  title: 'Xoay một trục, đo phương sai hình chiếu',
  duration: 13,
  beats: () => [
    { at: 0, text: 'Chọn một hướng u (vector đơn vị). Chiếu từng chiếc xe vuông góc lên đường thẳng theo hướng đó.' },
    { at: 2.5, text: 'Các điểm cam là hình chiếu. Chúng trải rộng hay co cụm tùy hướng: đó là phương sai Var(uᵀz) = uᵀSu.' },
    { at: 6, text: 'Biểu đồ bên phải ghi lại phương sai theo mọi góc. Có đúng một đỉnh.' },
    { at: 9.5, text: `Đỉnh nằm ở ${vn(PC1_ANGLE, 0)}°, phương sai ${vn(EIG.values[0], 3)}. Hướng giữ lại nhiều thông tin nhất này là thành phần chính thứ nhất (PC1).` },
  ],
  formula: () => { const u = unit(angleNow()); return `u = (${vn(u[0], 3)}; ${vn(u[1], 3)}),  Var(uᵀz) = uᵀSu = ${vn(varianceAlong(u), 3)}\nPC1 = argmax‖u‖=1 uᵀSu`; },
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.4, 1] : [1.45, 1], narrow);
    const angle = angleNow(), u = unit(angle);
    const plot = stdPlot(f, left, `Hướng u = ${vn(angle, 0)}°`);
    scatter(plot, STD, C.cyan, 2.4, 0.55);
    axisLine(plot, u, C.amber, 2.2);
    plot.clip(() => STD.forEach((p, i) => {
      const s = dot(p, u), q: Vec = [s * u[0], s * u[1]];
      if (i % 3 === 0) plot.line([p, q], alpha(C.white, 0.18), { width: 0.8 });
      g.circle(plot.px(q[0]), plot.py(q[1]), 2.4, C.amber, { glow: 6 });
    }));
    const max = Math.max(...VARS);
    categoryChart(f, right, 'Phương sai hình chiếu theo góc', ANGLES.map(a => (a % 45 === 0 ? `${a}°` : '')), [{ label: 'uᵀSu', values: VARS, color: C.amber, format: v => vn(v, 2) }], [0, 2], Math.round((angle + 90) / 2.5), state.angle === null && t < 8 ? seg(t, 0.5, 8) : 1, v => vn(v, 1));
    if (t > 9.5 || state.angle !== null) g.tag(`max ${vn(max, 3)} = λ₁`, right.x + right.w - 14, right.y + 50, C.lime, 'right');
  },
  metrics: () => { const u = unit(angleNow()); return [{ label: 'Góc', value: `${vn(angleNow(), 0)}°`, tone: 'violet' }, { label: 'Phương sai', value: vn(varianceAlong(u), 3), tone: 'amber' }, { label: 'Tối đa (λ₁)', value: vn(EIG.values[0], 3), tone: 'lime' }]; },
  table: () => ({ headers: ['Góc', 'uᵀSu'], rows: ANGLES.filter((_, i) => i % 3 === 0).map((a, i) => [`${vn(a, 1)}°`, vn(VARS[i * 3], 3)]) }),
};

// ---------- Chapter 3: eigenvectors of S ----------
const eigAngleNow = () => state.angle ?? lerp(-80, 100, seg(currentT, 1, 11));
const chapter3: Scene = {
  title: 'Vector riêng: hướng mà S không xoay',
  duration: 14,
  beats: () => [
    { at: 0, text: 'Nhân ma trận S với một vector u: kết quả Su thường lệch hướng so với u.' },
    { at: 3, text: 'S biến vòng tròn đơn vị thành một elip. Theo dõi mũi tên trắng (u) và mũi tên cam (Su) khi u quay.' },
    { at: 6.5, text: 'Ở hai hướng đặc biệt, Su nằm đúng trên đường của u: Su = λu. Đó là các vector riêng.' },
    { at: 10, text: `Bài toán tối đa uᵀSu dẫn đúng tới vector riêng có giá trị riêng lớn nhất: λ₁ = ${vn(EIG.values[0], 3)}, λ₂ = ${vn(EIG.values[1], 3)}. Elip trên dữ liệu có trục chính là PC1.` },
  ],
  formula: () => { const u = unit(eigAngleNow()), su = times(COV, u), cos = dot(su, u) / Math.hypot(...su); return `S u = λ u\nu = (${vn(u[0], 2)}; ${vn(u[1], 2)}),  Su = (${vn(su[0], 2)}; ${vn(su[1], 2)}),  góc lệch = ${vn(Math.acos(Math.min(1, Math.abs(cos))) * 180 / Math.PI, 1)}°`; },
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1, 1] : [1, 1.15], narrow);
    // Left: circle → ellipse under S, with u and Su.
    g.panel(left.x, left.y, left.w, left.h, 'Ma trận S tác động lên vector đơn vị', C.amber);
    const [x, y] = fitEqual(left, { l: 20, r: 20, t: 40, b: 20 }, [-2.1, 2.1], [-2.1, 2.1]);
    const p = new Plot(g, left, x, y, { l: 20, r: 20, t: 40, b: 20 });
    p.line([[-2, 0], [2, 0]], alpha(C.white, 0.25)); p.line([[0, -2], [0, 2]], alpha(C.white, 0.25));
    const circle: Point[] = [], ell: Point[] = [];
    for (let i = 0; i <= 96; i++) { const v = unit(i / 96 * 360); circle.push(v); ell.push(times(COV, v)); }
    p.line(circle, alpha(C.white, 0.35), { width: 1.2, dash: [4, 4] });
    p.line(ell, alpha(C.amber, 0.6), { width: 1.6 });
    const u = unit(eigAngleNow()), su = times(COV, u);
    const aligned = Math.abs(dot(su, u) / Math.hypot(...su)) > 0.999;
    EIG.vectors.forEach((v, i) => { if (t > 6.5) p.line([[-v[0] * 2, -v[1] * 2], [v[0] * 2, v[1] * 2]], alpha(i ? C.magenta : C.lime, 0.5), { width: 1.2, dash: [6, 5] }); });
    g.arrow(p.px(0), p.py(0), p.px(u[0]), p.py(u[1]), C.white, { width: 2.4, glow: 10 });
    g.arrow(p.px(0), p.py(0), p.px(su[0]), p.py(su[1]), aligned ? C.lime : C.amber, { width: 2.6, glow: 14 });
    g.text('u', p.px(u[0] * 1.12), p.py(u[1] * 1.12), { size: 13, color: C.white, align: 'center', weight: 650 });
    g.text('Su', p.px(su[0] * 1.08) + 8, p.py(su[1] * 1.08), { size: 13, color: aligned ? C.lime : C.amber, weight: 650 });
    if (aligned) g.tag('Su = λu', left.x + left.w - 12, left.y + left.h - 16, C.lime, 'right');
    // Right: data with covariance ellipses and both principal axes.
    const plot = stdPlot(f, right, 'Trục chính trên dữ liệu thật', C.violet);
    scatter(plot, STD, C.cyan, 2.4, 0.6);
    const r = seg(t, 9.5, 11.5);
    if (r > 0) {
      ellipse(plot, 1, alpha(C.white, 0.5 * r)); ellipse(plot, 2, alpha(C.white, 0.35 * r));
      axisLine(plot, EIG.vectors[0], alpha(C.lime, r), 2.6, `PC1 · λ₁ = ${vn(EIG.values[0], 2)}`);
      axisLine(plot, EIG.vectors[1], alpha(C.magenta, r), 1.8, `PC2 · λ₂ = ${vn(EIG.values[1], 2)}`);
    }
  },
  metrics: () => [{ label: 'λ₁', value: vn(EIG.values[0], 3), tone: 'lime' }, { label: 'λ₂', value: vn(EIG.values[1], 3), tone: 'magenta' }, { label: 'PC1', value: `(${vn(EIG.vectors[0][0], 3)}; ${vn(EIG.vectors[0][1], 3)})`, tone: 'violet' }],
  table: () => ({ headers: ['', 'Giá trị riêng', 'Vector riêng', 'Tỷ lệ phương sai', 'sklearn'], rows: [0, 1].map(i => [`PC${i + 1}`, vn(EIG.values[i], 4), `(${vn(EIG.vectors[i][0], 4)}; ${vn(EIG.vectors[i][1], 4)})`, `${vn(EIG.values[i] / (EIG.values[0] + EIG.values[1]) * 100, 1)}%`, vn(MPG.eigenvalues[i], 4)]) }),
};

// ---------- Chapter 4: projection = dimensionality reduction ----------
const chapter4: Scene = {
  title: 'Giảm chiều: chiếu lên PC1',
  duration: 11,
  beats: () => [
    { at: 0, text: 'Giữ một con số cho mỗi xe thay vì hai: tọa độ của nó trên PC1.' },
    { at: 2, text: 'Mọi điểm trượt vuông góc xuống trục PC1. Đoạn trượt chính là phần thông tin bị bỏ đi.' },
    { at: 6, text: `PC1 giữ ${vn(MPG.explained[0] * 100, 1)}% tổng phương sai; PC2 chỉ ${vn(MPG.explained[1] * 100, 1)}%. Một chiều đã mô tả gần hết hai chiều.` },
    { at: 8.5, text: 'PC1 ≈ (khối lượng + mã lực)/√2 sau chuẩn hóa: một “chỉ số độ lớn” của xe. Đó là ý nghĩa của thành phần chính.' },
  ],
  formula: `tᵢ = PC1ᵀ zᵢ,  ẑᵢ = tᵢ · PC1\nphương sai giữ lại = λ₁ / (λ₁ + λ₂) = ${vn(MPG.explained[0] * 100, 1)}%`,
  render(f) {
    const { t, narrow } = f; currentT = t;
    const [left, right] = split(f.safe, narrow ? [1.4, 1] : [1.5, 1], narrow);
    const k = seg(t, 2, 5.5), pc = EIG.vectors[0];
    const plot = stdPlot(f, left, 'Chiếu từng xe lên PC1', C.violet);
    axisLine(plot, pc, C.lime, 2.4, 'PC1');
    plot.clip(() => STD.forEach(p => {
      const s = dot(p, pc), q: Vec = [s * pc[0], s * pc[1]], m: Vec = [lerp(p[0], q[0], k), lerp(p[1], q[1], k)];
      if (k > 0.02 && k < 0.98) plot.line([p, m], alpha(C.red, 0.35), { width: 0.8 });
      plot.g.circle(plot.px(m[0]), plot.py(m[1]), 2.4, k > 0.98 ? C.lime : C.cyan, { glow: 5 });
    }));
    categoryChart(f, right, 'Tỷ lệ phương sai giữ lại', ['PC1', 'PC2'], [{ label: 'phương sai', values: MPG.explained, color: C.lime, format: v => `${vn(v * 100, 1)}%` }], [0, 1], 0, seg(t, 6, 7), v => `${vn(v * 100, 0)}%`);
  },
  metrics: () => [{ label: 'PC1 giữ', value: `${vn(MPG.explained[0] * 100, 1)}%`, tone: 'lime' }, { label: 'Bỏ đi', value: `${vn(MPG.explained[1] * 100, 1)}%`, tone: 'red' }],
};

// ---------- Chapter 5: digits, 64 dimensions ----------
const kNow = () => state.k ?? Math.max(1, Math.round(lerp(1, 40, seg(currentT, 5, 13))));
function image(f: Frame, x: number, y: number, size: number, pixels: number[], signed = false, label?: string) {
  const { g } = f, c = size / 8;
  const max = signed ? Math.max(...pixels.map(Math.abs)) : 16;
  pixels.forEach((v, i) => {
    const px = x + (i % 8) * c, py = y + Math.floor(i / 8) * c;
    if (signed) { const k = Math.min(1, Math.abs(v) / max); g.ctx.fillStyle = v >= 0 ? `rgba(255,182,72,${k})` : `rgba(79,214,255,${k})`; }
    else { const k = Math.max(0, Math.min(1, v / max)); g.ctx.fillStyle = `rgb(${Math.round(10 + 235 * k)},${Math.round(14 + 236 * k)},${Math.round(30 + 225 * k)})`; }
    g.ctx.fillRect(px, py, c + 0.5, c + 0.5);
  });
  g.ctx.strokeStyle = 'rgba(255,255,255,0.15)'; g.ctx.strokeRect(x, y, size, size);
  if (label) g.text(label, x + size / 2, y + size + 13, { size: 10.5, color: C.muted, align: 'center' });
}
const chapter5: Scene = {
  title: '64 chiều: ảnh chữ số viết tay',
  duration: 15,
  idle: false,
  beats: () => [
    { at: 0, text: `1 797 ảnh chữ số 8×8: mỗi ảnh là một điểm trong không gian 64 chiều. Chiếu lên PC1 và PC2 (giữ ${vn((DIGITS.explained[0] + DIGITS.explained[1]) * 100, 1)}% phương sai), các chữ số đã tự tụ thành cụm.` },
    { at: 3, text: 'Mỗi thành phần chính cũng là một “ảnh” 64 điểm: vàng là cộng thêm sáng, xanh là trừ bớt. Đó là các eigen-digit.' },
    { at: 5, text: 'Một ảnh = ảnh trung bình + c₁·PC1 + c₂·PC2 + … Thêm dần thành phần, ảnh tái tạo nét dần.' },
    { at: 11, text: `${DIGITS.k95} thành phần giữ 95% phương sai: nén 64 con số xuống ${DIGITS.k95} mà ảnh vẫn đọc được.` },
  ],
  formula: () => { const k = kNow(); return `x ≈ μ + Σⱼ₌₁ᵏ cⱼ · PCⱼ,  cⱼ = PCⱼᵀ(x − μ)\nk = ${k}: giữ ${vn(cumulative[k - 1] * 100, 1)}% phương sai`; },
  render(f) {
    const { g, t, narrow } = f; currentT = t;
    const [left, rightAll] = split(f.safe, narrow ? [1, 1.2] : [1.05, 1], narrow);
    // Left: 2-D map of all digits.
    g.panel(left.x, left.y, left.w, left.h, 'Mỗi ảnh là một điểm: PC1 × PC2', C.violet);
    const xs = DIGITS.coords.map(c => c[0]), ys = DIGITS.coords.map(c => c[1]);
    const [xr, yr] = fitEqual(left, { l: 16, r: 16, t: 40, b: 30 }, [Math.min(...xs), Math.max(...xs)], [Math.min(...ys), Math.max(...ys)]);
    const map = new Plot(g, left, xr, yr, { l: 16, r: 16, t: 40, b: 30 });
    const show = seg(t, 0, 2.5);
    map.clip(() => DIGITS.coords.forEach(([a, b], i) => { if (i / DIGITS.coords.length > show) return; g.circle(map.px(a), map.py(b), 1.9, DIGIT_COLORS[DIGITS.labels[i]], { alpha: 0.8 }); }));
    for (let d = 0; d < 10; d++) {
      const idx = DIGITS.labels.map((l, i) => (l === d ? i : -1)).filter(i => i >= 0);
      const cx = idx.reduce((s, i) => s + DIGITS.coords[i][0], 0) / idx.length, cy = idx.reduce((s, i) => s + DIGITS.coords[i][1], 0) / idx.length;
      if (show > 0.9) g.text(String(d), map.px(cx), map.py(cy), { size: 17, color: C.white, align: 'center', weight: 750, shadow: true });
    }
    const sample = DIGITS.samples[state.sample], c = DIGITS.coords[sample.index];
    g.circle(map.px(c[0]), map.py(c[1]), 7, null, { stroke: C.white, width: 2 });
    // Right top: mean + eigen-digits; right bottom: reconstruction.
    const [top, bottom] = split(rightAll, [1, 1.25], true, 12);
    g.panel(top.x, top.y, top.w, top.h, 'Ảnh trung bình và 7 eigen-digit đầu', C.amber);
    const n = 8, size = Math.min((top.w - 24 - (n - 1) * 8) / n, top.h - 70);
    const ox = top.x + (top.w - (n * size + (n - 1) * 8)) / 2, oy = top.y + 44;
    image(f, ox, oy, size, DIGITS.mean, false, 'μ');
    for (let j = 0; j < 7; j++) if (t > 3 + j * 0.2) image(f, ox + (j + 1) * (size + 8), oy, size, DIGITS.components[j], true, `PC${j + 1} · ${vn(DIGITS.explained[j] * 100, 1)}%`);
    g.panel(bottom.x, bottom.y, bottom.w, bottom.h, `Tái tạo chữ số ${sample.label} từ k = ${kNow()} thành phần`, C.lime);
    const k = kNow(), rec = reconstruct(sample.coef, k);
    const big = Math.min(bottom.h - 70, (bottom.w - 60) / 3.2);
    const bx = bottom.x + 20, by = bottom.y + 46;
    image(f, bx, by, big, sample.pixels, false, 'ảnh gốc (64 số)');
    image(f, bx + big + 20, by, big, rec, false, `tái tạo (${k} số)`);
    // cumulative explained variance bar
    const cx0 = bx + 2 * big + 50, cw = bottom.x + bottom.w - 20 - cx0, ch = big;
    g.text('phương sai tích lũy', cx0, by - 6, { size: 11, color: C.muted });
    g.line([[cx0, by + ch], [cx0 + cw, by + ch]], C.faint);
    g.line(cumulative.map((v, i) => [cx0 + i / 63 * cw, by + ch - v * ch] as Point), C.lime, { width: 2, glow: 8 });
    g.circle(cx0 + (k - 1) / 63 * cw, by + ch - cumulative[k - 1] * ch, 5, C.amber, { glow: 12 });
    g.text(`${vn(cumulative[k - 1] * 100, 1)}%`, cx0 + (k - 1) / 63 * cw + 8, by + ch - cumulative[k - 1] * ch - 10, { size: 12, color: C.amber, weight: 650 });
    g.text(`sai số MSE = ${vn(mse(rec, sample.pixels), 2)}`, cx0, by + ch + 18, { size: 11.5, color: C.ink });
  },
  metrics: () => { const k = kNow(); return [{ label: 'Số chiều gốc', value: '64', tone: 'violet' }, { label: 'Giữ k', value: String(k), tone: 'amber' }, { label: 'Phương sai giữ lại', value: `${vn(cumulative[k - 1] * 100, 1)}%`, tone: 'lime' }]; },
  table: () => ({ headers: ['Thành phần', 'Tỷ lệ phương sai', 'Tích lũy'], rows: DIGITS.explained.slice(0, 30).map((v, i) => [`PC${i + 1}`, `${vn(v * 100, 2)}%`, `${vn(cumulative[i] * 100, 1)}%`]) }),
};

export const initPcaLab = () => glassLab('pca', [chapter1, chapter2, chapter3, chapter4, chapter5], lab => {
  lab.range('angle', value => `${value}°`, value => { state.angle = value; });
  lab.range('k', value => String(value), value => { state.k = value; });
  lab.on('[data-sample]', 'click', element => { state.sample = Number(element.dataset.sample); lab.invalidate(); });
  lab.on('[data-autoplay]', 'click', () => { state.angle = null; state.k = null; lab.replay(); });
});
