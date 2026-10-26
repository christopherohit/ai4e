// SVM lab view: many separating lines → the maximum-margin one and its support vectors → soft margin and C → RBF
// kernel on a ring. Every model is trained in the browser with the SMO solver in model.ts.
import { $, $$, boot, el, fillTable, fmt, lineChart, onResize, percent, setCard, slider, wireSteps, type Root, type StepText } from '../../_shared/sequence-lab/ui';
import { field, frame, marker, polyline, type Frame } from '../../_shared/optimization-lab/plots';
import { C_GRID, GAMMA_GRID, HARD_C, OVERLAP, RINGS, RINGS_TEST, SEPARABLE, corridor, decision, errors, marginWidth, roles, supportCount, train, violations, weights, type Model, type Point, type Sample } from './model';

export const explainerKind = 'educational-simulation';

const texts: StepText[] = [
  ['Có vô số đường phân tách', 'Hai lớp tách được bằng một đường thẳng, nhưng có rất nhiều đường như vậy. Xoay đường và xem “hành lang” trống giữa hai lớp rộng hay hẹp. Đường sát một lớp dễ phân loại sai điểm mới; SVM chọn đường có hành lang rộng nhất.', 'lề = khoảng cách từ đường tới điểm gần nhất của mỗi lớp'],
  ['Lề cực đại và support vector', 'SVM tìm w, b sao cho mọi điểm thỏa y·(w·x + b) ≥ 1 và ‖w‖ nhỏ nhất, tức lề 2/‖w‖ lớn nhất. Chỉ vài điểm nằm đúng trên mép lề quyết định lời giải: đó là support vector (vòng tròn). Nhấp một điểm để bỏ nó khỏi dữ liệu.', 'min ½‖w‖²  sao cho  yᵢ(w·xᵢ + b) ≥ 1  ∀i'],
  ['Lề mềm: đánh đổi bằng C', 'Khi hai lớp chồng lấn, không có lề cứng. Lề mềm cho phép điểm lấn vào lề hoặc sang nhầm phía, mỗi vi phạm bị phạt theo hinge loss nhân C. C nhỏ: lề rộng, chấp nhận nhiều vi phạm. C lớn: lề hẹp, cố phân loại đúng từng điểm.', 'min ½‖w‖² + C Σ max(0, 1 − yᵢ(w·xᵢ + b))'],
  ['Kernel: ranh giới phi tuyến', 'Một vòng bao quanh lõi: không đường thẳng nào tách được. Kernel RBF đo độ giống nhau K(x, x′) = exp(−γ‖x − x′‖²), tương đương ánh xạ dữ liệu lên không gian nhiều chiều rồi tách bằng siêu phẳng ở đó, mà không cần tính ánh xạ. γ lớn: ranh giới uốn sát từng điểm.', 'f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b,  chỉ support vector có αᵢ > 0'],
];

const BOX: Point = [0, 10];
const VIEW = { x: BOX, y: BOX, xTicks: [0, 2, 4, 6, 8, 10], yTicks: [0, 2, 4, 6, 8, 10] };
const ANGLES = Array.from({ length: 37 }, (_, index) => -90 + index * 5);
const degrees = (theta: number) => theta * 180 / Math.PI;

function start(root: Root) {
  const view = $(root, '[data-stage-view]');
  const chartSvg = $(root, '[data-chart]') as unknown as SVGSVGElement;
  const state = { stage: 0, removed: new Set<number>(), kernel: 'rbf' as 'linear' | 'rbf' };
  const angle = slider(root, 'angle', value => `${value}°`, () => draw());
  const cIndex = slider(root, 'c', value => fmt(C_GRID[value], 2), () => draw());
  const gIndex = slider(root, 'gamma', value => fmt(GAMMA_GRID[value], 2), () => draw());
  $$<HTMLInputElement>(root, 'input[name="kernel"]').forEach(input => input.addEventListener('change', () => { state.kernel = input.value as typeof state.kernel; draw(); }));
  wireSteps(root, stage => { state.stage = stage; draw(); });
  $(root, '[data-reset]').addEventListener('click', () => { angle.set(0); cIndex.set(4); gIndex.set(2); state.removed.clear(); state.kernel = 'rbf'; $<HTMLInputElement>(root, 'input[name="kernel"][value="rbf"]').checked = true; draw(); });

  const hardFull = train(SEPARABLE, HARD_C);

  function draw() {
    const stage = state.stage;
    setCard(root, stage, 4, texts[stage]);
    for (const [selector, visible] of [['[data-angle-field]', stage === 0], ['[data-c-field]', stage === 2], ['[data-kernel-field]', stage === 3], ['[data-gamma-field]', stage === 3 && state.kernel === 'rbf']] as const) $(root, selector).hidden = !visible;
    view.replaceChildren();
    if (stage === 0) drawCandidates();
    else if (stage === 1) drawHard();
    else if (stage === 2) drawSoft();
    else drawKernel();
  }

  function plot(label: string) {
    return frame({ width: 440, height: 400, ...VIEW, xLabel: 'đặc trưng x₁', yLabel: 'đặc trưng x₂', label });
  }

  function points(f: Frame, data: Sample[], options: { dim?: Set<number>; onPick?: (index: number) => void } = {}) {
    data.forEach((sample, index) => {
      const shape = marker(f, sample.x, `cls-${sample.y === 1 ? 1 : 0}${options.dim?.has(index) ? ' dim' : ''}${options.onPick ? ' pickable-point' : ''}`, sample.y === 1, 5);
      if (shape && options.onPick) shape.addEventListener('click', event => { event.stopPropagation(); options.onPick!(index); });
    });
  }

  function drawCandidates() {
    const theta = angle.value * Math.PI / 180;
    const c = corridor(SEPARABLE, theta);
    const f = plot(`Dữ liệu tách được và đường phân tách với pháp tuyến ${angle.value}°`);
    if (c.width > 0) {
      polyline(f, strip(c.normal, c.offset - c.width / 2, c.offset + c.width / 2), 'margin-band');
      polyline(f, line(c.normal, c.offset - c.width / 2), 'margin');
      polyline(f, line(c.normal, c.offset + c.width / 2), 'margin');
    }
    polyline(f, line(c.normal, c.offset), 'boundary');
    points(f, SEPARABLE);
    const best = marginWidth(hardFull);
    view.append(legend(), el('div', { class: 'duo' }, figure(f, 'Đường đậm: đường phân tách đặt giữa hành lang. Nét đứt: mép hành lang, chạm điểm gần nhất của mỗi lớp.'),
      el('div', {}, el('dl', { class: 'answer stack' },
        el('div', {}, el('dt', {}, 'Độ rộng hành lang'), el('dd', {}, c.width > 0 ? fmt(c.width) : 'không tách được', ' ', el('span', { class: `badge ${c.width > 0 ? 'ok' : 'bad'}` }, c.width > 0 ? 'tách được' : 'hai lớp chồng lên nhau'))),
        el('div', {}, el('dt', {}, 'Hành lang rộng nhất có thể (SVM)'), el('dd', {}, `${fmt(best)} tại ${Math.round(degrees(Math.atan2(weights(hardFull)[1], weights(hardFull)[0])))}°`))),
        el('p', { class: 'note' }, c.width > 0 && c.width < best * 0.6 ? 'Hành lang hẹp: đường này đúng trên dữ liệu huấn luyện nhưng một điểm mới hơi lệch là bị phân loại sai.' : c.width >= best * 0.97 ? 'Gần như hành lang rộng nhất. Đây chính là lời giải SVM ở bước 2.' : 'Thử xoay tiếp để mở rộng hành lang.'))));
    const widths = ANGLES.map(value => Math.max(0, corridor(SEPARABLE, value * Math.PI / 180).width));
    chart('Độ rộng hành lang theo góc pháp tuyến', [{ label: 'độ rộng hành lang', values: widths, tone: 2 }], ANGLES.map(value => `${value}°`), 3.5, [0, 1, 2, 3], ANGLES.indexOf(Math.round(angle.value / 5) * 5), index => { angle.set(ANGLES[index]); draw(); },
      ['Góc', 'Độ rộng'], ANGLES.map((value, index) => [`${value}°`, widths[index] ? fmt(widths[index]) : 'không tách được']));
  }

  function drawHard() {
    const kept = SEPARABLE.filter((_, index) => !state.removed.has(index));
    const both = kept.some(p => p.y === 1) && kept.some(p => p.y === -1);
    const f = plot('Lời giải SVM lề cứng, các support vector được khoanh tròn');
    let model: Model | null = null;
    if (both) {
      model = train(kept, HARD_C);
      drawLinear(f, model);
      const role = roles(model);
      const map = SEPARABLE.map((_, index) => index).filter(index => !state.removed.has(index));
      map.forEach((original, index) => { if (role[index] !== 'free') marker(f, SEPARABLE[original].x, 'ring', false, 9); });
    }
    points(f, SEPARABLE, { dim: state.removed, onPick: index => { state.removed.has(index) ? state.removed.delete(index) : state.removed.add(index); draw(); } });
    const fullRole = roles(hardFull);
    const supportIdx = fullRole.map((role, index) => (role !== 'free' ? index : -1)).filter(index => index >= 0);
    const removedSupport = [...state.removed].some(index => supportIdx.includes(index));
    const [w1, w2] = model ? weights(model) : [0, 0];
    view.append(legend(true), el('div', { class: 'duo' }, figure(f, 'Nhấp một điểm để bỏ (mờ đi) hoặc đưa lại. Dải xám là lề |w·x + b| ≤ 1.'),
      el('div', {}, model ? el('dl', { class: 'answer stack' },
        el('div', {}, el('dt', {}, 'w, b'), el('dd', {}, `w = (${fmt(w1, 3)}, ${fmt(w2, 3)}),  b = ${fmt(model.b, 3)}`)),
        el('div', {}, el('dt', {}, 'Độ rộng lề 2/‖w‖'), el('dd', {}, fmt(marginWidth(model)))),
        el('div', {}, el('dt', {}, 'Support vector'), el('dd', {}, `${supportCount(model)} / ${kept.length} điểm`))) : el('p', { class: 'note' }, 'Cần ít nhất một điểm của mỗi lớp.'),
      el('p', { class: 'note' }, !state.removed.size ? `Chỉ ${supportIdx.length} điểm có αᵢ > 0. Bỏ bất kỳ điểm nào khác, đường biên không nhúc nhích. Thử bỏ một điểm có vòng tròn.` : removedSupport ? 'Bạn đã bỏ một support vector: lề mở rộng và đường biên dịch chuyển, vì ràng buộc đang “giữ” nó đã biến mất.' : 'Các điểm bị bỏ đều không phải support vector, nên lời giải giữ nguyên.'))));
    const margins = SEPARABLE.map(p => (model ? p.y * decision(model, p.x) : 0));
    const labels = SEPARABLE.map((_, index) => `#${index + 1}`);
    chart('Khoảng cách có dấu y·f(x) của từng điểm', [{ label: 'y·f(x)', values: margins.map(value => Math.min(6, value)), tone: 2 }, { label: 'mép lề (= 1)', values: margins.map(() => 1), tone: 3 }], labels, 6, [0, 1, 3, 6], undefined, undefined,
      ['Điểm', 'Lớp', 'y·f(x)', 'αᵢ'], SEPARABLE.map((p, index) => [labels[index], p.y === 1 ? '+1' : '−1', state.removed.has(index) ? 'đã bỏ' : fmt(margins[index], 3), model && !state.removed.has(index) ? fmt(model.alpha[SEPARABLE.slice(0, index).filter((_, j) => !state.removed.has(j)).length], 3) : '—']));
  }

  function drawSoft() {
    const C = C_GRID[cIndex.value];
    const model = train(OVERLAP, C);
    const f = plot(`SVM lề mềm với C = ${C}`);
    drawLinear(f, model);
    const role = roles(model);
    OVERLAP.forEach((p, index) => { if (role[index] !== 'free') marker(f, p.x, role[index] === 'bound' ? 'ring violate' : 'ring', false, 9); });
    points(f, OVERLAP);
    view.append(legend(true, true), el('div', { class: 'duo' }, figure(f, 'Vòng liền: support vector nằm đúng mép lề. Vòng đứt: điểm bị phạt (αᵢ = C), nằm trong lề hoặc sang nhầm phía.'),
      el('div', {}, el('dl', { class: 'answer stack' },
        el('div', {}, el('dt', {}, 'Độ rộng lề'), el('dd', {}, fmt(marginWidth(model)))),
        el('div', {}, el('dt', {}, 'Support vector'), el('dd', {}, `${supportCount(model)} / ${OVERLAP.length}`)),
        el('div', {}, el('dt', {}, 'Điểm vi phạm lề (y·f < 1)'), el('dd', {}, String(violations(model)))),
        el('div', {}, el('dt', {}, 'Phân loại sai trên tập huấn luyện'), el('dd', {}, String(errors(model))))),
      el('p', { class: 'note' }, C <= 0.1 ? 'C nhỏ: phạt vi phạm rẻ, lề rộng và gần một nửa số điểm trở thành support vector. Mô hình “mềm”, ít nhạy với từng điểm.' : C >= 10 ? 'C lớn: phạt vi phạm đắt, lề co lại để ôm sát các điểm khó. Ranh giới nhạy với nhiễu và điểm ngoại lai.' : 'C vừa phải: cân bằng giữa lề rộng và ít vi phạm. Trong thực tế C được chọn bằng cross-validation.'))));
    const models = C_GRID.map(value => train(OVERLAP, value));
    chart('Số support vector, vi phạm và lỗi theo C', [
      { label: 'support vector', values: models.map(supportCount), tone: 1 }, { label: 'vi phạm lề', values: models.map(violations), tone: 2 }, { label: 'phân loại sai', values: models.map(m => errors(m)), tone: 3 },
    ], C_GRID.map(String), OVERLAP.length, [0, 9, 18, 27, 36], cIndex.value, index => { cIndex.set(index); draw(); },
      ['C', 'Độ rộng lề', 'Support vector', 'Vi phạm', 'Sai'], models.map((m, index) => [C_GRID[index], fmt(marginWidth(m)), supportCount(m), violations(m), errors(m)]));
  }

  function drawKernel() {
    const gamma = GAMMA_GRID[gIndex.value];
    const rbf = state.kernel === 'rbf';
    const model = train(RINGS, 1, rbf ? { type: 'rbf', gamma } : { type: 'linear' });
    const f = plot(rbf ? `SVM kernel RBF với γ = ${gamma}` : 'SVM tuyến tính trên dữ liệu hình vòng');
    field(f, BOX, BOX, 40, (a, b) => decision(model, [a, b]), 1.5);
    contour(f, model);
    const role = roles(model);
    RINGS.forEach((p, index) => { if (role[index] !== 'free') marker(f, p.x, 'ring', false, 8); });
    points(f, RINGS);
    const trainErr = errors(model), testErr = errors(model, RINGS_TEST);
    view.append(legend(true), el('div', { class: 'duo' }, figure(f, 'Nền cam: f(x) > 0 (lớp vòng ngoài); nền xanh: f(x) < 0. Màu càng đậm, |f(x)| càng lớn. Đường đậm: f(x) = 0.'),
      el('div', {}, el('dl', { class: 'answer stack' },
        el('div', {}, el('dt', {}, 'Độ chính xác trên tập huấn luyện'), el('dd', {}, `${percent(1 - trainErr / RINGS.length)} (${trainErr} sai / ${RINGS.length})`)),
        el('div', {}, el('dt', {}, 'Độ chính xác trên tập kiểm tra'), el('dd', {}, `${percent(1 - testErr / RINGS_TEST.length)} (${testErr} sai / ${RINGS_TEST.length})`)),
        el('div', {}, el('dt', {}, 'Support vector'), el('dd', {}, `${supportCount(model)} / ${RINGS.length}`))),
      el('p', { class: 'note' }, !rbf ? 'Đường thẳng không thể bao quanh lõi: mô hình tuyến tính underfit, đoán gần như toàn bộ là lớp đông hơn.' : gamma <= 0.01 ? 'γ quá nhỏ: mỗi điểm ảnh hưởng gần như khắp mặt phẳng, kernel gần tuyến tính nên vẫn chưa bao được lõi.' : gamma >= 10 ? 'γ quá lớn: mỗi support vector chỉ ảnh hưởng một vùng nhỏ quanh nó. Đúng 100% trên tập huấn luyện nhưng sai nhiều trên điểm mới: overfitting.' : 'γ vừa: ranh giới gần hình tròn, đúng trên cả điểm huấn luyện lẫn điểm mới.'))));
    const models = GAMMA_GRID.map(value => train(RINGS, 1, { type: 'rbf', gamma: value }));
    const trainAcc = models.map(m => 1 - errors(m) / RINGS.length), testAcc = models.map(m => 1 - errors(m, RINGS_TEST) / RINGS_TEST.length);
    chart('Độ chính xác theo γ (kernel RBF, C = 1)', [{ label: 'tập huấn luyện', values: trainAcc, tone: 1 }, { label: 'tập kiểm tra', values: testAcc, tone: 2 }], GAMMA_GRID.map(String), 1, [0, 0.5, 1], rbf ? gIndex.value : undefined, index => { gIndex.set(index); draw(); },
      ['γ', 'Huấn luyện', 'Kiểm tra', 'Support vector'], models.map((m, index) => [GAMMA_GRID[index], percent(trainAcc[index]), percent(testAcc[index]), supportCount(m)]));
  }

  function chart(title: string, series: { label: string; values: number[]; tone: 1 | 2 | 3 }[], xLabels: string[], yMax: number, yTicks: number[], cursor: number | undefined, onPick: ((index: number) => void) | undefined, headers: string[], rows: (string | number)[][]) {
    $(root, '[data-chart-title]').textContent = title;
    lineChart(chartSvg, { series, xLabels, yMax, yTicks, yFormat: value => (yMax === 1 ? percent(value) : fmt(value, yMax > 10 ? 0 : 1)), cursor: cursor !== undefined && cursor >= 0 ? cursor : undefined, onPick });
    fillTable($<HTMLTableElement>(root, '[data-table]'), headers, rows);
  }

  onResize(chartSvg.parentElement!, () => draw());
  draw();
}

/** The line n·x = offset, extended across the plot box. */
function line(normal: Point, offset: number): Point[] {
  const [a, b] = normal;
  if (Math.abs(b) > Math.abs(a)) return [[-1, (offset + a) / b], [11, (offset - 11 * a) / b]];
  return [[(offset + b) / a, -1], [(offset - 11 * b) / a, 11]];
}
/** Closed polygon between two parallel lines, for the margin band. */
function strip(normal: Point, low: number, high: number): Point[] {
  const [p1, p2] = line(normal, low), [q1, q2] = line(normal, high);
  return [p1, p2, q2, q1, p1];
}

function drawLinear(f: Frame, model: Model) {
  const [w1, w2] = weights(model), norm = Math.hypot(w1, w2);
  if (norm < 1e-9) return;
  const normal: Point = [w1 / norm, w2 / norm];
  const at = (value: number) => (value - model.b) / norm;
  polyline(f, strip(normal, at(-1), at(1)), 'margin-band');
  polyline(f, line(normal, at(-1)), 'margin');
  polyline(f, line(normal, at(1)), 'margin');
  polyline(f, line(normal, at(0)), 'boundary');
}

/** Zero level set of a kernel decision function by marching squares on a coarse grid, drawn as short segments. */
function contour(f: Frame, model: Model, cells = 60) {
  const step = (BOX[1] - BOX[0]) / cells;
  const values = Array.from({ length: cells + 1 }, (_, i) => Array.from({ length: cells + 1 }, (_, j) => decision(model, [BOX[0] + i * step, BOX[0] + j * step])));
  const cross = (p: Point, q: Point, a: number, b: number): Point => { const t = a / (a - b); return [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]; };
  for (let i = 0; i < cells; i++) for (let j = 0; j < cells; j++) {
    const corners: [Point, number][] = [[[i, j], values[i][j]], [[i + 1, j], values[i + 1][j]], [[i + 1, j + 1], values[i + 1][j + 1]], [[i, j + 1], values[i][j + 1]]];
    const hits: Point[] = [];
    for (let k = 0; k < 4; k++) {
      const [[p, a], [q, b]] = [corners[k], corners[(k + 1) % 4]];
      if ((a > 0) !== (b > 0)) hits.push(cross([BOX[0] + p[0] * step, BOX[0] + p[1] * step], [BOX[0] + q[0] * step, BOX[0] + q[1] * step], a, b));
    }
    if (hits.length >= 2) polyline(f, hits.slice(0, 2), 'boundary');
    if (hits.length === 4) polyline(f, hits.slice(2, 4), 'boundary');
  }
}

function legend(rings = false, dashed = false) {
  return el('div', { class: 'legend' },
    el('span', {}, el('span', { class: 'swatch c0' }), ' lớp −1'), el('span', {}, el('span', { class: 'swatch c1' }), ' lớp +1'),
    rings ? el('span', {}, '◯ support vector') : null, dashed ? el('span', {}, '◌ vi phạm lề (αᵢ = C)') : null);
}
const figure = (f: Frame, caption: string) => el('figure', { class: 'opt-figure' }, f.root as unknown as Node, el('figcaption', { class: 'hint' }, caption));

export const initSvmLab = () => boot('svm', start);
