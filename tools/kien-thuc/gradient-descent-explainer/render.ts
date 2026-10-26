// Gradient descent lab view: one update rule on a 1D slice of the MSE, the full MSE over (w, b) and a non-convex curve.
// Each step keeps its own learning rate and start; the player walks through the iterations.
import { $, $$, boot, el, fillTable, fmt, lineChart, onResize, setCard, slider, wireSteps, type Root, type StepText } from '../../_shared/sequence-lab/ui';
import { MAX_STABLE_LR, MIN_LOSS, OPTIMUM, gradient, loss, type Point } from '../../_shared/optimization-lab/model';
import { arrow, curve, dataPlot, dots, frame, label, lossMap, polyline } from '../../_shared/optimization-lab/plots';
import { GLOBAL_MIN, HUMP, LOCAL_MIN, SLICE_CURVATURE, bumpy, contraction, descend1D, descend2D, regime, slice, type Problem1D } from './model';

export const explainerKind = 'educational-simulation';

const texts: StepText[] = [
  ['Đạo hàm cho biết hướng xuống dốc', 'Giữ b cố định, loss chỉ còn phụ thuộc w và có dạng parabol. Đạo hàm L′(w) là độ dốc của tiếp tuyến: dương thì loss tăng khi w tăng, nên ta lùi w lại; âm thì tiến w lên. Mỗi bước đi một đoạn tỉ lệ với độ dốc.', 'w ← w − η · L′(w)'],
  ['Learning rate quyết định độ dài bước', 'Trên parabol, mỗi bước nhân khoảng cách tới đáy với (1 − η·L″). η nhỏ: bò chậm. η vừa: hội tụ nhanh. Khi hệ số âm, w nhảy qua lại hai bên đáy. Khi |1 − η·L″| ≥ 1, mỗi bước lại xa đáy hơn: thuật toán phân kỳ.', `|wₖ − w*| = |1 − η·L″|ᵏ · |w₀ − w*|\nL″ = ${fmt(SLICE_CURVATURE)}  →  hội tụ khi η < ${fmt(2 / SLICE_CURVATURE)}`],
  ['Hai tham số: gradient vuông góc với đường đồng mức', 'Gradient là vector đạo hàm riêng (∂L/∂w, ∂L/∂b). Đi ngược gradient là đi xuống dốc nhanh nhất tại chỗ, vuông góc với đường đồng mức. Thung lũng ở đây hẹp và dài nên bước đi chữ chi ngang thung lũng và tiến chậm dọc theo nó.', 'θ ← θ − η ∇L(θ),  ∇L = (2/N) Σ (w·xᵢ + b − yᵢ) · (xᵢ, 1)'],
  ['Mặt loss không lồi: điểm xuất phát quan trọng', 'Hồi quy tuyến tính chỉ có một đáy. Loss của mạng nơ-ron thì có nhiều thung lũng. Gradient descent chỉ nhìn độ dốc tại chỗ, nên nó trượt vào thung lũng gần điểm xuất phát nhất, kể cả khi thung lũng đó không sâu nhất.', 'f(w) = w⁴/4 − w² + 0,3w + 1,5'],
];

interface StageSetup { lr: number; start: number; steps: number; compare: [number, number]; range?: [number, number] }
const SETUP: StageSetup[] = [
  { lr: 0.1, start: -0.5, steps: 25, compare: [0.05, 0.45], range: [-1, 4] },
  { lr: 0.45, start: -0.5, steps: 25, compare: [0.05, 0.2], range: [-1, 4] },
  { lr: 0.3, start: 0, steps: 60, compare: [0.1, 0.38] },
  { lr: 0.1, start: 0.5, steps: 40, compare: [0.05, 0.4], range: [-2.2, 2.2] },
];
const START_2D: Point = [-0.5, 2.6];
const REGIME: Record<ReturnType<typeof regime>, string> = { slow: 'hội tụ chậm', smooth: 'hội tụ đều', oscillating: 'dao động nhưng hội tụ', diverging: 'phân kỳ' };
const SLICE_VIEW = { x: [-1, 4] as Point, y: [0, 12] as Point, xTicks: [-1, 0, 1, 2, 3, 4], yTicks: [0, 3, 6, 9, 12] };
const BUMPY_VIEW = { x: [-2.3, 2.3] as Point, y: [0, 3.5] as Point, xTicks: [-2, -1, 0, 1, 2], yTicks: [0, 1, 2, 3] };
const out = (value: number) => (!Number.isFinite(value) || Math.abs(value) > 1e4 ? 'rất lớn' : fmt(value, 3));

function start(root: Root) {
  const view = $(root, '[data-stage-view]');
  const chartSvg = $(root, '[data-chart]') as unknown as SVGSVGElement;
  const position = $(root, '[data-position]');
  const state = { stage: 0, k: 0, start2D: START_2D, timer: 0 };
  const lr = slider(root, 'lr', value => fmt(value), () => { state.k = Math.min(state.k, last()); draw(); });
  const startInput = slider(root, 'start', value => fmt(value, 1), () => { state.k = 0; draw(); });
  wireSteps(root, stage => { stop(); state.stage = stage; applySetup(); draw(); });
  $$<HTMLButtonElement>(root, '[data-lr]').forEach(button => button.addEventListener('click', () => { lr.set(Number(button.dataset.lr)); state.k = last(); draw(); }));

  const play = $<HTMLButtonElement>(root, '[data-play]');
  const stop = () => { clearInterval(state.timer); state.timer = 0; play.textContent = '▶ Chạy'; play.setAttribute('aria-pressed', 'false'); };
  const go = (k: number) => { state.k = Math.max(0, Math.min(last(), k)); draw(); };
  $(root, '[data-prev]').addEventListener('click', () => { stop(); go(state.k - 1); });
  $(root, '[data-next]').addEventListener('click', () => { stop(); go(state.k + 1); });
  play.addEventListener('click', () => {
    if (state.timer) { stop(); return; }
    if (state.k >= last()) state.k = 0;
    play.textContent = '❚❚ Dừng'; play.setAttribute('aria-pressed', 'true');
    draw();
    state.timer = window.setInterval(() => { if (state.k >= last()) { stop(); return; } go(state.k + 1); }, matchMedia('(prefers-reduced-motion: reduce)').matches ? 1200 : 450);
  });
  $(root, '[data-reset]').addEventListener('click', () => { stop(); state.start2D = START_2D; applySetup(); draw(); });
  document.addEventListener('astro:before-swap', stop, { once: true });

  function applySetup() {
    const setup = SETUP[state.stage];
    lr.set(setup.lr);
    const field = $(root, '[data-start-field]');
    field.hidden = !setup.range;
    if (setup.range) { startInput.input.min = String(setup.range[0]); startInput.input.max = String(setup.range[1]); startInput.set(setup.start); }
    $(root, '[data-map-hint]').hidden = state.stage !== 2;
    // The presets are tuned to the parabola's threshold 2/L″ ≈ 0.56; the other losses have their own thresholds.
    $(root, '[data-lr-field]').hidden = state.stage > 1;
    state.k = state.stage === 0 ? 0 : last();
  }

  const problem = (): Problem1D => (state.stage === 3 ? bumpy : slice);
  const run1D = (rate = lr.value) => descend1D(problem(), startInput.value, rate, SETUP[state.stage].steps);
  const run2D = (rate = lr.value) => descend2D(state.start2D, rate, SETUP[state.stage].steps);
  const lossesFor = (rate: number) => (state.stage === 2 ? run2D(rate).map(theta => loss(theta)) : run1D(rate).map(problem().f));
  const last = () => lossesFor(lr.value).length - 1;

  function draw() {
    const stage = state.stage, k = state.k, rate = lr.value;
    setCard(root, stage, 4, texts[stage]);
    position.textContent = `bước ${k} / ${last()}`;
    view.replaceChildren();
    if (stage === 2) drawPlane(k, rate); else drawCurve(stage, k, rate);

    const setup = SETUP[stage];
    const rates = [rate, ...setup.compare];
    const series = rates.map(value => lossesFor(value));
    const length = setup.steps + 1;
    const pad = (values: number[]) => Array.from({ length }, (_, index) => values[index] ?? Infinity);
    const yMax = stage === 3 ? 3.5 : Math.ceil(series[0][0] * 1.15);
    lineChart(chartSvg, {
      series: [{ label: `η = ${fmt(rate)} (đang chọn)`, values: pad(series[0]), tone: 2 }, { label: `η = ${fmt(setup.compare[0])}`, values: pad(series[1]), tone: 1 }, { label: `η = ${fmt(setup.compare[1])}`, values: pad(series[2]), tone: 3 }],
      xLabels: Array.from({ length }, (_, index) => String(index)), yMax, yTicks: [0, yMax / 2, yMax], yFormat: value => fmt(value, 1), cursor: Math.min(k, length - 1), onPick: index => { stop(); go(index); },
    });
    $(root, '[data-chart-note]').textContent = stage === 3 ? `Xuất phát tại w₀ = ${fmt(startInput.value, 1)}` : stage === 2 ? `Xuất phát tại (${fmt(state.start2D[0])}, ${fmt(state.start2D[1])})` : `Xuất phát tại w₀ = ${fmt(startInput.value, 1)}`;
    fillTable($<HTMLTableElement>(root, '[data-table]'), ['Bước', ...rates.map(value => `η = ${fmt(value)}`)], Array.from({ length }, (_, index) => [index, ...series.map(values => (values[index] === undefined ? 'dừng (phân kỳ)' : out(values[index])))]));
  }

  function drawCurve(stage: number, k: number, rate: number) {
    const p = problem(), path = run1D(rate), w = path[k], next = path[k + 1];
    const viewSpec = stage === 3 ? BUMPY_VIEW : SLICE_VIEW;
    const f = frame({ width: 460, height: 320, ...viewSpec, xLabel: 'w', yLabel: stage === 3 ? 'f(w)' : 'L(w) với b = b*', label: stage === 3 ? 'Đồ thị hàm không lồi với hai thung lũng và đường đi của gradient descent' : 'Parabol loss theo w và bước gradient descent hiện tại' });
    f.root.classList.add('pickable');
    f.root.addEventListener('click', event => { const [value] = f.pick(event); const range = SETUP[stage].range!; stop(); startInput.set(Math.round(Math.max(range[0], Math.min(range[1], value)) * 10) / 10); state.k = stage === 0 ? 0 : last(); draw(); });
    curve(f, p.f, viewSpec.x[0], viewSpec.x[1]);
    if (stage === 3) {
      dots(f, [[GLOBAL_MIN, bumpy.f(GLOBAL_MIN)], [LOCAL_MIN, bumpy.f(LOCAL_MIN)]], 'minimum', 5);
      label(f, [GLOBAL_MIN, bumpy.f(GLOBAL_MIN)], 'đáy sâu', 'mark-label', -24, 18);
      label(f, [LOCAL_MIN, bumpy.f(LOCAL_MIN)], 'đáy nông', 'mark-label', -24, 18);
      label(f, [HUMP, bumpy.f(HUMP)], 'đỉnh dốc', 'mark-label', -22, -10);
    } else {
      dots(f, [[OPTIMUM[0], MIN_LOSS]], 'minimum', 5);
      label(f, [OPTIMUM[0], MIN_LOSS], 'w*', 'mark-label', -6, 18);
    }
    if (stage === 0) {
      const slope = p.df(w);
      polyline(f, [[w - 0.9, p.f(w) - 0.9 * slope], [w + 0.9, p.f(w) + 0.9 * slope]], 'tangent');
      dots(f, path.slice(0, k).map(value => [value, p.f(value)] as Point), 'visit', 3);
      if (next !== undefined) arrow(f, [w, p.f(w)], [next, p.f(w)], 'step');
    } else {
      const visited = path.slice(0, k + 1);
      polyline(f, visited.map(value => [value, p.f(value)] as Point), 'path tone-2 thin');
      dots(f, visited.map(value => [value, p.f(value)] as Point), 'visit', 3);
    }
    dots(f, [[w, p.f(w)]], 'current', 6);
    label(f, [w, p.f(w)], `w${subscript(k)}`);

    const side = el('div', { class: 'opt-figure' });
    if (stage === 0) {
      const slope = p.df(w);
      const rows: [string, string][] = [
        [`w${subscript(k)}`, out(w)],
        [`L(w${subscript(k)})`, out(p.f(w))],
        [`L′(w${subscript(k)})`, out(slope)],
        ['η · L′', `${fmt(rate)} × ${out(slope)} = ${out(rate * slope)}`],
        [`w${subscript(k + 1)}`, `${out(w)} − (${out(rate * slope)}) = ${out(w - rate * slope)}`],
        [`L(w${subscript(k + 1)})`, out(p.f(w - rate * slope))],
      ];
      const width = Math.max(...rows.map(([name]) => name.length));
      side.append(el('h4', {}, `Phép tính ở bước ${k}`), el('pre', { class: 'calc' }, rows.map(([name, value]) => `${name.padEnd(width)} = ${value}`).join('\n')), el('p', { class: 'note' }, Math.abs(slope) < 1e-3 ? 'Độ dốc gần bằng 0: w gần như đứng yên vì đã ở đáy.' : slope < 0 ? 'Độ dốc âm: tăng w làm loss giảm, nên bước cập nhật đẩy w sang phải.' : 'Độ dốc dương: tăng w làm loss tăng, nên bước cập nhật kéo w sang trái.'), el('p', { class: 'hint' }, 'Đường nét đứt là tiếp tuyến; mũi tên cam là bước −η·L′(w). Dốc càng đứng, bước càng dài, dù η không đổi.'));
    } else if (stage === 1) {
      const factor = contraction(rate), kind = regime(rate);
      side.append(el('dl', { class: 'answer stack' },
        el('div', {}, el('dt', {}, 'Hệ số mỗi bước 1 − η·L″'), el('dd', {}, fmt(factor))),
        el('div', {}, el('dt', {}, 'Chế độ'), el('dd', {}, REGIME[kind], ' ', el('span', { class: `badge ${kind === 'diverging' ? 'bad' : 'ok'}` }, kind === 'diverging' ? `η ≥ ${fmt(2 / SLICE_CURVATURE)}` : `η < ${fmt(2 / SLICE_CURVATURE)}`))),
        el('div', {}, el('dt', {}, `Khoảng cách tới w* sau ${k} bước`), el('dd', {}, out(Math.abs(w - OPTIMUM[0]))))),
      el('p', { class: 'note' }, kind === 'oscillating' ? 'Hệ số âm: mỗi bước vượt qua đáy rồi quay lại, nhưng biên độ vẫn nhỏ dần.' : kind === 'diverging' ? 'Mỗi bước vượt đáy xa hơn bước trước; loss tăng vọt và w chạy khỏi khung hình.' : kind === 'slow' ? 'Mỗi bước chỉ rút ngắn một phần nhỏ khoảng cách; cần rất nhiều bước.' : 'Mỗi bước rút ngắn phần lớn khoảng cách còn lại mà không vượt đáy.'));
    } else {
      const end = path.at(-1)!;
      const where = Math.abs(end - GLOBAL_MIN) < 0.05 ? 'đáy sâu' : Math.abs(end - LOCAL_MIN) < 0.05 ? 'đáy nông' : Number.isFinite(end) && Math.abs(end) < 10 ? 'chưa dừng hẳn' : 'phân kỳ';
      side.append(el('dl', { class: 'answer stack' },
        el('div', {}, el('dt', {}, 'Điểm đang xét'), el('dd', {}, `w = ${out(w)},  f = ${out(bumpy.f(w))}`)),
        el('div', {}, el('dt', {}, `Sau ${SETUP[3].steps} bước`), el('dd', {}, where, ' ', el('span', { class: `badge ${where === 'đáy sâu' ? 'ok' : 'bad'}` }, where === 'đáy sâu' ? `f ≈ ${fmt(bumpy.f(GLOBAL_MIN))}` : where === 'đáy nông' ? `f ≈ ${fmt(bumpy.f(LOCAL_MIN))}` : '—')))),
      el('p', { class: 'note' }, `Đỉnh dốc nằm ở w ≈ ${fmt(HUMP)}. Xuất phát bên trái đỉnh thì trượt vào đáy sâu, bên phải thì kẹt ở đáy nông, dù đáy nông cao hơn gần 0,85.`));
    }
    view.append(el('div', { class: 'duo' }, el('figure', { class: 'opt-figure' }, f.root as unknown as Node, el('figcaption', { class: 'hint' }, 'Nhấp lên đồ thị để chọn điểm xuất phát w₀.')), side));
  }

  function drawPlane(k: number, rate: number) {
    const path = run2D(rate), theta = path[k], next = path[k + 1];
    const map = lossMap('Đường đồng mức của loss MSE theo w và b, với đường đi của gradient descent');
    map.root.classList.add('pickable');
    map.root.addEventListener('click', event => { const [w, b] = map.pick(event); stop(); state.start2D = [Math.round(w * 20) / 20, Math.round(b * 20) / 20]; state.k = last(); draw(); });
    polyline(map, path.slice(0, k + 1), 'path tone-2 thin');
    dots(map, path.slice(0, k + 1), 'visit', 2.5);
    dots(map, [OPTIMUM], 'optimum', 5);
    label(map, OPTIMUM, 'θ*', 'mark-label', 7, 14);
    if (next) arrow(map, theta, next, 'step');
    dots(map, [theta], 'current', 5.5);
    const [gw, gb] = gradient(theta);
    view.append(el('p', { class: 'note' }, `Mỗi elip nối các cặp (w, b) có cùng loss. Tâm θ* = (${fmt(OPTIMUM[0])}, ${fmt(OPTIMUM[1])}) là nghiệm bình phương tối thiểu, loss ${fmt(MIN_LOSS, 3)}. Với dữ liệu này, η phải nhỏ hơn ${fmt(MAX_STABLE_LR, 3)} để không phân kỳ.`));
    view.append(el('div', { class: 'duo' },
      el('figure', { class: 'opt-figure' }, el('h4', {}, 'Không gian tham số'), map.root as unknown as Node, el('figcaption', { class: 'hint' }, 'Nhấp lên bản đồ để đặt điểm xuất phát. Mũi tên cam: bước −η∇L tiếp theo.')),
      el('figure', { class: 'opt-figure' }, el('h4', {}, `Đường thẳng ở bước ${k}`), dataPlot(`Dữ liệu và đường thẳng ŷ = ${fmt(theta[0])}x + ${fmt(theta[1])}`, theta).root as unknown as Node, el('figcaption', { class: 'hint' }, `ŷ = ${out(theta[0])} · x + ${out(theta[1])}`))));
    view.append(el('dl', { class: 'answer' },
      el('div', {}, el('dt', {}, 'Loss L(w, b)'), el('dd', {}, out(loss(theta)))),
      el('div', {}, el('dt', {}, '∇L = (∂L/∂w, ∂L/∂b)'), el('dd', {}, `(${out(gw)}, ${out(gb)})`)),
      el('div', {}, el('dt', {}, 'Độ lớn ‖∇L‖'), el('dd', {}, out(Math.hypot(gw, gb))))));
  }

  onResize(chartSvg.parentElement!, () => draw());
  applySetup();
  draw();
}

const DIGITS = '₀₁₂₃₄₅₆₇₈₉';
const subscript = (value: number) => String(value).split('').map(digit => DIGITS[Number(digit)]).join('');

export const initGradientDescentLab = () => boot('gradient-descent', start);
