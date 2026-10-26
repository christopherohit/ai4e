// Bias–variance lab view: one polynomial fit (underfit / overfit) → many fits from fresh training sets → the error
// decomposition → how more data moves the sweet spot. All fits run in the browser.
import { $, $$, bars, boot, el, fillTable, fmt, lineChart, onResize, setCard, slider, wireSteps, type Root, type StepText } from '../../_shared/sequence-lab/ui';
import { band, curve as plotCurve, frame, marker, polyline, type Frame } from '../../_shared/optimization-lab/plots';
import { DEGREES, GRID, NOISE, curve, dataset, decompose, fit, predict, testError, trainError, truth, type Decomposition } from './model';

export const explainerKind = 'educational-simulation';

const texts: StepText[] = [
  ['Một tập dữ liệu: underfit và overfit', 'Dữ liệu sinh từ đường cong thật (nét đứt xanh) cộng nhiễu. Bậc thấp không đủ linh hoạt: sai cả trên tập huấn luyện (underfit). Bậc cao uốn theo từng điểm nhiễu: lỗi huấn luyện gần 0 nhưng lỗi trên dữ liệu mới tăng vọt (overfit).', 'ŷ(x) = w₀ + w₁x + … + w_d x^d,  học bằng bình phương tối thiểu'],
  ['Nhiều tập huấn luyện khác nhau', 'Lấy 20 tập huấn luyện mới từ cùng phân phối và huấn luyện lại. Đường trung bình (đen) lệch khỏi đường thật là bias. Độ tản của các đường quanh trung bình là variance. Bậc thấp: các đường giống nhau nhưng cùng sai. Bậc cao: trung bình đúng nhưng mỗi đường một kiểu.', 'bias(x) = E[ŷ(x)] − f(x),  variance(x) = E[(ŷ(x) − E[ŷ(x)])²]'],
  ['Phân rã lỗi kỳ vọng', 'Lỗi trên dữ liệu mới, lấy trung bình qua mọi tập huấn luyện có thể, tách đúng thành ba phần: bias², variance và nhiễu không thể giảm σ². Tăng độ phức tạp giảm bias² nhưng tăng variance. Điểm tốt nhất là nơi tổng nhỏ nhất, không phải nơi một phần bằng 0.', 'E[(y − ŷ)²] = bias² + variance + σ²'],
  ['Thêm dữ liệu', 'Nhiều dữ liệu hơn hầu như không đổi bias (mô hình vẫn là đa thức bậc d) nhưng giảm mạnh variance, vì mỗi điểm nhiễu ảnh hưởng ít hơn. Vì vậy điểm tối ưu dịch về phía mô hình phức tạp hơn khi dữ liệu tăng.', 'variance ≈ σ² · (d + 1) / n  (hồi quy tuyến tính, x phân bố đều)'],
];

const N_OPTIONS = [12, 15, 20, 30, 50, 100] as const;
const RUNS = 150;
const VIEW = { x: [0, 1] as [number, number], y: [-2, 2] as [number, number], xTicks: [0, 0.25, 0.5, 0.75, 1], yTicks: [-2, -1, 0, 1, 2] };

function start(root: Root) {
  const view = $(root, '[data-stage-view]');
  const chartSvg = $(root, '[data-chart]') as unknown as SVGSVGElement;
  const state = { stage: 0, run: 0 };
  const cache = new Map<number, Decomposition[]>();
  const table = (n: number) => { if (!cache.has(n)) cache.set(n, DEGREES.map(degree => decompose(degree, n, RUNS))); return cache.get(n)!; };
  const degree = slider(root, 'degree', value => `${value}`, () => draw());
  const nIndex = slider(root, 'n', value => `${N_OPTIONS[value]} điểm`, () => draw());
  wireSteps(root, stage => { state.stage = stage; $(root, '[data-n-field]').hidden = stage !== 3; draw(); });
  $(root, '[data-resample]').addEventListener('click', () => { state.run = (state.run + 1) % RUNS; draw(); });
  $$<HTMLButtonElement>(root, '[data-degree]').forEach(button => button.addEventListener('click', () => { degree.set(Number(button.dataset.degree)); draw(); }));
  $(root, '[data-reset]').addEventListener('click', () => { degree.set(3); nIndex.set(0); state.run = 0; draw(); });

  const n = () => (state.stage === 3 ? N_OPTIONS[nIndex.value] : N_OPTIONS[0]);

  function draw() {
    const stage = state.stage, d = degree.value, size = n();
    setCard(root, stage, 4, texts[stage]);
    const rows = table(size), row = rows[d];
    const best = rows.reduce((top, item) => (item.expected < top.expected ? item : top));
    view.replaceChildren();
    if (stage === 0) drawSingle(d, size, best.degree);
    else drawMany(stage, row, size, best);

    const clamp = (value: number) => Math.min(value, 0.8);
    const series = stage === 0
      ? [{ label: 'lỗi kiểm tra', values: rows.map(item => clamp(item.test)), tone: 2 as const }, { label: 'lỗi huấn luyện', values: rows.map(item => clamp(item.train)), tone: 1 as const }]
      : [{ label: 'lỗi kỳ vọng', values: rows.map(item => clamp(item.expected)), tone: 2 as const }, { label: 'bias²', values: rows.map(item => clamp(item.bias2)), tone: 1 as const }, { label: 'variance', values: rows.map(item => clamp(item.variance)), tone: 3 as const }];
    lineChart(chartSvg, { series, xLabels: DEGREES.map(String), yMax: 0.8, yTicks: [0, 0.2, 0.4, 0.6, 0.8], yFormat: value => fmt(value, 1), cursor: d, onPick: index => { degree.set(index); draw(); } });
    $(root, '[data-chart-title]').textContent = stage === 0 ? 'Lỗi huấn luyện và lỗi kiểm tra theo bậc đa thức' : 'Bias², variance và lỗi kỳ vọng theo bậc đa thức';
    $(root, '[data-chart-note]').textContent = `n = ${size} điểm · trung bình ${RUNS} tập huấn luyện · giá trị trên 0.8 bị cắt`;
    fillTable($<HTMLTableElement>(root, '[data-table]'), ['Bậc', 'Lỗi huấn luyện', 'Lỗi kiểm tra', 'bias²', 'variance', 'σ²', 'bias² + var + σ²'], rows.map(item => [item.degree, num(item.train), num(item.test), num(item.bias2), num(item.variance), num(item.noise), num(item.expected)]));
  }

  function drawSingle(d: number, size: number, bestDegree: number) {
    const data = dataset(size, state.run), coef = fit(data, d);
    const f = plot(`Một tập ${size} điểm và đa thức bậc ${d}`);
    plotCurve(f, truth, 0, 1, 'truth');
    plotCurve(f, x => clip(predict(coef, x)), 0, 1, 'model');
    data.x.forEach((x, i) => marker(f, [x, data.y[i]], 'cls-0', false, 4));
    const train = trainError(coef, data), test = testError(coef);
    const kind = d < bestDegree && train > 0.15 ? 'underfit' : test > 1.5 * (NOISE ** 2 + 0.03) && train < test / 2 ? 'overfit' : 'vừa';
    view.append(legend(false), el('div', { class: 'duo' }, figure(f, 'Chấm xanh: tập huấn luyện. Đường cam: mô hình học được. Nét đứt xanh lá: hàm thật f(x) = sin 2πx.'),
      el('div', {}, el('dl', { class: 'answer stack' },
        el('div', {}, el('dt', {}, 'Lỗi huấn luyện (MSE)'), el('dd', {}, num(train))),
        el('div', {}, el('dt', {}, 'Lỗi kiểm tra trên 3000 điểm mới'), el('dd', {}, num(test))),
        el('div', {}, el('dt', {}, 'Chẩn đoán'), el('dd', {}, kind, ' ', el('span', { class: `badge ${kind === 'vừa' ? 'ok' : 'bad'}` }, kind === 'underfit' ? 'cả hai lỗi đều cao' : kind === 'overfit' ? 'lỗi kiểm tra ≫ lỗi huấn luyện' : 'hai lỗi gần nhau')))),
      el('p', { class: 'note' }, `Mức nhiễu σ² = ${fmt(NOISE ** 2, 2)} là giới hạn dưới: không mô hình nào có lỗi kiểm tra thấp hơn. ${d + 1 >= size ? `Với ${d + 1} hệ số và ${size} điểm, đa thức đi qua (gần như) mọi điểm: lỗi huấn luyện ≈ 0.` : ''}`),
      el('p', { class: 'hint' }, 'Bấm “Lấy mẫu lại” để xem cùng bậc đa thức trên một tập huấn luyện khác. Bậc cao đổi hình dạng mạnh mỗi lần: đó là variance.'))));
  }

  function drawMany(stage: number, row: Decomposition, size: number, best: Decomposition) {
    const d = row.degree;
    const f = plot(`${stage === 2 ? 'Dải trung bình ± độ lệch chuẩn' : '20 mô hình'} của đa thức bậc ${d} với ${size} điểm`);
    if (stage === 2) band(f, GRID, row.mean.map((m, g) => clip(m - row.spread[g])), row.mean.map((m, g) => clip(m + row.spread[g])));
    else for (let run = 0; run < 20; run++) { const values = curve(d, size, run); polyline(f, GRID.map((x, g) => [x, clip(values[g])]), 'ensemble'); }
    plotCurve(f, truth, 0, 1, 'truth');
    polyline(f, GRID.map((x, g) => [x, clip(row.mean[g])]), 'mean-fit');
    const side = el('div', {});
    if (stage === 2) {
      const max = Math.max(row.expected, 0.3);
      side.append(el('h4', {}, `Phân rã lỗi, bậc ${d}`), bars([
        { label: 'bias²', value: row.bias2, text: num(row.bias2), tone: 1 },
        { label: 'variance', value: row.variance, text: num(row.variance), tone: 3 },
        { label: 'nhiễu σ²', value: row.noise, text: num(row.noise), tone: 0 },
        { label: 'tổng (kỳ vọng)', value: row.expected, text: num(row.expected), tone: 2, on: true },
      ], max, 'Các thành phần của lỗi kỳ vọng'), el('p', { class: 'note' }, `Đo trực tiếp trên dữ liệu mới: ${num(row.test)}, khớp với tổng ba thành phần. Lỗi kỳ vọng nhỏ nhất ở bậc ${best.degree} (${num(best.expected)}).`));
    } else {
      side.append(el('dl', { class: 'answer stack' },
        el('div', {}, el('dt', {}, 'bias² (trung bình theo x)'), el('dd', {}, num(row.bias2))),
        el('div', {}, el('dt', {}, 'variance (trung bình theo x)'), el('dd', {}, num(row.variance))),
        el('div', {}, el('dt', {}, 'Bậc tốt nhất với n này'), el('dd', {}, `${best.degree} (lỗi ${num(best.expected)})`))));
    }
    side.append(el('p', { class: 'note' }, d <= 2 ? 'Bias cao: đường trung bình không theo kịp hình sin, dù các mô hình khá giống nhau.' : row.variance > 3 * row.bias2 && row.variance > 0.05 ? 'Variance cao: trung bình gần đúng nhưng từng mô hình dao động mạnh, nhất là gần hai đầu khoảng nơi ít dữ liệu.' : 'Cân bằng: bias nhỏ và các mô hình chưa tản nhiều.'));
    if (stage === 3) side.append(el('p', { class: 'hint' }, `Với n = ${size}, bậc tốt nhất là ${best.degree}. Kéo n lên 100 rồi đặt bậc 9: variance giảm mạnh, bias gần như không đổi.`));
    view.append(legend(true), el('div', { class: 'duo' }, figure(f, stage === 2 ? 'Dải cam: E[ŷ(x)] ± độ lệch chuẩn qua 150 tập huấn luyện. Đường đen: trung bình. Nét đứt xanh lá: hàm thật.' : 'Mỗi đường cam mờ là mô hình học từ một tập huấn luyện khác. Đường đen: trung bình của 150 mô hình.'), side));
  }

  onResize(chartSvg.parentElement!, () => draw());
  draw();
}

const plot = (label: string) => frame({ width: 460, height: 340, ...VIEW, xLabel: 'x', yLabel: 'y', label });
const clip = (value: number) => Math.max(-3, Math.min(3, value));
const num = (value: number) => (value > 99 ? '> 99' : fmt(value, 3));
const legend = (many: boolean) => el('div', { class: 'legend' },
  el('span', {}, el('span', { class: 'key-line tone-3 dashed' }), 'hàm thật f(x)'),
  many ? el('span', {}, el('span', { class: 'key-line', style: 'border-color: var(--color-text)' }), 'trung bình các mô hình') : el('span', {}, el('span', { class: 'key-line' }), 'mô hình học được'),
  many ? el('span', {}, el('span', { class: 'key-line', style: 'opacity:.4' }), 'từng mô hình') : null);
const figure = (f: Frame, caption: string) => el('figure', { class: 'opt-figure' }, f.root as unknown as Node, el('figcaption', { class: 'hint' }, caption));

export const initBiasVarianceLab = () => boot('bias-variance', start);
