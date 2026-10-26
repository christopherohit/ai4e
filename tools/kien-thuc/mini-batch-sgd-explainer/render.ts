// Batch / mini-batch / SGD lab view: the same MSE problem trained with B = 64, a chosen B and B = 1. Steps 1–3 walk one
// run update by update (which samples fed the gradient, how far it points from the full gradient); step 4 compares all.
import { $, bars, boot, el, fillTable, fmt, lineChart, onResize, setCard, slider, wireSteps, type Root, type StepText } from '../../_shared/sequence-lab/ui';
import { MIN_LOSS, N, OPTIMUM, gradient, gradientNoise, loss, sampleGradients, type Run } from '../../_shared/optimization-lab/model';
import { arrow, dataPlot, dots, label, lossMap, polyline, type Frame } from '../../_shared/optimization-lab/plots';
import { BATCH_SIZES, START, angle, compare, downhillShare, updatesPerEpoch, type Comparison, type Variant } from './model';

export const explainerKind = 'educational-simulation';

const texts: StepText[] = [
  ['Batch GD: một bước sau khi xem hết dữ liệu', 'Mỗi lần cập nhật, gradient được tính trên cả 64 mẫu. Hướng đi chính xác nên đường đi mượt và có thể dự đoán. Đổi lại, một epoch chỉ cho đúng một bước, và với tập dữ liệu lớn mỗi bước rất tốn bộ nhớ và thời gian.', 'θ ← θ − η · (1/N) Σᵢ₌₁ᴺ ∇ℓᵢ(θ),  N = 64'],
  ['SGD: một mẫu, một bước', 'Mỗi lần cập nhật chỉ dùng một mẫu (điểm cam bên phải). Gradient của một mẫu là ước lượng không chệch của gradient thật nhưng rất nhiễu: các mũi tên mờ cho thấy những hướng có thể rơi vào. Một epoch cho 64 bước; đường đi rung lắc và không đứng yên ở đáy khi η giữ cố định.', 'θ ← θ − η · ∇ℓᵢ(θ),  i lấy lần lượt theo thứ tự đã xáo trộn'],
  ['Mini-batch: trung bình B mẫu', 'Chia dữ liệu đã xáo trộn thành các lô B mẫu. Trung bình B gradient giảm độ nhiễu khoảng √B lần, vẫn cho nhiều bước mỗi epoch và tận dụng phép tính song song trên GPU. Đây là cách huấn luyện mạng nơ-ron phổ biến nhất; “SGD” trong thư viện thường có nghĩa là mini-batch.', 'θ ← θ − η · (1/B) Σ ∇ℓᵢ(θ), tổng trên B mẫu của lô\nđộ lệch chuẩn của ĝ ∝ 1/√B'],
  ['So sánh ba cách', 'Cùng η, cùng số epoch, cùng điểm xuất phát. Mỗi epoch cả ba đều tính 64 gradient mẫu, nhưng số lần cập nhật khác nhau: 1, N/B và 64. Nhiều bước nhỏ nhiễu thường đi xa hơn ít bước chính xác, cho tới khi nhiễu cản việc tiến sát đáy.', 'số cập nhật mỗi epoch = ⌈N / B⌉'],
];

const VARIANT_LABEL: Record<Variant, string> = { batch: 'Batch GD (B = 64)', mini: 'Mini-batch', sgd: 'SGD (B = 1)' };
const TONE: Record<Variant, 1 | 2 | 3> = { batch: 1, mini: 2, sgd: 3 };
const STAGE_VARIANT: Variant[] = ['batch', 'sgd', 'mini'];
const out = (value: number) => (!Number.isFinite(value) || Math.abs(value) > 1e4 ? 'rất lớn' : fmt(value, 3));

function start(root: Root) {
  const view = $(root, '[data-stage-view]');
  const chartSvg = $(root, '[data-chart]') as unknown as SVGSVGElement;
  const position = $(root, '[data-position]');
  const state = { stage: 0, k: 0, seed: 1, timer: 0 };
  let runs: Comparison[] = [];
  const recompute = () => { runs = compare({ lr: lr.value, epochs: epochs.value, mini: BATCH_SIZES[batch.value], seed: state.seed }); state.k = Math.min(state.k, last()); draw(); };
  const lr = slider(root, 'lr', value => fmt(value), () => recompute());
  const epochs = slider(root, 'epochs', value => String(value), () => recompute());
  const batch = slider(root, 'batch', value => `B = ${BATCH_SIZES[value]} · ${updatesPerEpoch(BATCH_SIZES[value])} bước/epoch`, () => recompute());
  // Steps 1–3 start before the first update; the comparison opens on the finished runs.
  wireSteps(root, stage => { stop(); state.stage = stage; state.k = stage === 3 ? epochs.value : 0; $(root, '[data-batch-field]').hidden = stage < 2; draw(); });
  $(root, '[data-shuffle]').addEventListener('click', () => { state.seed++; recompute(); });

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
    // Aim for a few seconds per run whatever its length: SGD has 64× more updates than batch GD.
    const delay = Math.max(40, Math.min(600, 4000 / Math.max(1, last())));
    state.timer = window.setInterval(() => { if (state.k >= last()) { stop(); return; } go(state.k + 1); }, matchMedia('(prefers-reduced-motion: reduce)').matches ? Math.max(delay, 400) : delay);
  });
  $(root, '[data-reset]').addEventListener('click', () => { stop(); lr.set(0.1); epochs.set(5); batch.set(4); state.seed = 1; state.k = 0; recompute(); });
  document.addEventListener('astro:before-swap', stop, { once: true });

  const current = () => runs.find(item => item.variant === STAGE_VARIANT[state.stage])!;
  /** Steps 1–3 walk updates of one run; step 4 walks epochs of all three. */
  const last = () => (state.stage === 3 ? epochs.value : current().run.updates.length);

  function draw() {
    const stage = state.stage;
    setCard(root, stage, 4, texts[stage]);
    view.replaceChildren();
    let epochCursor: number;
    if (stage === 3) {
      position.textContent = `epoch ${state.k} / ${epochs.value}`;
      drawComparison(state.k);
      epochCursor = state.k;
    } else {
      const item = current(), perEpoch = updatesPerEpoch(item.batchSize);
      const epoch = Math.min(epochs.value, Math.ceil(state.k / perEpoch));
      position.textContent = `cập nhật ${state.k} / ${item.run.updates.length} · epoch ${epoch} / ${epochs.value}`;
      drawRun(item, state.k);
      epochCursor = Math.floor(state.k / perEpoch);
    }

    const values = runs.map(item => item.perEpoch);
    const yMax = 2;
    lineChart(chartSvg, {
      series: runs.map(item => ({ label: item.variant === 'mini' ? `Mini-batch (B = ${item.batchSize})` : VARIANT_LABEL[item.variant], values: item.perEpoch, tone: TONE[item.variant] })),
      xLabels: values[0].map((_, index) => String(index)), yMax, yTicks: [0, 0.5, 1, 1.5, 2], yFormat: value => fmt(value, 2), cursor: Math.min(epochCursor, epochs.value),
      onPick: index => { stop(); go(stage === 3 ? index : Math.min(last(), index * updatesPerEpoch(current().batchSize))); },
    });
    $(root, '[data-chart-note]').textContent = `η = ${fmt(lr.value)} · loss nhỏ nhất có thể ${fmt(MIN_LOSS, 3)}`;
    fillTable($<HTMLTableElement>(root, '[data-table]'), ['Epoch', 'Batch GD', `Mini-batch B = ${BATCH_SIZES[batch.value]}`, 'SGD'], values[0].map((_, index) => [index, ...values.map(series => out(series[index]))]));
  }

  function drawRun(item: Comparison, k: number) {
    const { run: result, batchSize } = item;
    const theta = result.points[k], update = result.updates[k];
    const map = lossMap(`Đường đi của ${VARIANT_LABEL[item.variant]} trên đường đồng mức của loss`);
    drawPath(map, result, k, TONE[item.variant]);
    landmarks(map);
    if (update && batchSize < N) {
      for (const g of sampleGradients(theta, batchSize, 24, state.seed + k)) arrow(map, theta, [theta[0] - lr.value * g[0], theta[1] - lr.value * g[1]], 'cloud');
    }
    if (update) {
      arrow(map, theta, [theta[0] - lr.value * update.full[0], theta[1] - lr.value * update.full[1]], 'full');
      arrow(map, theta, update.after, 'step');
    }
    dots(map, [theta], 'current', 5.5);

    const used = update?.batch ?? [];
    const heading = !update ? `Kết thúc sau ${result.updates.length} cập nhật` : batchSize === N ? `Cập nhật ${k + 1}: dùng cả ${N} mẫu` : `Cập nhật ${k + 1}: lô ${used.length} mẫu (tô cam)`;
    view.append(el('div', { class: 'legend' },
      el('span', {}, el('span', { class: 'key-line' }), 'bước thực tế −η·ĝ'),
      el('span', {}, el('span', { class: 'key-line tone-1 dashed' }), 'bước nếu dùng gradient thật −η·∇L'),
      batchSize < N ? el('span', {}, el('span', { class: 'key-line', style: 'opacity:.35' }), '24 lô ngẫu nhiên khác có thể rơi vào') : null));
    view.append(el('div', { class: 'duo' },
      el('figure', { class: 'opt-figure' }, el('h4', {}, 'Không gian tham số (w, b)'), map.root as unknown as Node),
      el('figure', { class: 'opt-figure' }, el('h4', {}, heading), dataPlot('Dữ liệu, các mẫu trong lô hiện tại và đường thẳng hiện tại', theta, used).root as unknown as Node, el('figcaption', { class: 'hint' }, `ŷ = ${out(theta[0])} · x + ${out(theta[1])}. ${batchSize < N ? 'Nét đứt cam: phần dư của từng mẫu trong lô; chỉ những mẫu này quyết định bước đi.' : 'Mọi điểm đều góp phần vào gradient.'}`))));
    const cards = [
      el('div', {}, el('dt', {}, 'Mẫu mỗi cập nhật'), el('dd', {}, String(batchSize))),
      el('div', {}, el('dt', {}, 'Cập nhật mỗi epoch'), el('dd', {}, String(updatesPerEpoch(batchSize)))),
      el('div', {}, el('dt', {}, 'Loss trên cả 64 mẫu'), el('dd', {}, out(loss(theta)))),
    ];
    if (update) cards.push(el('div', {}, el('dt', {}, 'Lệch hướng so với ∇L'), el('dd', {}, `${Math.round(angle(update.step, update.full))}°`)));
    view.append(el('dl', { class: 'answer' }, ...cards));

    if (state.stage === 2) {
      const noise = BATCH_SIZES.map(size => gradientNoise(theta, size));
      const full = Math.hypot(...gradient(theta));
      view.append(el('div', {}, el('h4', {}, `Độ nhiễu của gradient mini-batch tại θ hiện tại (‖∇L‖ = ${fmt(full)})`),
        bars(BATCH_SIZES.map((size, index) => ({ label: `B = ${size}`, value: noise[index], text: fmt(noise[index]), tone: size === batchSize ? 2 : 1, on: size === batchSize })), Math.max(...noise), 'Độ lệch chuẩn của gradient mini-batch theo B'),
        el('p', { class: 'hint' }, 'Độ lệch chuẩn tổng của ĝ quanh ∇L khi rút B mẫu không hoàn lại. Gấp 4 lần B thì nhiễu giảm khoảng một nửa; B = 64 là toàn bộ dữ liệu nên nhiễu bằng 0.')));
    } else if (state.stage === 1 && update) {
      const share = downhillShare(sampleGradients(theta, 1, 64, state.seed + k), theta);
      view.append(el('p', { class: 'note' }, `Tại điểm này, khoảng ${Math.round(share * 100)}% gradient một mẫu còn chỉ xuống dốc (tích vô hướng với ∇L dương). Phần còn lại tạm thời đẩy loss toàn cục lên; chỉ tính trung bình qua nhiều bước, SGD mới đi đúng hướng.`));
    } else if (state.stage === 0) {
      view.append(el('p', { class: 'note' }, `Bước thực tế trùng với bước của gradient thật, vì ở đây ĝ chính là ∇L. Với η = ${fmt(lr.value)} và ${epochs.value} epoch, batch GD chỉ được ${result.updates.length} bước nên còn cách đáy θ* khá xa.`));
    }
  }

  function drawComparison(epoch: number) {
    const map = lossMap('Đường đi của batch GD, mini-batch và SGD trên cùng bản đồ loss');
    for (const item of runs) {
      const upto = epoch * updatesPerEpoch(item.batchSize);
      drawPath(map, item.run, Math.min(upto, item.run.updates.length), TONE[item.variant]);
    }
    landmarks(map);
    const legend = el('div', { class: 'legend' }, ...runs.map(item => el('span', {}, el('span', { class: `key-line tone-${TONE[item.variant]}` }), item.variant === 'mini' ? `Mini-batch (B = ${item.batchSize})` : VARIANT_LABEL[item.variant])));
    const table = el('table', { class: 'data' });
    table.append(el('thead', {}, el('tr', {}, ...['Cách', 'B', 'Cập nhật/epoch', `Đã cập nhật`, `Loss sau epoch ${epoch}`].map(text => el('th', { scope: 'col' }, text)))));
    const body = el('tbody');
    for (const item of runs) {
      const done = Math.min(epoch * updatesPerEpoch(item.batchSize), item.run.updates.length);
      body.append(el('tr', {}, el('td', {}, item.variant === 'mini' ? 'Mini-batch' : VARIANT_LABEL[item.variant].split(' (')[0]), el('td', {}, String(item.batchSize)), el('td', {}, String(updatesPerEpoch(item.batchSize))), el('td', {}, String(done)), el('td', {}, out(loss(item.run.points[done])))));
    }
    table.append(body);
    const sgd = runs.find(item => item.variant === 'sgd')!;
    const tail = sgd.perEpoch.slice(1);
    view.append(legend, el('div', { class: 'duo' },
      el('figure', { class: 'opt-figure' }, el('h4', {}, epoch ? `Sau ${epoch} epoch` : 'Điểm xuất phát chung'), map.root as unknown as Node),
      el('div', { class: 'opt-figure' }, el('h4', {}, 'Cùng chi phí tính gradient, khác số bước'), el('div', { class: 'table-scroll' }, table),
        el('p', { class: 'note' }, tail.length > 1 ? `Loss của SGD sau mỗi epoch dao động trong khoảng ${out(Math.min(...tail))} – ${out(Math.max(...tail))}: nó tới gần đáy (${fmt(MIN_LOSS, 3)}) nhanh nhưng không dừng hẳn. Batch GD giảm đều nhưng chậm vì quá ít bước.` : 'Tăng số epoch để thấy SGD dao động quanh đáy.'),
        el('p', { class: 'hint' }, 'Thử tăng η lên 0.3: batch GD và mini-batch nhanh hơn, còn SGD bắt đầu văng xa vì mỗi bước một mẫu quá nhiễu.'))));
  }

  function drawPath(map: Frame, result: Run, k: number, tone: 1 | 2 | 3) {
    const points = result.points;
    polyline(map, points, `path tone-${tone} faded thin`);
    polyline(map, points.slice(0, k + 1), `path tone-${tone}${points.length > 80 ? ' thin' : ''}`);
    const perEpoch = updatesPerEpoch(result.batchSize);
    const marks = points.length > 80 ? points.filter((_, index) => index % perEpoch === 0 && index <= k) : points.slice(0, k + 1);
    dots(map, marks, `visit tone-${tone}`, 2.5);
  }

  function landmarks(map: Frame) {
    dots(map, [START], 'start', 4);
    dots(map, [OPTIMUM], 'optimum', 5);
    label(map, OPTIMUM, 'θ*', 'mark-label', 7, 14);
    label(map, START, 'θ₀', 'mark-label', 6, -6);
  }

  onResize(chartSvg.parentElement!, () => draw());
  recompute();
}

export const initMiniBatchSgdLab = () => boot('mini-batch-sgd', start);
