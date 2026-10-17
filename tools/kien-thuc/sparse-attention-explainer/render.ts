// Sparse attention lab view: a 32×32 causal attention matrix over a short Python snippet, under dense, fixed sparse
// patterns and Qwen-style block selection.
import { $, $$, bars, boot, el, fillTable, fmt, lineChart, onResize, percent, setCard, slider, svg, wireSteps, type Root, type StepText } from '../../_shared/sequence-lab/ui';
import { BLOCK, BLOCKS, LENGTH, TOKENS, allowed, blockOf, completeBlocks, coverage, indexerScores, strongest, tailStart, topBlocks, topicName, weights, type Pattern } from './model';

export const explainerKind = 'educational-simulation';

const texts: StepText[] = [
  ['Dense causal attention', 'Mỗi hàng là một query, mỗi cột là một key. Query chỉ đọc key ở phía trước (tam giác dưới). Ô càng đậm, trọng số softmax càng lớn. Dense đọc toàn bộ prefix nên chi phí tăng theo L².', 'a(i,j) = softmax_j( q_i · k_j / √d ),  j ≤ i'],
  ['Mẫu thưa cố định', 'Local chỉ giữ cửa sổ gần; global giữ thêm block đầu chuỗi. Mẫu không phụ thuộc nội dung, nên nếu bằng chứng nằm ngoài mẫu thì query không thể đọc được nó trong layer này.', 'a(i,j) = softmax trên j ∈ S(i),  ô ngoài S(i) = 0'],
  ['QSA: indexer chấm điểm block', 'Key được average-pool theo micro-block 4 token. Indexer (4 head, ReLU) chấm các block đã hoàn chỉnh trước query rồi giữ top-k. Block chưa đủ 4 token là phần đuôi, luôn được giữ.', 'I(i,b) = Σ_h ReLU(⟨q_ih, k̄_b⟩)   chỉ khi 4b + 3 ≤ i'],
  ['QSA: core attention trên token được chọn', 'Block được chọn mở lại thành token gốc; softmax chỉ chạy trên các token đó cùng phần đuôi. Điểm indexer dùng để chọn vùng, còn trọng số core mới quyết định value nào được trộn.', 'S(i) = token của top-k block ∪ đuôi;  y_i = Σ_{j∈S(i)} a(i,j) v_j'],
];

const LABEL = 62, TOP = 52, CELL = 16;

function start(root: Root) {
  const view = $(root, '[data-stage-view]');
  const chartSvg = $(root, '[data-chart]') as unknown as SVGSVGElement;
  const state = { stage: 0, pattern: 'local' as Exclude<Pattern, 'dense' | 'qsa'> };
  const query = slider(root, 'query', value => `${value + 1} · “${TOKENS[value].text}”`, () => draw());
  const budget = slider(root, 'budget', value => `${value} block = ${value * BLOCK} token`, () => draw());
  $$<HTMLInputElement>(root, 'input[name="pattern"]').forEach(input => input.addEventListener('change', () => { state.pattern = input.value as typeof state.pattern; draw(); }));
  const setStage = wireSteps(root, stage => { state.stage = stage; draw(); });
  $(root, '[data-reset]').addEventListener('click', () => { query.set(28); budget.set(2); state.stage = 2; setStage(2); draw(); });
  $$<HTMLButtonElement>(root, '[data-example]').forEach(button => button.addEventListener('click', () => { query.set(Number(button.dataset.example)); draw(); }));

  function draw() {
    const i = query.value, k = budget.value;
    setCard(root, state.stage, 4, texts[state.stage]);
    const pattern: Pattern = state.stage === 0 ? 'dense' : state.stage === 1 ? state.pattern : 'qsa';
    $(root, '[data-pattern-field]').hidden = state.stage !== 1;
    view.replaceChildren();
    const dense = weights(i, allowed('dense', i, k));
    const target = strongest(dense);
    const intro = el('p', { class: 'note' }, `Query “${TOKENS[i].text}” (vị trí ${i + 1}) đang tìm `, el('strong', {}, topicName(TOKENS[i].seeks)), `. Với dense attention, key mạnh nhất là “${TOKENS[target].text}” ở vị trí ${target + 1} (${percent(dense[target])}).`);

    if (state.stage === 2) {
      view.append(intro, compressed(i, k, row => { query.set(row); draw(); }));
      const scores = indexerScores(i);
      const picked = topBlocks(i, k);
      const max = Math.max(1e-9, ...scores.map(score => score ?? 0));
      view.append(el('div', {}, el('h4', {}, `Điểm indexer của query ${i + 1}: giữ ${picked.length} block cao nhất`), bars(scores.map((score, block) => ({
        label: `B${block} · ${TOKENS.slice(block * BLOCK, block * BLOCK + BLOCK).map(token => token.text).join(' ')}`,
        value: score ?? 0, text: score === null ? (block === completeBlocks(i) && tailStart(i) <= i ? 'đuôi' : 'tương lai') : fmt(score, 1),
        tone: score === null ? 0 : picked.includes(block) ? 2 : 1, on: picked.includes(block),
      })), max, 'Điểm indexer theo block')));
      const tail = TOKENS.slice(tailStart(i), i + 1).map(token => token.text).join(' ');
      view.append(el('p', { class: 'note' }, `Block được chọn: ${picked.map(block => 'B' + block).join(', ') || 'chưa có block hoàn chỉnh'}. ${tail ? `Phần đuôi “${tail}” (${i + 1 - tailStart(i)} token) luôn được giữ vì block của nó chưa đủ 4 token.` : 'Query vừa đóng một block nên không có phần đuôi.'}`));
    } else {
      const keep = allowed(pattern, i, k);
      const row = weights(i, keep);
      view.append(intro, matrix(pattern, i, k, picked => { query.set(picked); draw(); }));
      if (state.stage > 0) {
        const kept = keep.includes(target);
        view.append(el('dl', { class: 'answer' },
          el('div', {}, el('dt', {}, 'Token được đọc'), el('dd', {}, `${keep.length} / ${i + 1}`)),
          el('div', {}, el('dt', {}, `Giữ được “${TOKENS[target].text}” (vị trí ${target + 1})?`), el('dd', {}, kept ? 'Có ' : 'Không ', el('span', { class: `badge ${kept ? 'ok' : 'bad'}` }, kept ? 'đọc được' : 'bị mask'))),
          el('div', {}, el('dt', {}, 'Phần khối lượng dense được giữ'), el('dd', {}, percent(coverage(pattern, i, k))))));
      }
      const order = [...dense.keys()].filter(j => j <= i).sort((a, b) => dense[b] - dense[a]).slice(0, 6);
      const rowsFor = (values: number[], tone: 1 | 2) => order.map(j => ({ label: `${j + 1} · ${TOKENS[j].text}`, value: values[j], text: values[j] ? percent(values[j]) : 'mask', tone: values[j] ? tone : 0 as const, on: j === target }));
      view.append(state.stage === 0
        ? el('div', {}, el('h4', {}, `6 key mạnh nhất của query ${i + 1}`), bars(rowsFor(dense, 1), 1, 'Trọng số dense'))
        : el('div', { class: 'duo' }, el('div', {}, el('h4', {}, 'Dense'), bars(rowsFor(dense, 1), 1, 'Trọng số dense')), el('div', {}, el('h4', {}, pattern === 'qsa' ? 'QSA (sau khi chọn block)' : pattern === 'local' ? 'Local window' : 'Global + local'), bars(rowsFor(row, 2), 1, 'Trọng số sau mask'))));
    }

    const budgets = Array.from({ length: Math.max(1, completeBlocks(i)) }, (_, index) => index + 1);
    lineChart(chartSvg, {
      series: [{ label: 'QSA top-k', values: budgets.map(b => coverage('qsa', i, b)), tone: 2 }, { label: 'Local', values: budgets.map(b => coverage('local', i, b)), tone: 1 }, { label: 'Global + local', values: budgets.map(b => coverage('global', i, b)), tone: 3 }],
      xLabels: budgets.map(b => `${b * BLOCK}`), yMax: 1, yTicks: [0, 0.5, 1], yFormat: percent, cursor: Math.min(k, budgets.length) - 1, onPick: index => { budget.set(index + 1); draw(); },
    });
    $(root, '[data-chart-note]').textContent = `Query ${i + 1} · “${TOKENS[i].text}”`;
    fillTable($<HTMLTableElement>(root, '[data-table]'), ['Budget (token)', 'QSA', 'Local', 'Global + local'], budgets.map(b => [b * BLOCK, percent(coverage('qsa', i, b)), percent(coverage('local', i, b)), percent(coverage('global', i, b))]));
  }

  onResize(chartSvg.parentElement!, () => draw());
  draw();
}

/** Full attention matrix: shade = weight under the pattern, empty = masked, blank = future. */
function matrix(pattern: Pattern, focus: number, budget: number, pick: (row: number) => void) {
  const size = LENGTH * CELL;
  const figure = svg('svg', { viewBox: `0 0 ${LABEL + size + 4} ${TOP + size + 4}`, class: 'attn', role: 'img', 'aria-label': `Ma trận attention ${LENGTH}×${LENGTH}; hàng ${focus + 1} đang chọn` });
  axisLabels(figure, focus);
  for (let i = 0; i < LENGTH; i++) {
    const keep = allowed(pattern, i, budget);
    const row = weights(i, keep);
    const group = svg('g', { class: 'row', 'data-row': i });
    for (let j = 0; j <= i; j++) {
      const on = keep.includes(j);
      const rect = svg('rect', { x: LABEL + j * CELL + 1, y: TOP + i * CELL + 1, width: CELL - 2, height: CELL - 2, rx: 2, class: on ? 'cell' : 'cell masked' });
      if (on) rect.setAttribute('style', `--w:${Math.round(Math.min(1, Math.sqrt(row[j])) * 100)}%`);
      group.append(rect);
    }
    group.append(svg('rect', { x: 0, y: TOP + i * CELL, width: LABEL + size, height: CELL, class: 'row-hit' }));
    group.addEventListener('click', () => pick(i));
    figure.append(group);
  }
  for (let b = 1; b < BLOCKS; b++) figure.append(svg('line', { x1: LABEL + b * BLOCK * CELL, x2: LABEL + b * BLOCK * CELL, y1: TOP, y2: TOP + size, class: 'block-line' }));
  figure.append(svg('rect', { x: LABEL - 1, y: TOP + focus * CELL, width: (focus + 1) * CELL + 2, height: CELL, class: 'row-focus' }));
  return wrap(figure, 'Nhấp một hàng để chọn query. Đường dọc mờ chia micro-block 4 token; ô viền nhạt là key bị mask.');
}

/** Compressed indexer mask: query rows × key blocks. */
function compressed(focus: number, budget: number, pick: (row: number) => void) {
  const width = BLOCKS * CELL * BLOCK;
  const figure = svg('svg', { viewBox: `0 0 ${LABEL + width + 4} ${TOP + LENGTH * CELL + 4}`, class: 'attn', role: 'img', 'aria-label': 'Mask nén của indexer: hàng là query, cột là block 4 token' });
  axisLabels(figure, focus, true);
  for (let i = 0; i < LENGTH; i++) {
    const scores = indexerScores(i);
    const picked = topBlocks(i, budget);
    const max = Math.max(1e-9, ...scores.map(score => score ?? 0));
    const group = svg('g', { class: 'row' });
    scores.forEach((score, b) => {
      const x = LABEL + b * BLOCK * CELL + 1, y = TOP + i * CELL + 1;
      if (score !== null) {
        group.append(svg('rect', { x, y, width: BLOCK * CELL - 2, height: CELL - 2, rx: 2, class: picked.includes(b) ? 'cell picked' : 'cell', style: `--w:${Math.round(score / max * 85)}%` }));
      } else if (b * BLOCK <= i) {
        group.append(svg('rect', { x, y, width: (i - b * BLOCK + 1) * CELL - 2, height: CELL - 2, rx: 2, class: 'cell tail' }));
      }
    });
    group.append(svg('rect', { x: 0, y: TOP + i * CELL, width: LABEL + width, height: CELL, class: 'row-hit' }));
    group.addEventListener('click', () => pick(i));
    figure.append(group);
  }
  figure.append(svg('rect', { x: LABEL - 1, y: TOP + focus * CELL, width: (focus + 1) * CELL + 2, height: CELL, class: 'row-focus' }));
  return wrap(figure, 'Mỗi ô rộng là một block 4 key. Màu cam: block lọt top-k của hàng đó. Ô sọc: phần đuôi chưa đủ block, luôn giữ.');
}

function axisLabels(figure: SVGSVGElement, focus: number, blocks = false) {
  TOKENS.forEach((token, index) => {
    figure.append(svg('text', { x: LABEL - 6, y: TOP + index * CELL + CELL / 2 + 3.5, class: index === focus ? 'lbl on' : 'lbl', 'text-anchor': 'end' }, `${token.text} ${index + 1}`));
    if (!blocks) figure.append(svg('text', { x: 0, y: 0, class: blockOf(index) % 2 ? 'lbl col alt' : 'lbl col', transform: `translate(${LABEL + index * CELL + CELL / 2 + 3}, ${TOP - 6}) rotate(-60)` }, token.text));
  });
  if (blocks) for (let b = 0; b < BLOCKS; b++) figure.append(svg('text', { x: LABEL + (b + 0.5) * BLOCK * CELL, y: TOP - 8, class: 'lbl', 'text-anchor': 'middle' }, `B${b}`));
}

const wrap = (figure: SVGSVGElement, caption: string) => el('figure', { class: 'attn-figure' }, figure as unknown as Node, el('figcaption', { class: 'hint' }, caption));

export const initSparseAttentionLab = () => boot('sparse-attention', start);
