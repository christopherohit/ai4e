// Qwen3.8-Flash-Next lab view: layer schedule, long-context memory, MoE routing and gated residual.
import { $, $$, bars, boot, el, fillTable, fmt, heatTable, lineChart, onResize, percent, setCard, slider, svg, wireSteps, type Root, type StepText } from '../../_shared/sequence-lab/ui';
import { BRANCHES, CH, CONFIG, SAMPLE_TOKENS, coreReads, expertParams, memory, mixerOf, residual, route } from './model';

export const explainerKind = 'educational-simulation';

const texts: StepText[] = [
  ['Lịch 48 layer: 3 GDN rồi 1 QSA', 'Mỗi layer có hai sublayer: token mixer (trao đổi thông tin giữa các vị trí) rồi MoE (biến đổi đặc trưng của từng token). Token mixer lặp theo nhóm bốn: ba Gated DeltaNet và một Qwen Sparse Attention.', 'layer ℓ: mixer = QSA nếu ℓ chia hết cho 4, ngược lại GDN'],
  ['Bộ nhớ khi ngữ cảnh dài', 'GDN giữ một trạng thái cố định nên không lớn thêm theo độ dài. Chỉ 12 layer QSA cần KV cache, và mỗi query trong đó đọc khoảng 2048 token được chọn thay vì toàn bộ prefix.', 'KV ≈ 12 layer × 2 (K,V) × 2 head × 256 × 2 byte × L'],
  ['MoE: 10 trong 512 expert', 'Router chấm điểm 512 expert, chọn 10, chuẩn hóa trọng số rồi cộng thêm shared expert luôn chạy. Mọi expert đều phải được lưu, nhưng mỗi token chỉ tính qua 11 expert.', 'MoE(x) = Σ_{e∈Top10} p_e · Expert_e(x) + Shared(x)'],
  ['Gated Residual: đọc theo channel, ghi theo nhánh', 'Residual stream có 4 nhánh. Trước mỗi sublayer, GR chuẩn hóa từng nhánh rồi đọc qua gate riêng cho từng channel. Sau sublayer, cùng output y được ghi vào mỗi nhánh với một hệ số vô hướng trong (0, 2).', 'x = ¼ Σ_i G_i ⊙ RMSNorm(R_i)     R_i ← R_i + s_i · y'],
];

const formatBytes = (bytes: number) => bytes >= 1e9 ? `${fmt(bytes / 1e9, 1)} GB` : `${fmt(bytes / 1e6, bytes >= 1e8 ? 0 : 1)} MB`;
const formatTokens = (tokens: number) => tokens >= 1 << 20 ? `${Math.round(tokens / (1 << 20))}M` : `${Math.round(tokens / 1024)}K`;
const CONTEXTS = [4096, 16384, 65536, 131072, 262144, 524288, 1_048_576];

function start(root: Root) {
  const view = $(root, '[data-stage-view]');
  const chartSvg = $(root, '[data-chart]') as unknown as SVGSVGElement;
  const state = { stage: 0, layer: 4, token: 'attention' as string };
  const context = slider(root, 'context', value => formatTokens(CONTEXTS[value]) + ' token', () => draw());
  const reads = $$<HTMLInputElement>(root, 'input[name^="read"]');
  const writes = $$<HTMLInputElement>(root, 'input[name^="write"]');
  [...reads, ...writes].forEach(input => input.addEventListener('input', () => draw()));
  const tokenSelect = $<HTMLSelectElement>(root, '[data-token]');
  tokenSelect.addEventListener('change', () => { state.token = tokenSelect.value; draw(); });
  const controls = (stage: number) => $$<HTMLElement>(root, '[data-for]').forEach(element => { element.hidden = !element.dataset.for!.split(' ').includes(String(stage)); });
  const setStage = wireSteps(root, stage => { state.stage = stage; draw(); });
  $(root, '[data-reset]').addEventListener('click', () => {
    context.set(4); reads.forEach(input => { input.value = '1'; }); writes.forEach((input, index) => { input.value = String([1.4, 0.6, 1, 0.3][index]); });
    state.layer = 4; state.stage = 0; setStage(0); draw();
  });

  function draw() {
    setCard(root, state.stage, 4, texts[state.stage]);
    controls(state.stage);
    $$<HTMLOutputElement>(root, 'output[data-gate]').forEach(output => { output.textContent = fmt(Number($<HTMLInputElement>(root, `input[name="${output.dataset.gate}"]`).value)); });
    view.replaceChildren();
    if (state.stage === 0) drawSchedule();
    if (state.stage === 1) drawMemory();
    if (state.stage === 2) drawMoe();
    if (state.stage === 3) drawResidual();
    drawChart();
  }

  function drawSchedule() {
    const grid = el('div', { class: 'layer-map', role: 'group', 'aria-label': '48 layer, nhóm theo 4' });
    for (let group = 0; group < CONFIG.layers / 4; group++) {
      const column = el('div', { class: 'layer-group' }, el('span', { class: 'group-label' }, `nhóm ${group + 1}`));
      for (let offset = 1; offset <= 4; offset++) {
        const layer = group * 4 + offset;
        const mixer = mixerOf(layer);
        const button = el('button', { type: 'button', class: `layer ${mixer.toLowerCase()}`, 'aria-pressed': String(layer === state.layer), 'aria-label': `Layer ${layer}: ${mixer} và MoE${layer === 2 ? ', có n-gram embedding' : ''}` }, el('b', {}, `L${layer}`), el('span', {}, mixer), layer === 2 ? el('i', { class: 'ngram', title: 'n-gram embedding' }, 'n') : null);
        button.addEventListener('click', () => { state.layer = layer; draw(); });
        column.append(button);
      }
      grid.append(column);
    }
    view.append(el('div', { class: 'legend' }, el('span', { class: 'swatch gdn' }), 'GDN · 36 layer', el('span', { class: 'swatch qsa' }), 'QSA · 12 layer', el('span', { class: 'swatch ngram' }, 'n'), 'n-gram embedding vào tại layer 2'), grid);
    const layer = state.layer, mixer = mixerOf(layer);
    const flow = el('ol', { class: 'flow', 'aria-label': `Luồng xử lý trong layer ${layer}` });
    const node = (text: string, kind = '', detail = '') => flow.append(el('li', { class: kind }, el('b', {}, text), detail ? el('small', {}, detail) : null));
    node('4 nhánh residual', 'stream', `4 × ${CONFIG.hidden}`);
    if (layer === 2) node('+ n-gram', 'ngram', 'bigram/trigram lookup');
    node('GR read', 'gr', `→ 1 × ${CONFIG.hidden}`);
    if (mixer === 'GDN') node('Gated DeltaNet', 'gdn', `${CONFIG.gdn.qkHeads} head Q/K · ${CONFIG.gdn.vHeads} head V · conv ${CONFIG.gdn.conv}`);
    else node('Qwen Sparse Attention', 'qsa', `indexer ${CONFIG.qsa.indexerHeads} head → top ${CONFIG.qsa.budget} token · core ${CONFIG.qsa.coreHeads}Q/${CONFIG.qsa.kvHeads}KV`);
    node('GR write', 'gr', '4 hệ số s_i');
    node('GR read', 'gr');
    node('MoE', 'moe', `${CONFIG.moe.topK}/${CONFIG.moe.experts} expert + ${CONFIG.moe.shared} shared`);
    node('GR write', 'gr');
    view.append(el('div', { class: 'layer-detail' }, el('h4', {}, `Layer ${layer} · ${mixer} + MoE`), flow,
      el('p', { class: 'note' }, mixer === 'GDN'
        ? el('span', {}, 'Token mixer nén lịch sử vào trạng thái cố định bằng delta rule. ', el('a', { href: '/kien-thuc/gated-delta-net/#cong-cu' }, 'Mở lab Gated DeltaNet →'))
        : el('span', {}, 'Token mixer đọc lại token gốc nhưng chỉ ở các micro-block được indexer chọn. ', el('a', { href: '/kien-thuc/sparse-attention/#cong-cu' }, 'Mở lab Sparse Attention →')))));
  }

  function drawMemory() {
    const tokens = CONTEXTS[context.value];
    const used = memory(tokens);
    const reads = coreReads(tokens);
    view.append(el('dl', { class: 'answer' },
      el('div', {}, el('dt', {}, 'Ngữ cảnh'), el('dd', {}, `${tokens.toLocaleString('vi-VN')} token`)),
      el('div', {}, el('dt', {}, 'KV cache 12 layer QSA + state GDN'), el('dd', {}, formatBytes(used.hybrid))),
      el('div', {}, el('dt', {}, 'Nếu cả 48 layer dùng cùng attention'), el('dd', {}, formatBytes(used.allAttention))),
      el('div', {}, el('dt', {}, 'Token core attention đọc mỗi query'), el('dd', {}, `${reads.toLocaleString('vi-VN')} (${percent(reads / tokens)})`))));
    view.append(el('div', {}, el('h4', {}, 'Bộ nhớ theo loại layer (một chuỗi, bf16)'), bars([
      { label: '36 GDN: state cố định', value: used.gdnPart, text: formatBytes(used.gdnPart), tone: 1 },
      { label: '12 QSA: KV cache', value: used.hybrid - used.gdnPart, text: formatBytes(used.hybrid - used.gdnPart), tone: 2 },
      { label: 'Giả định 48 layer attention', value: used.allAttention, text: formatBytes(used.allAttention), tone: 0 },
    ], used.allAttention, 'So sánh bộ nhớ')));
    view.append(el('p', { class: 'note' }, `State của mỗi layer GDN là ${CONFIG.gdn.vHeads} ma trận ${CONFIG.gdn.headDim}×${CONFIG.gdn.headDim}, không đổi dù ngữ cảnh dài bao nhiêu. KV cache của QSA vẫn tăng tuyến tính: QSA giảm số token cần đọc chứ không xóa token khỏi cache, vì token cũ có thể được chọn lại ở bước sau.`));
  }

  function drawMoe() {
    const routed = route(state.token, state.layer);
    const chosen = new Map(routed.map(item => [item.expert, item.weight]));
    const columns = 32, size = 14, gap = 3;
    const rows = CONFIG.moe.experts / columns;
    const grid = svg('svg', { viewBox: `0 0 ${columns * (size + gap)} ${rows * (size + gap) + 30}`, class: 'experts', role: 'img', 'aria-label': `Lưới 512 expert, 10 expert được chọn cho token ${state.token} ở layer ${state.layer}` });
    for (let expert = 0; expert < CONFIG.moe.experts; expert++) {
      const weight = chosen.get(expert);
      const rect = svg('rect', { x: (expert % columns) * (size + gap), y: Math.floor(expert / columns) * (size + gap), width: size, height: size, rx: 2, class: weight ? 'expert on' : 'expert' });
      if (weight) rect.setAttribute('style', `--w:${Math.round(40 + weight * 140)}%`);
      rect.append(svg('title', {}, weight ? `Expert ${expert}: ${percent(weight)}` : `Expert ${expert}`));
      grid.append(rect);
    }
    const sharedY = rows * (size + gap) + 8;
    grid.append(svg('rect', { x: 0, y: sharedY, width: 4 * (size + gap) - gap, height: size + 6, rx: 3, class: 'expert shared' }));
    grid.append(svg('text', { x: 4 * (size + gap) + 6, y: sharedY + 14, class: 'grid-label' }, 'shared expert: luôn chạy, không qua router'));
    view.append(el('figure', { class: 'attn-figure' }, grid as unknown as Node, el('figcaption', { class: 'hint' }, `Token “${state.token}” · layer ${state.layer}. Ô cam là 10 expert được chọn; đậm hơn = trọng số lớn hơn.`)));
    view.append(el('div', { class: 'duo' },
      el('div', {}, el('h4', {}, 'Trọng số top-10 sau softmax'), bars(routed.map(item => ({ label: `expert ${item.expert}`, value: item.weight, text: percent(item.weight), tone: 2 })), Math.max(...routed.map(item => item.weight)), 'Trọng số expert')),
      el('dl', { class: 'answer stack' },
        el('div', {}, el('dt', {}, 'Tham số mỗi routed expert'), el('dd', {}, `${fmt(expertParams / 1e6, 1)}M`)),
        el('div', {}, el('dt', {}, 'Expert được tính cho token này'), el('dd', {}, `${CONFIG.moe.topK + CONFIG.moe.shared} / ${CONFIG.moe.experts + CONFIG.moe.shared}`)),
        el('div', {}, el('dt', {}, 'Phần routed expert phải lưu nhưng không dùng'), el('dd', {}, percent(1 - CONFIG.moe.topK / CONFIG.moe.experts))))));
  }

  function drawResidual() {
    const readScale = reads.map(input => Number(input.value));
    const write = writes.map(input => Number(input.value));
    const result = residual(readScale, write);
    const channels = Array.from({ length: CH }, (_, index) => `c${index + 1}`);
    const scale = Math.max(...BRANCHES.flat().map(Math.abs), ...result.updated.flat().map(Math.abs));
    view.append(el('div', { class: 'duo' },
      heatTable({ caption: 'GR read: gate G_i theo channel', note: 'Sigmoid (0, 1), do bottleneck rank 320 dự đoán; thanh trượt nhân thêm để thử.', columns: channels, rows: result.gates.map((values, index) => ({ label: `R${index + 1}`, values })), range: 1 }),
      heatTable({ caption: 'Đầu vào sublayer x và output y', note: '8 channel thay cho 2560. x là trung bình có gate của 4 nhánh đã chuẩn hóa.', columns: channels, rows: [{ label: 'x', values: result.x }, { label: '', values: [] }, { label: 'y', values: result.y, focus: true }], range: Math.max(0.5, ...result.x.map(Math.abs), ...result.y.map(Math.abs)) })));
    view.append(el('div', { class: 'duo' },
      heatTable({ caption: 'Trước: 4 nhánh R_i', columns: channels, rows: BRANCHES.map((values, index) => ({ label: `R${index + 1}`, values })), range: scale }),
      heatTable({ caption: 'Sau GR write: R_i + s_i · y', note: `Cùng một y, hệ số s = ${write.map(value => fmt(value)).join(' / ')} cho R1…R4.`, columns: channels, rows: result.updated.map((values, index) => ({ label: `R${index + 1}`, values })), range: scale })));
  }

  function drawChart() {
    const series = CONTEXTS.map(memory);
    lineChart(chartSvg, {
      series: [{ label: 'Hybrid', values: series.map(item => item.hybrid / 1e9), tone: 1 }, { label: '48 attention', values: series.map(item => item.allAttention / 1e9), tone: 2 }],
      xLabels: CONTEXTS.map(formatTokens), xPositions: CONTEXTS.map(tokens => Math.log2(tokens / CONTEXTS[0]) / Math.log2(CONTEXTS.at(-1)! / CONTEXTS[0])),
      yMax: Math.ceil(series.at(-1)!.allAttention / 1e9 / 25) * 25, yTicks: [0, 50, 100], yFormat: value => `${value} GB`, cursor: context.value,
      onPick: index => { context.set(index); state.stage = 1; setStage(1); draw(); },
    });
    fillTable($<HTMLTableElement>(root, '[data-table]'), ['Ngữ cảnh', 'Hybrid', '48 attention', 'Core đọc / query'], CONTEXTS.map((tokens, index) => [tokens.toLocaleString('vi-VN'), formatBytes(series[index].hybrid), formatBytes(series[index].allAttention), coreReads(tokens).toLocaleString('vi-VN')]));
  }

  SAMPLE_TOKENS.forEach(token => tokenSelect.append(el('option', { value: token, ...(token === state.token ? { selected: '' } : {}) }, token)));
  onResize(chartSvg.parentElement!, () => draw());
  draw();
}

export const initQwenLab = () => boot('qwen3-8-flash-next', start);
