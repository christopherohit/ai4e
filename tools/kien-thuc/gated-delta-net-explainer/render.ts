// Gated DeltaNet lab view: one 4×4 state matrix stepped through eight key→value tokens.
import { $, $$, bars, boot, el, fillTable, fmt, heatTable, lineChart, onResize, setCard, slider, wireSteps, type Root, type StepText } from '../../_shared/sequence-lab/ui';
import { maxAbs, type Matrix } from '../../_shared/sequence-lab/math';
import { ANSWERS, CHANNELS, FACTS, TOKENS, read, simulate, type Fact, type Step } from './model';

export const explainerKind = 'educational-simulation';

const texts: StepText[] = [
  ['Decay: làm mờ trạng thái cũ', 'Trước khi đọc token mới, toàn bộ ma trận S được nhân với α. α gần 1 giữ lâu; α nhỏ quên nhanh. Ô nào cũng giảm cùng tỉ lệ.', 'S̃ = α · S(t−1)'],
  ['Đọc: bộ nhớ đang dự đoán gì?', 'Nhân key của token với S̃ cho ra value mà bộ nhớ đang tin. Sai số e là phần value mới mà bộ nhớ chưa biết, có thể âm khi cần xóa liên kết cũ.', 'dự đoán = S̃ᵀk     e = v − S̃ᵀk'],
  ['Ghi: chỉ ghi phần sai số', 'Cập nhật là ma trận hạng 1 theo hướng key, cường độ β. Khi thông tin lặp lại, e ≈ 0 nên gần như không ghi. Khi thông tin đổi, e âm ở đáp án cũ và dương ở đáp án mới.', 'S(t) = S̃ + β · k eᵀ'],
  ['Truy vấn: hỏi lại bộ nhớ', 'Dùng một query để đọc S hiện tại. GDN trả về đáp án mới nhất; linear attention chỉ cộng dồn k vᵀ nên giữ cả đáp án cũ lẫn mới. Trong GDN thật, kết quả còn qua RMSNorm và output gate.', 'y = S(t)ᵀ q'],
];

const range = (...matrices: Matrix[]) => Math.max(0.6, ...matrices.map(maxAbs));
const matrixRows = (m: Matrix, focus?: number[], changed?: Matrix) => m.map((values, row) => ({ label: CHANNELS[row], values, focus: focus?.includes(row), changed: changed?.[row].map(value => Math.abs(value) > 0.02) }));
const keyRows = (k: number[]) => k.map((value, index) => (value > 0.5 ? index : -1)).filter(index => index >= 0);
const best = (values: number[]) => values.indexOf(Math.max(...values));

function start(root: Root) {
  const view = $(root, '[data-stage-view]');
  const strip = $(root, '[data-tokens]');
  const position = $(root, '[data-position]');
  const chartSvg = $<SVGSVGElement>(root, '[data-chart]') as unknown as SVGSVGElement;
  const state = { stage: 0, t: 4, query: 'capital' as Fact, compare: true, timer: 0 };
  let steps: Step[] = [];

  const alpha = slider(root, 'alpha', value => fmt(value), () => recompute());
  const beta = slider(root, 'beta', value => fmt(value), () => recompute());
  const compare = $<HTMLInputElement>(root, '[data-compare]');
  const query = $<HTMLSelectElement>(root, '[data-query]');
  compare.addEventListener('change', () => { state.compare = compare.checked; draw(); });
  query.addEventListener('change', () => { state.query = query.value as Fact; draw(); });

  TOKENS.forEach((token, index) => strip.append(el('li', {}, el('button', { type: 'button', class: 'token', 'data-t': String(index) }, el('small', {}, `t${index + 1}`), token.text))));
  $$<HTMLButtonElement>(strip, '[data-t]').forEach(button => button.addEventListener('click', () => go(Number(button.dataset.t))));

  const setStage = wireSteps(root, stage => { state.stage = stage; draw(); });
  const go = (t: number) => { state.t = Math.max(0, Math.min(TOKENS.length - 1, t)); draw(); };
  $(root, '[data-prev]').addEventListener('click', () => go(state.t - 1));
  $(root, '[data-next]').addEventListener('click', () => go(state.t + 1));
  const play = $<HTMLButtonElement>(root, '[data-play]');
  const stop = () => { clearInterval(state.timer); state.timer = 0; play.textContent = '▶ Phát'; play.setAttribute('aria-pressed', 'false'); };
  // Play walks every token through all four steps, so the reader sees decay → read → write → query in order.
  play.addEventListener('click', () => {
    if (state.timer) { stop(); return; }
    if (state.t === TOKENS.length - 1 && state.stage === 3) { state.t = 0; state.stage = 0; }
    play.textContent = '❚❚ Dừng'; play.setAttribute('aria-pressed', 'true');
    const tick = () => {
      if (state.stage < 3) state.stage++; else if (state.t < TOKENS.length - 1) { state.t++; state.stage = 0; } else { stop(); return; }
      setStage(state.stage); draw();
    };
    state.timer = window.setInterval(tick, matchMedia('(prefers-reduced-motion: reduce)').matches ? 2600 : 1500);
  });
  $(root, '[data-reset]').addEventListener('click', () => { stop(); alpha.set(0.95); beta.set(0.9); compare.checked = true; state.compare = true; query.value = 'capital'; state.query = 'capital'; state.t = 4; state.stage = 1; setStage(1); recompute(); });
  document.addEventListener('astro:before-swap', stop, { once: true });

  function recompute() { steps = simulate(alpha.value, beta.value); draw(); }

  function draw() {
    const step = steps[state.t];
    const { token } = step;
    setCard(root, state.stage, 4, texts[state.stage]);
    $$<HTMLButtonElement>(strip, '[data-t]').forEach((button, index) => {
      button.setAttribute('aria-current', String(index === state.t));
      button.classList.toggle('past', index < state.t); button.classList.toggle('future', index > state.t);
    });
    position.textContent = `token ${state.t + 1} / ${TOKENS.length}`;
    const scale = range(...steps.flatMap(item => [item.state, item.prev]), ...(state.compare && state.stage === 3 ? [step.linear] : []));
    const focus = keyRows(token.k);
    view.replaceChildren();

    if (state.stage === 0) {
      view.append(el('div', { class: 'duo' },
        heatTable({ caption: `S(t−1): trước token ${state.t + 1}`, note: 'Hàng = kênh key, cột = đáp án mà bộ nhớ liên kết.', columns: [...ANSWERS], rows: matrixRows(step.prev), range: scale }),
        heatTable({ caption: `S̃ = ${fmt(alpha.value)} · S(t−1)`, note: `Mỗi ô còn ${Math.round(alpha.value * 100)}% giá trị cũ.`, columns: [...ANSWERS], rows: matrixRows(step.decayed), range: scale })));
      view.append(el('p', { class: 'note' }, state.t === 0 ? 'Token đầu tiên: bộ nhớ đang trống nên decay chưa thay đổi gì.' : `Tổng độ lớn của S giảm từ ${fmt(sumAbs(step.prev))} xuống ${fmt(sumAbs(step.decayed))}. Decay làm mọi liên kết mờ đều nhau, không chọn thông tin nào để xóa.`));
    }

    if (state.stage === 1) {
      view.append(heatTable({ caption: `Key của “${token.text}”`, note: 'Vector đã L2-normalize; kênh lớn nhất quyết định hàng của S được đọc và ghi.', columns: [...CHANNELS], rows: [{ label: 'k', values: token.k }], range: 1 }));
      view.append(heatTable({ caption: 'So sánh dự đoán với value thật', note: 'Ô xanh dương là giá trị dương, ô cam là giá trị âm; sai số âm nghĩa là phải xóa bớt.', columns: [...ANSWERS], rows: [
        { label: 'dự đoán S̃ᵀk', values: step.prediction }, { label: 'value v', values: token.v }, { label: '', values: [] }, { label: 'sai số e', values: step.error, focus: true },
      ], range: 1 }));
      view.append(el('p', { class: 'note' }, describeError(step)));
    }

    if (state.stage === 2) {
      view.append(el('div', { class: 'duo' },
        heatTable({ caption: `Phần ghi β · k eᵀ (β = ${fmt(beta.value)})`, note: 'Ma trận hạng 1: hàng theo k, cột theo sai số e.', columns: [...ANSWERS], rows: matrixRows(step.write, focus), range: scale }),
        heatTable({ caption: `S(t) sau token ${state.t + 1}`, note: 'Viền đậm: ô thay đổi đáng kể ở bước này.', columns: [...ANSWERS], rows: matrixRows(step.state, focus, step.write), range: scale })));
      view.append(el('p', { class: 'note' }, `Độ lớn phần ghi: ${fmt(sumAbs(step.write))}. ${sumAbs(step.write) < 0.15 ? 'Gần như không ghi vì bộ nhớ đã biết thông tin này.' : 'Phần ghi đáng kể vì token mang thông tin bộ nhớ chưa có hoặc đã lỗi thời.'}`));
    }

    if (state.stage === 3) {
      const gdn = read(step.state, state.query);
      const linear = read(step.linear, state.query);
      const gdnBest = best(gdn); const linearBest = best(linear);
      const known = TOKENS.slice(0, state.t + 1).filter(item => item.fact === state.query).at(-1);
      const truth = known ? ANSWERS[best(known.v)] : '—';
      const answers = el('dl', { class: 'answer' },
        el('div', {}, el('dt', {}, 'Câu hỏi'), el('dd', {}, FACTS[state.query].label)),
        el('div', {}, el('dt', {}, 'Đáp án mới nhất trong chuỗi'), el('dd', {}, truth)),
        el('div', {}, el('dt', {}, 'GDN trả lời'), el('dd', {}, known ? ANSWERS[gdnBest] + ' ' : 'chưa có', known ? badge(ANSWERS[gdnBest] === truth) : '')));
      if (state.compare) answers.append(el('div', {}, el('dt', {}, 'Linear attention trả lời'), el('dd', {}, known ? ANSWERS[linearBest] + ' ' : 'chưa có', known ? badge(ANSWERS[linearBest] === truth && margin(linear) > 0.15) : '')));
      view.append(answers);
      const max = Math.max(1, ...gdn, ...(state.compare ? linear : []));
      const group = (title: string, values: number[], tone: 1 | 2) => el('div', {}, el('h4', {}, title), bars(ANSWERS.map((label, index) => ({ label, value: Math.max(0, values[index]), text: fmt(values[index]), tone: index === best(values) ? tone : 0, on: index === best(values) })), max, title));
      view.append(el('div', { class: 'duo' }, group('GDN: y = S(t)ᵀq', gdn, 1), state.compare ? group('Linear attention: chỉ cộng k vᵀ', linear, 2) : el('p', { class: 'note' }, 'Bật “So với linear attention” để xem bộ nhớ không có delta rule.')));
      view.append(el('p', { class: 'note' }, known ? queryNote(gdn, linear, state.compare) : `Chuỗi chưa nhắc tới “${FACTS[state.query].label}” trước token ${state.t + 1}; giá trị đọc ra chỉ là nhiễu từ các key gần giống.`));
    }

    lineChart(chartSvg, {
      series: [{ label: 'GDN', values: steps.map(item => item.gdnError), tone: 1 }, ...(state.compare ? [{ label: 'Linear attention', values: steps.map(item => item.linearError), tone: 2 as const }] : [])],
      xLabels: TOKENS.map((_, index) => `t${index + 1}`), yMax: Math.max(1.5, ...steps.map(item => item.linearError)), yTicks: [0, 0.5, 1, 1.5], yFormat: value => fmt(value, 1), cursor: state.t, onPick: go,
    });
    fillTable($<HTMLTableElement>(root, '[data-table]'), ['Token', 'Văn bản', 'Lỗi GDN', 'Lỗi linear', '‖phần ghi‖'], steps.map((item, index) => [`t${index + 1}`, item.token.text, fmt(item.gdnError, 3), fmt(item.linearError, 3), fmt(sumAbs(item.write), 3)]));
  }

  const describeError = (step: Step) => {
    const largest = step.error.reduce((top, value, index) => (Math.abs(value) > Math.abs(step.error[top]) ? index : top), 0);
    if (Math.max(...step.error.map(Math.abs)) < 0.15) return 'Sai số gần 0: bộ nhớ đã dự đoán đúng value này, nên bước ghi sẽ rất nhỏ.';
    const negative = step.error.findIndex(value => value < -0.2);
    if (negative >= 0) return `Sai số âm ở “${ANSWERS[negative]}” (${fmt(step.error[negative])}): bộ nhớ đang nhớ đáp án cũ và delta rule sẽ trừ bớt nó. Sai số dương ở “${ANSWERS[largest]}” là phần cần ghi thêm.`;
    return `Bộ nhớ chưa biết liên kết này, nên sai số lớn nhất ở “${ANSWERS[largest]}” (${fmt(step.error[largest])}).`;
  };
  const queryNote = (gdn: number[], linear: number[], compareOn: boolean) => {
    const gdnText = `GDN chênh ${fmt(margin(gdn))} giữa đáp án mạnh nhất và đáp án thứ hai.`;
    return compareOn ? `${gdnText} Linear attention chỉ chênh ${fmt(margin(linear))}: các liên kết cũ không bao giờ bị trừ đi nên câu trả lời lẫn lộn.` : gdnText;
  };

  onResize(chartSvg.parentElement!, () => draw());
  setStage(1); state.stage = 1; recompute();
}

const sumAbs = (m: Matrix) => m.flat().reduce((sum, value) => sum + Math.abs(value), 0);
const margin = (values: number[]) => { const sorted = [...values].sort((a, b) => b - a); return sorted[0] - sorted[1]; };
const badge = (ok: boolean) => el('span', { class: `badge ${ok ? 'ok' : 'bad'}` }, ok ? 'đúng' : 'lẫn/sai');

export const initGatedDeltaNetLab = () => boot('gated-delta-net', start);
