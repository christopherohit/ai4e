// Naive Bayes lab view: a spam filter built from 16 emails. Steps go prior → smoothed word likelihoods → log-score of
// the composed message → what the independence assumption does to words that always travel together.
import { $, $$, bars, boot, el, fillTable, fmt, lineChart, onResize, percent, setCard, slider, wireSteps, type Root, type StepText } from '../../_shared/sequence-lab/ui';
import { CLASSES, EXAMPLES, TRAIN, VOCAB, classCounts, classify, classifyMerged, evidence, likelihood, prior, totalWords, wordCounts, type Counts, type Label, type Word } from './model';

export const explainerKind = 'educational-simulation';

const texts: StepText[] = [
  ['Prior: trước khi đọc email', 'Chưa nhìn nội dung, xác suất một email là spam chỉ dựa vào tỉ lệ trong dữ liệu huấn luyện. Đây là P(lớp), điểm xuất phát của mọi dự đoán.', 'P(spam) = số email spam / tổng số email'],
  ['Likelihood: mỗi lớp “thích” từ nào', 'Đếm số lần mỗi từ xuất hiện trong từng lớp rồi chia cho tổng số từ của lớp đó. Làm trơn Laplace cộng α vào mọi ô đếm để không từ nào có xác suất bằng 0. Kéo α về 0 để thấy vì sao cần làm trơn.', 'P(w | c) = (đếm(w, c) + α) / (tổng từ của c + α·|V|)'],
  ['Phân loại: cộng bằng chứng bằng log', 'Theo định lý Bayes và giả định các từ độc lập khi đã biết lớp, điểm của mỗi lớp là log prior cộng log likelihood của từng từ. Lớp có điểm cao hơn thắng; chuẩn hóa hai điểm cho ra xác suất hậu nghiệm.', 'log P(c | email) = log P(c) + Σ n_w · log P(w | c) − hằng số'],
  ['Giả định “ngây thơ”: đếm trùng bằng chứng', '“trúng” và “thưởng” luôn đi cùng nhau trong spam nên thực chất chỉ là một bằng chứng. Naive Bayes coi chúng độc lập và cộng hai lần, làm xác suất bị đẩy ra cực đoan. Gộp thành một đặc trưng “trúng thưởng” để so sánh.', 'độc lập: P(trúng, thưởng | c) = P(trúng | c) · P(thưởng | c)'],
];

const LABEL_TEXT = ['spam', 'thường'];
const START: Counts = EXAMPLES[0].counts;

function start(root: Root) {
  const view = $(root, '[data-stage-view]');
  const chartSvg = $(root, '[data-chart]') as unknown as SVGSVGElement;
  const box = $(root, '[data-words]');
  const state = { stage: 0, message: { ...START } as Counts };
  const alpha = slider(root, 'alpha', value => fmt(value, 2), () => draw());
  wireSteps(root, stage => { state.stage = stage; draw(); });
  $(root, '[data-reset]').addEventListener('click', () => { alpha.set(1); state.message = { ...START }; draw(); });

  for (const word of VOCAB) box.append(el('button', { type: 'button', class: 'word', 'data-word': word, title: 'Nhấp để thêm một lần; lần thứ tư đưa về 0' }, word, el('b', {}, '0')));
  $$<HTMLButtonElement>(box, '[data-word]').forEach(button => button.addEventListener('click', () => {
    const word = button.dataset.word as Word;
    state.message = { ...state.message, [word]: ((state.message[word] ?? 0) + 1) % 4 };
    draw();
  }));
  const examples = $(root, '[data-examples]');
  EXAMPLES.forEach((example, index) => examples.append(el('button', { type: 'button', 'data-example': String(index) }, example.name)));
  $$<HTMLButtonElement>(examples, '[data-example]').forEach(button => button.addEventListener('click', () => { state.message = { ...EXAMPLES[Number(button.dataset.example)].counts }; draw(); }));

  function draw() {
    const a = alpha.value, stage = state.stage;
    setCard(root, stage, 4, texts[stage]);
    $$<HTMLButtonElement>(box, '[data-word]').forEach(button => {
      const count = state.message[button.dataset.word as Word] ?? 0;
      button.setAttribute('aria-pressed', String(count > 0));
      button.querySelector('b')!.textContent = String(count);
    });
    $(root, '[data-message]').textContent = describe(state.message);
    view.replaceChildren();
    if (stage === 0) drawPrior();
    else if (stage === 1) drawLikelihood(a);
    else if (stage === 2) drawScore(a);
    else drawMerged(a);

    const steps = evidence(state.message, a);
    const safe = steps.map(step => (Number.isNaN(step.pSpam) ? 0.5 : step.pSpam));
    lineChart(chartSvg, {
      series: [{ label: 'P(spam | các từ đã đọc)', values: safe, tone: 2 }, { label: 'ngưỡng 50%', values: safe.map(() => 0.5), tone: 3 }],
      xLabels: steps.map(step => step.label), yMax: 1, yTicks: [0, 0.5, 1], yFormat: percent, cursor: steps.length - 1,
    });
    $(root, '[data-chart-note]').textContent = `α = ${fmt(a, 2)}`;
    fillTable($<HTMLTableElement>(root, '[data-table]'), ['Sau khi đọc', 'P(spam)'], steps.map(step => [step.label, Number.isNaN(step.pSpam) ? 'không xác định (0/0)' : percent(step.pSpam)]));
  }

  function drawPrior() {
    const [spam, normal] = classCounts();
    const list = (label: Label) => el('ul', { class: 'emails' }, ...TRAIN.filter(email => email.label === label).map(email => el('li', { class: label === 0 ? 'spam' : '' }, el('span', {}, LABEL_TEXT[label]), el('span', {}, ...highlight(email.text, email.words)))));
    view.append(el('div', { class: 'duo' }, el('div', {}, el('h4', {}, `${spam} email spam`), list(0)), el('div', {}, el('h4', {}, `${normal} email thường`), list(1))));
    view.append(el('dl', { class: 'answer' },
      el('div', {}, el('dt', {}, 'P(spam)'), el('dd', {}, `${spam}/${TRAIN.length} = ${fmt(prior(0), 3)}`)),
      el('div', {}, el('dt', {}, 'P(thường)'), el('dd', {}, `${normal}/${TRAIN.length} = ${fmt(prior(1), 3)}`))));
    view.append(el('p', { class: 'note' }, 'Phần tô màu là các từ trong bộ từ vựng 10 từ mà mô hình dùng; các từ khác bị bỏ qua. Naive Bayes chỉ nhìn số lần xuất hiện của từng từ, không quan tâm thứ tự (mô hình “túi từ”).'));
  }

  function drawLikelihood(a: number) {
    const [spamCounts, normalCounts] = [wordCounts(0), wordCounts(1)];
    const table = el('table', { class: 'data compact' });
    table.append(el('thead', {}, el('tr', {}, ...['Từ', 'đếm spam', 'P(w | spam)', 'đếm thường', 'P(w | thường)', 'tỉ số'].map(text => el('th', { scope: 'col' }, text)))));
    const body = el('tbody');
    for (const word of VOCAB) {
      const [ps, pn] = [likelihood(word, 0, a), likelihood(word, 1, a)];
      body.append(el('tr', {}, el('td', {}, word), el('td', {}, String(spamCounts[word])), el('td', { class: ps === 0 ? 'zero' : '' }, fmt(ps, 3)), el('td', {}, String(normalCounts[word])), el('td', { class: pn === 0 ? 'zero' : '' }, fmt(pn, 3)), el('td', { class: 'hit' }, ratio(ps, pn))));
    }
    table.append(body);
    view.append(el('div', { class: 'duo' },
      el('div', {}, el('h4', {}, `Tổng số từ: spam ${totalWords(0)}, thường ${totalWords(1)}; |V| = ${VOCAB.length}`), el('div', { class: 'table-scroll' }, table)),
      el('div', {}, el('h4', {}, 'log tỉ số P(w | spam) / P(w | thường)'), bars(VOCAB.map(word => {
        const value = Math.log(likelihood(word, 0, a) / likelihood(word, 1, a));
        return { label: word, value: Number.isFinite(value) ? Math.abs(value) : 4, text: Number.isFinite(value) ? fmt(value) : value > 0 ? '+∞' : '−∞', tone: value >= 0 ? 2 : 1 };
      }), 4, 'Độ lớn log tỉ số theo từ; cam nghiêng về spam, xanh nghiêng về thường'), el('p', { class: 'hint' }, 'Cam: từ đẩy về spam. Xanh: từ đẩy về thường. Thanh càng dài, từ đó càng phân biệt hai lớp.'))));
    view.append(el('p', { class: 'note' }, a === 0
      ? '“trúng” và “thưởng” chưa từng xuất hiện trong email thường nên P(trúng | thường) = 0. Một email thường lỡ chứa chữ “trúng” sẽ bị chấm chắc chắn là spam, bất kể các từ còn lại. Đó là lý do phải làm trơn.'
      : `Với α = ${fmt(a, 2)}, từ chưa gặp trong một lớp vẫn có xác suất ${fmt(likelihood('trúng', 1, a), 3)} (ví dụ P(trúng | thường)). α lớn kéo mọi xác suất về gần 1/|V| = 0.1, làm các từ kém phân biệt hơn.`));
  }

  function drawScore(a: number) {
    const score = classify(state.message, a);
    const table = el('table', { class: 'data compact' });
    table.append(el('thead', {}, el('tr', {}, ...['Thành phần', 'log (spam)', 'log (thường)', 'chênh lệch'].map(text => el('th', { scope: 'col' }, text)))));
    const body = el('tbody');
    const row = (name: string, [s, n]: [number, number], strong = false) => body.append(el('tr', {}, el('td', { class: strong ? 'hit' : '' }, name), el('td', {}, logText(s)), el('td', {}, logText(n)), el('td', { class: 'hit' }, logText(s - n))));
    row('log P(lớp)', score.logPrior);
    for (const term of score.terms) row(term.count > 1 ? `${term.count} × log P(${term.word} | c)` : `log P(${term.word} | c)`, term.logs);
    row('Tổng điểm', score.logScore, true);
    table.append(body);
    const [ps, pn] = score.posterior;
    const verdict: Label = ps >= pn ? 0 : 1;
    view.append(el('div', { class: 'duo' },
      el('div', {}, el('h4', {}, 'Điểm log của từng lớp'), el('div', { class: 'table-scroll' }, table), el('p', { class: 'hint' }, 'Cột chênh lệch dương đẩy về spam, âm đẩy về thường. Cộng log thay vì nhân xác suất để tránh tràn số khi email dài.')),
      el('div', {}, el('h4', {}, 'Xác suất hậu nghiệm'), Number.isNaN(ps)
        ? el('p', { class: 'note' }, 'Cả hai lớp đều có điểm −∞ vì email chứa một từ chưa từng gặp ở mỗi lớp (“trúng” chưa gặp trong thư thường, “họp” chưa gặp trong spam). Tỉ số 0/0 không xác định: mô hình không thể quyết định. Tăng α để sửa.')
        : bars(CLASSES.map((name, index) => ({ label: `P(${name} | email)`, value: score.posterior[index], text: percent(score.posterior[index]), tone: index === 0 ? 2 : 1, on: index === verdict })), 1, 'Xác suất hậu nghiệm'),
      Number.isNaN(ps) ? null : el('dl', { class: 'answer' }, el('div', {}, el('dt', {}, 'Kết luận'), el('dd', {}, LABEL_TEXT[verdict], ' ', el('span', { class: `badge ${verdict === 0 ? 'bad' : 'ok'}` }, percent(Math.max(ps, pn)))))))));
    if (!score.terms.length) view.append(el('p', { class: 'note' }, 'Email chưa có từ nào trong bộ từ vựng: dự đoán chỉ dựa vào prior. Nhấp các từ phía trên để soạn email.'));
  }

  function drawMerged(a: number) {
    const separate = classify(state.message, a), merged = classifyMerged(state.message, a);
    const has = (state.message['trúng'] ?? 0) + (state.message['thưởng'] ?? 0) > 0;
    const together = TRAIN.filter(email => email.words.includes('trúng')).every(email => email.words.includes('thưởng'));
    view.append(el('ul', { class: 'emails' }, ...TRAIN.filter(email => email.words.includes('trúng') || email.words.includes('thưởng')).map(email => el('li', { class: 'spam' }, el('span', {}, 'spam'), el('span', {}, ...highlight(email.text, ['trúng', 'thưởng']))))));
    view.append(el('p', { class: 'note' }, together ? '“trúng” xuất hiện ở đâu thì “thưởng” cũng có mặt: biết từ này là biết luôn từ kia. Giả định độc lập sai hoàn toàn với cặp từ này.' : ''));
    const show = (posterior: [number, number]) => (Number.isNaN(posterior[0]) ? 'không xác định' : percent(posterior[0]));
    view.append(el('dl', { class: 'answer' },
      el('div', {}, el('dt', {}, 'Coi là hai từ độc lập'), el('dd', {}, `P(spam) = ${show(separate.posterior)}`)),
      el('div', {}, el('dt', {}, 'Gộp “trúng thưởng” thành một đặc trưng'), el('dd', {}, `P(spam) = ${show(merged.posterior)}`)),
      el('div', {}, el('dt', {}, 'Bằng chứng của cặp từ (log-odds)'), el('dd', {}, has ? `${logText(sumPair(separate, 0) - sumPair(separate, 1))} so với ${logText(merged.pairLog[0] - merged.pairLog[1])}` : '—'))));
    view.append(el('p', { class: 'note' }, has
      ? 'Bằng chứng của cặp từ bị cộng gần gấp đôi khi tách riêng. Thứ tự lớp thường vẫn đúng (Naive Bayes phân loại khá tốt), nhưng xác suất bị phóng đại: đừng đọc con số này như độ tin cậy đã hiệu chỉnh.'
      : 'Email hiện tại không có “trúng” hay “thưởng”. Chọn ví dụ “Trúng thưởng… mời họp” để thấy kết luận đổi chiều.'));
  }

  onResize(chartSvg.parentElement!, () => draw());
  draw();
}

const sumPair = (score: ReturnType<typeof classify>, label: Label) => score.terms.filter(term => term.word === 'trúng' || term.word === 'thưởng').reduce((sum, term) => sum + term.logs[label], 0);
const logText = (value: number) => (value === -Infinity ? '−∞' : value === Infinity ? '+∞' : Number.isNaN(value) ? '∞ − ∞' : fmt(value, 2));
const ratio = (a: number, b: number) => (b === 0 ? (a === 0 ? '0/0' : '∞') : `${fmt(a / b, 2)}×`);
const describe = (message: Counts) => {
  const words = VOCAB.flatMap(word => Array(message[word] ?? 0).fill(word));
  return words.length ? `Email đang xét: “${words.join(' · ')}”` : 'Email đang xét chưa có từ nào trong bộ từ vựng.';
};

/** Split an email text so the vocabulary words it contains are wrapped in <mark>. */
function highlight(text: string, words: readonly Word[]) {
  const pattern = new RegExp(`(${[...new Set(words)].sort((a, b) => b.length - a.length).map(word => word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'giu');
  return text.split(pattern).map(part => (words.some(word => word.toLowerCase() === part.toLowerCase()) ? el('mark', {}, part) : part));
}

export const initNaiveBayesLab = () => boot('naive-bayes', start);
