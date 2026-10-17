// SDVX Modeler lab: pick an agent proposal, watch it run down the 13 validation stages, then read the measured data.
import { $, $$, boot, el, setCard, svg, wireSteps, type Root, type StepText } from '../../_shared/sequence-lab/ui';
import { DATA, PROPOSALS, STAGES, run, type Outcome, type Proposal } from './model';

export const explainerKind = 'educational-simulation';

const texts: StepText[] = [
  ['Pipeline kiểm tra trước khi ghi', 'Agent không ghi file. Nó đề xuất một transaction; Codeguard dựng thay đổi trong overlay bộ nhớ và cho chạy qua 13 tầng kiểm tra theo thứ tự. Chỉ khi mọi tầng không báo lỗi mới phát sinh, thay đổi mới được ghi đúng một lần qua journal.', 'đề xuất → overlay → 13 tầng → journal → ghi'],
  ['Project thật không sạch', 'Đo trên một project MICROSAR tham chiếu: 107 file, 14 MB, 37.622 tham chiếu. Vai trò file phải được nhận diện theo nội dung; sáu kiểu serializer khác nhau cùng tồn tại; baseline đã có sẵn 253 tham chiếu treo và 294 UUID trùng.', 'kiểm tra vi sai: chỉ lỗi mới phát sinh mới chặn'],
  ['Hiệu quả đo được', 'Cùng 40 yêu cầu DID, mỗi câu chạy trên một bản sao baseline riêng. Engine cũ tự viết so với kiến trúc hiện tại (DeepSeek Harness + tool MCP của Codeguard). Bên dưới là thời gian transaction khi số DID trong một ChangeSet tăng.', 'eval40: 25/40 → 32/40 · trung vị 162 s → 67,5 s'],
  ['Token đi đâu?', 'Snapshot telemetry của engine cũ, 237 lượt trong 3,5 tuần. 94,6% token là prompt: agent chủ yếu trả tiền cho việc đọc context. 87,5% thời gian nằm ở LLM, công cụ chỉ chiếm 0,53%. Đây là lý do mọi tool đọc đều có ngân sách ký tự cứng.', 'p50 55 s · p95 1.582 s · max 3.232 s mỗi lượt'],
];
const OUTCOME_TEXT: Record<Outcome, string> = { PASS: 'qua', FAIL: 'chặn', DEGRADED: 'giảm cấp', SKIPPED: 'bỏ qua', NOT_RUN: 'không chạy' };

function start(root: Root) {
  const view = $(root, '[data-stage-view]');
  const state = { stage: 0, proposal: PROPOSALS[3] as Proposal };
  const list = $(root, '[data-proposals]');
  PROPOSALS.forEach(proposal => {
    const label = el('label', {}, el('input', { type: 'radio', name: 'proposal', value: proposal.id, ...(proposal === state.proposal ? { checked: '' } : {}) }), el('span', {}, proposal.title));
    label.querySelector('input')!.addEventListener('change', () => { state.proposal = proposal; draw(); });
    list.append(label);
  });
  const setStage = wireSteps(root, stage => { state.stage = stage; draw(); });
  $(root, '[data-reset]').addEventListener('click', () => { state.proposal = PROPOSALS[3]; $$<HTMLInputElement>(list, 'input')[3].checked = true; state.stage = 0; setStage(0); draw(); });

  function draw() {
    setCard(root, state.stage, 4, texts[state.stage]);
    $(root, '[data-proposal-field]').hidden = state.stage !== 0;
    view.replaceChildren();
    if (state.stage === 0) drawPipeline();
    if (state.stage === 1) drawCorpus();
    if (state.stage === 2) drawEval();
    if (state.stage === 3) drawTelemetry();
  }

  function drawPipeline() {
    const proposal = state.proposal;
    const outcome = run(proposal);
    const committed = !proposal.stopsAt;
    view.append(el('div', { class: 'proposal' }, el('span', { class: 'kicker' }, 'Agent đề xuất'), el('code', {}, proposal.call)));
    const pipe = el('ol', { class: 'pipe', 'aria-label': 'Các tầng kiểm tra' });
    let group = '';
    STAGES.forEach((stage, index) => {
      const result = outcome[stage.id];
      const item = el('li', { class: `st ${result.toLowerCase()}`, title: stage.what },
        el('span', { class: 'st-n' }, String(index + 1)),
        el('span', { class: 'st-body' }, el('b', {}, stage.label), el('small', {}, stage.id)),
        el('span', { class: 'st-r' }, OUTCOME_TEXT[result]));
      if (stage.group !== group) { group = stage.group; item.dataset.group = group; }
      pipe.append(item);
    });
    view.append(pipe);
    const stopped = STAGES.find(stage => stage.id === proposal.stopsAt);
    view.append(el('div', { class: `verdict ${committed ? 'ok' : 'bad'}` },
      el('div', { class: 'verdict-head' }, el('code', {}, proposal.code), el('span', {}, committed ? 'Ghi một lần qua journal' : `Dừng ở tầng ${STAGES.indexOf(stopped!) + 1}: ${stopped!.label}. Không byte nào được ghi.`)),
      el('p', {}, proposal.detail),
      proposal.repair ? el('p', { class: 'repair' }, proposal.repair) : null,
      el('p', { class: 'lesson' }, proposal.lesson)));
  }

  function drawCorpus() {
    const roleTotal = DATA.fileRoles.reduce((sum, [, count]) => sum + count, 0);
    view.append(el('div', { class: 'facts' },
      fact('107', 'file .arxml, 14 MB'), fact('37.622', 'tham chiếu'), fact('788', 'AR path định nghĩa hai lần'), fact('253 / 294', 'tham chiếu treo / UUID trùng có sẵn')));
    view.append(el('div', { class: 'duo' },
      el('figure', { class: 'mini' }, el('figcaption', {}, el('b', {}, 'Vai trò file'), ` · ${roleTotal} file, nhận diện theo nội dung`), hbars(DATA.fileRoles, ([label]) => label === 'ECUC_ACTIVE' ? 'w' : 'r')),
      el('figure', { class: 'mini' }, el('figcaption', {}, el('b', {}, 'Sáu dialect serializer'), ' · số file mỗi kiểu'), dialectStrip(DATA.dialects),
        el('p', { class: 'note' }, 'Khác nhau ở BOM, khai báo XML, indent, CRLF/LF, thứ tự thuộc tính gốc, chữ hoa UUID. Serializer chuẩn hóa về một kiểu sẽ tạo diff hàng nghìn dòng trên file không ai sửa, nên hệ thống sửa theo byte offset và giữ nguyên dialect của từng file (ADR-024).'))));
    view.append(el('figure', { class: 'mini' }, el('figcaption', {}, el('b', {}, 'Ba gốc định nghĩa, không cái nào đi kèm project'), ' · số tham chiếu DEFINITION-REF'), hbars(DATA.definitionRoots, () => 'r', true)));
  }

  function drawEval() {
    const { legacy, dsh, total } = DATA.eval40;
    view.append(el('div', { class: 'duo' },
      el('figure', { class: 'mini' }, el('figcaption', {}, el('b', {}, 'Số câu đạt'), ` · trên ${total} yêu cầu DID`), versus([['Engine cũ', legacy.pass, `${legacy.pass}/${total}`], ['Hiện tại', dsh.pass, `${dsh.pass}/${total}`]], total)),
      el('figure', { class: 'mini' }, el('figcaption', {}, el('b', {}, 'Thời gian trung vị mỗi câu'), ' · giây'), versus([['Engine cũ', legacy.median, `${legacy.median} s`], ['Hiện tại', dsh.median, `${String(dsh.median).replace('.', ',')} s`]], 180))));
    view.append(el('figure', { class: 'mini' }, el('figcaption', {}, el('b', {}, 'ChangeSet nhiều DID'), ' · thời gian transaction, kiểm tra sau mỗi nút, ghi một lần'), scaleChart(DATA.scale)));
    view.append(el('p', { class: 'note' }, 'Replay lại ChangeSet 52 DID cho kết quả NO_OP, giống từng byte. Tiêm lỗi ở checkpoint 51 thì không file nào được ghi. Các câu engine cũ trượt chủ yếu là thiếu ánh xạ Rte hoặc còn sót tên ở file khác: đúng những lỗi closure mà tầng 11 bắt.'));
  }

  function drawTelemetry() {
    const { prompt, completion } = DATA.tokens;
    const share = prompt / (prompt + completion);
    view.append(el('div', { class: 'facts' }, fact('237', 'lượt'), fact('2.049', 'lời gọi LLM'), fact('70,9 M', 'token'), fact(`${(share * 100).toFixed(1).replace('.', ',')}%`, 'là prompt')));
    view.append(el('div', { class: 'duo' },
      el('figure', { class: 'mini' }, el('figcaption', {}, el('b', {}, 'Thời gian một lượt'), ' · giây, đuôi rất dài'), hbars(DATA.latency.map(([label, value]) => [label, value]), () => 'r', false, value => `${Math.round(value).toLocaleString('vi-VN')} s`)),
      el('figure', { class: 'mini' }, el('figcaption', {}, el('b', {}, 'Thời gian rơi vào đâu')), split([['LLM', DATA.time.llm], ['Tool', DATA.time.tools], ['Khác', 100 - DATA.time.llm - DATA.time.tools]]))));
    view.append(el('p', { class: 'note' }, 'Token theo lượt có trung vị 35.368 nhưng p95 là 1,64 triệu. Số token tương quan 0,96 với số lời gọi và số bước: chi phí tăng theo độ dài vòng lặp, không theo độ khó. Vì vậy loop guard, ngân sách ký tự của tool đọc và hint cấu trúc khi tìm kiếm có tác dụng trực tiếp lên chi phí.'));
  }

  draw();
}

const fact = (value: string, label: string) => el('div', { class: 'fact' }, el('b', {}, value), el('span', {}, label));

function hbars(rows: [string, number][], tone: (row: [string, number]) => 'w' | 'r', log = false, format = (value: number) => value.toLocaleString('vi-VN')) {
  const max = Math.max(...rows.map(([, value]) => value));
  const scale = (value: number) => (log ? Math.log10(value + 1) / Math.log10(max + 1) : value / max);
  const list = el('ul', { class: 'hb' });
  for (const row of rows) {
    const fill = el('span', { class: `fill ${tone(row)}` }); fill.style.width = `${Math.max(1.5, scale(row[1]) * 100)}%`;
    list.append(el('li', {}, el('span', { class: 'lbl' }, row[0]), el('span', { class: 'track' }, fill), el('span', { class: 'val' }, format(row[1]))));
  }
  if (log) list.append(el('li', { class: 'hint-row' }, el('span', {}, 'Thang log để thấy cả nhóm nhỏ.')));
  return list;
}

function dialectStrip(counts: number[]) {
  const total = counts.reduce((sum, value) => sum + value, 0);
  const strip = el('div', { class: 'strip-bar', role: 'img', 'aria-label': `Sáu dialect: ${counts.join(', ')} file` });
  counts.forEach((count, index) => { const seg = el('span', { class: `seg s${index}`, title: `Dialect ${index + 1}: ${count} file` }, count > 4 ? String(count) : ''); seg.style.flexGrow = String(Math.max(count, 2.2)); strip.append(seg); });
  return el('div', {}, strip, el('p', { class: 'legend-row' }, `${total} file · dialect 4–6 chỉ 1 file mỗi kiểu nhưng vẫn phải giữ nguyên`));
}

function versus(rows: [string, number, string][], max: number) {
  const list = el('ul', { class: 'vs' });
  rows.forEach(([label, value, text], index) => { const fill = el('span', { class: `fill ${index ? 'w' : 'r'}` }); fill.style.width = `${(value / max) * 100}%`; list.append(el('li', {}, el('span', { class: 'lbl' }, label), el('span', { class: 'track' }, fill), el('b', { class: 'val' }, text))); });
  return list;
}

function split(parts: [string, number][]) {
  const bar = el('div', { class: 'strip-bar split', role: 'img', 'aria-label': parts.map(([label, value]) => `${label} ${value}%`).join(', ') });
  parts.forEach(([label, value], index) => { const seg = el('span', { class: `seg p${index}` }, value > 8 ? `${label} ${String(value).replace('.', ',')}%` : ''); seg.style.flexGrow = String(Math.max(value, 1.2)); bar.append(seg); });
  return el('div', {}, bar, el('p', { class: 'legend-row' }, parts.map(([label, value]) => `${label}: ${value.toFixed(2).replace('.', ',')}%`).join(' · ')));
}

function scaleChart(points: [number, number][]) {
  const W = 560, H = 190, m = { l: 44, r: 16, t: 14, b: 30 };
  const x = (n: number) => m.l + (n / 52) * (W - m.l - m.r);
  const y = (s: number) => H - m.b - (s / 18) * (H - m.t - m.b);
  const chart = svg('svg', { viewBox: `0 0 ${W + 24} ${H}`, class: 'scale', role: 'img', 'aria-label': points.map(([n, s]) => `${n} DID: ${s} giây`).join('; ') });
  for (const tick of [0, 6, 12, 18]) { chart.append(svg('line', { x1: m.l, x2: W - m.r, y1: y(tick), y2: y(tick), class: 'grid' })); chart.append(svg('text', { x: m.l - 8, y: y(tick) + 4, class: 'tick', 'text-anchor': 'end' }, `${tick}s`)); }
  chart.append(svg('polyline', { points: points.map(([n, s]) => `${x(n)},${y(s)}`).join(' '), class: 'line' }));
  for (const [n, s] of points) {
    chart.append(svg('circle', { cx: x(n), cy: y(s), r: 4.5, class: 'pt' }));
    chart.append(svg('text', { x: x(n), y: H - 10, class: 'tick', 'text-anchor': 'middle' }, `${n} DID`));
    // 1 and 10 DID sit almost on top of each other at this scale: one label above-left, the other below-right.
    const place = n === 1 ? { x: x(n) - 4, y: y(s) - 30, a: 'start' } : n === 10 ? { x: x(n) + 6, y: y(s) - 14, a: 'start' } : { x: x(n) - 8, y: y(s) - 10, a: 'end' };
    chart.append(svg('text', { x: place.x, y: place.y, class: 'pt-label', 'text-anchor': place.a }, `${String(s).replace('.', ',')} s`));
  }
  return chart as unknown as Node;
}

export const initSdvxModelerLab = () => boot('sdvx-modeler', start);
