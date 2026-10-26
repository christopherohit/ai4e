// L1 vs L2 regularization lab view: the diamond and the circle on a 2D loss → shrink vs soft-threshold → coefficient
// paths on an 8-feature regression → what each penalty does with two nearly identical features.
import { $, bars, boot, el, fillTable, fmt, lineChart, onResize, setCard, slider, wireSteps, type Root, type StepText } from '../../_shared/sequence-lab/ui';
import { curve, dots, frame, label, polyline, type Frame } from '../../_shared/optimization-lab/plots';
import { FEATURES, LAMBDAS, SIGNAL, TEST, TRAIN, TRUE_BETA, W_STAR, correlation12, ellipse, lasso2D, lassoShrink, mse, nonZero, paths, quad, ridge2D, ridgeShrink, type Point } from './model';

export const explainerKind = 'educational-simulation';

const texts: StepText[] = [
  ['Hình học: kim cương và hình tròn', 'Phạt L2 tương đương giới hạn w trong một hình tròn; phạt L1 giới hạn trong một hình thoi (kim cương). Lời giải là chỗ đường đồng mức của loss chạm vùng giới hạn lần đầu. Hình thoi có đỉnh nhọn nằm trên trục, nên điểm chạm hay rơi đúng vào đỉnh: một hệ số bằng 0.', 'Ridge: min L(w) + λ‖w‖₂²     Lasso: min L(w) + λ‖w‖₁'],
  ['Co hệ số: chia hay trừ', 'Khi các đặc trưng trực giao, có công thức đóng. Ridge chia mọi hệ số cho (1 + λ): co theo tỉ lệ, không bao giờ bằng 0. Lasso trừ đi λ rồi cắt ở 0 (soft-thresholding): hệ số nhỏ hơn λ bị xóa hẳn. Lý do nằm ở lực kéo về 0: L2 kéo yếu dần khi w nhỏ, L1 kéo đều.', 'Ridge: β = β̂ / (1 + λ)     Lasso: β = sign(β̂) · max(|β̂| − λ, 0)'],
  ['Đường đi hệ số theo λ', 'Hồi quy 25 mẫu, 8 đặc trưng: chỉ x₁, x₂, x₃, x₆ thật sự có ích. Tăng λ từ trái sang phải: Ridge thu nhỏ tất cả cùng lúc, Lasso đưa từng hệ số về đúng 0, tức tự chọn đặc trưng. Biểu đồ dưới cho thấy λ vừa phải giảm lỗi kiểm tra so với không phạt.', 'Ridge: (1/2n)‖y − Xβ‖² + (λ/2)‖β‖²     Lasso: (1/2n)‖y − Xβ‖² + λ‖β‖₁'],
  ['Hai đặc trưng gần như trùng nhau', `x₂ gần như là bản sao của x₁ (tương quan ${correlation12.toFixed(2)}), cả hai cùng có hệ số thật 1.5. Không phạt, mô hình chia trọng số rất tùy tiện. Ridge chia đều cho hai đặc trưng. Lasso giữ một và bỏ hẳn cái kia. Elastic Net trộn hai loại phạt để có cả hai tính chất.`, 'Elastic Net: λ₁‖β‖₁ + λ₂‖β‖₂²'],
];

const LABEL = (index: number) => `λ = ${fmt(LAMBDAS[index], LAMBDAS[index] < 0.01 ? 3 : 2)}`;
const PLANE = { x: [-0.6, 3] as Point, y: [-1, 2] as Point, xTicks: [0, 1, 2, 3], yTicks: [-1, 0, 1, 2] };

function start(root: Root) {
  const view = $(root, '[data-stage-view]');
  const chartSvg = $(root, '[data-chart]') as unknown as SVGSVGElement;
  const state = { stage: 0 };
  const { ridgePath, lassoPath, ols } = paths();
  const ridgeTest = ridgePath.map(beta => mse(beta, TEST)), lassoTest = lassoPath.map(beta => mse(beta, TEST)), olsTest = mse(ols, TEST);
  const lam2d = slider(root, 'lam2d', value => fmt(value, 1), () => draw());
  const lamShrink = slider(root, 'lamshrink', value => fmt(value, 2), () => draw());
  const lamIndex = slider(root, 'lamindex', LABEL, () => draw());
  wireSteps(root, stage => { state.stage = stage; draw(); });
  $(root, '[data-reset]').addEventListener('click', () => { lam2d.set(1.5); lamShrink.set(0.5); lamIndex.set(14); draw(); });

  function draw() {
    const stage = state.stage;
    setCard(root, stage, 4, texts[stage]);
    $(root, '[data-lam2d-field]').hidden = stage !== 0;
    $(root, '[data-lamshrink-field]').hidden = stage !== 1;
    $(root, '[data-lamindex-field]').hidden = stage < 2;
    view.replaceChildren();
    if (stage === 0) drawGeometry();
    else if (stage === 1) drawShrink();
    else if (stage === 2) drawPaths();
    else drawCorrelated();
    const clamp = (value: number) => Math.min(value, 6);
    lineChart(chartSvg, {
      series: [{ label: 'Lasso (L1)', values: lassoTest.map(clamp), tone: 2 }, { label: 'Ridge (L2)', values: ridgeTest.map(clamp), tone: 1 }, { label: 'không phạt', values: LAMBDAS.map(() => clamp(olsTest)), tone: 3 }],
      xLabels: LAMBDAS.map(value => fmt(value, value < 0.01 ? 3 : 2)), yMax: 6, yTicks: [0, 1, 2, 3, 4, 5, 6], yFormat: value => fmt(value, 0), cursor: stage >= 2 ? lamIndex.value : undefined,
      onPick: index => { lamIndex.set(index); if (state.stage < 2) { state.stage = 2; root.querySelector<HTMLButtonElement>('.steps [data-stage="2"]')!.click(); } else draw(); },
    });
    fillTable($<HTMLTableElement>(root, '[data-table]'), ['λ', 'Lỗi kiểm tra Ridge', 'Lỗi kiểm tra Lasso', 'Số hệ số ≠ 0 (Lasso)'], LAMBDAS.map((value, index) => [fmt(value, 3), fmt(ridgeTest[index], 3), fmt(lassoTest[index], 3), nonZero(lassoPath[index])]));
    const bestR = ridgeTest.indexOf(Math.min(...ridgeTest)), bestL = lassoTest.indexOf(Math.min(...lassoTest));
    $(root, '[data-chart-note]').textContent = `Không phạt ${fmt(olsTest, 2)} · Ridge tốt nhất ${fmt(ridgeTest[bestR], 2)} (${LABEL(bestR)}) · Lasso tốt nhất ${fmt(lassoTest[bestL], 2)} (${LABEL(bestL)})`;
  }

  function drawGeometry() {
    const lambda = lam2d.value;
    const ridgeW = ridge2D(lambda), lassoW = lasso2D(lambda);
    const panel = (kind: 'l2' | 'l1', w: Point) => {
      const f = frame({ width: 360, height: 320, ...PLANE, xLabel: 'w₁', yLabel: 'w₂', label: kind === 'l2' ? 'Vùng giới hạn hình tròn của L2 và lời giải Ridge' : 'Vùng giới hạn hình thoi của L1 và lời giải Lasso' });
      axes(f);
      const budget = kind === 'l2' ? Math.hypot(w[0], w[1]) : Math.abs(w[0]) + Math.abs(w[1]);
      const region: Point[] = kind === 'l2' ? Array.from({ length: 97 }, (_, i) => [budget * Math.cos(i / 96 * 2 * Math.PI), budget * Math.sin(i / 96 * 2 * Math.PI)]) : [[budget, 0], [0, budget], [-budget, 0], [0, -budget], [budget, 0]];
      polyline(f, region, `region ${kind}`);
      const touch = quad(w);
      for (const c of [touch, ...[0.05, 0.3, 0.8, 1.6].filter(c => Math.abs(c - touch) > 0.03)]) if (c > 1e-6) polyline(f, ellipse(c), c === touch ? 'boundary' : 'margin');
      const trace = Array.from({ length: 51 }, (_, i) => (kind === 'l2' ? ridge2D : lasso2D)(i / 10));
      polyline(f, trace, `path tone-${kind === 'l2' ? 1 : 2} thin`);
      dots(f, [W_STAR], 'optimum', 5); label(f, W_STAR, 'w* (không phạt)', 'mark-label', 6, -8);
      dots(f, [w], 'current', 6); label(f, w, `(${fmt(w[0])}, ${fmt(w[1])})`, 'mark-label', 8, 14);
      return f;
    };
    const zero = Math.abs(lassoW[1]) < 1e-6;
    view.append(el('div', { class: 'duo' },
      el('figure', { class: 'opt-figure' }, el('h4', {}, 'L2 · Ridge'), panel('l2', ridgeW).root as unknown as Node, el('figcaption', { class: 'hint' }, 'Elip đậm: đường đồng mức của loss chạm hình tròn. Đường xanh mảnh: lời giải khi λ tăng từ 0 đến 5.')),
      el('figure', { class: 'opt-figure' }, el('h4', {}, 'L1 · Lasso'), panel('l1', lassoW).root as unknown as Node, el('figcaption', { class: 'hint' }, 'Đường cam mảnh: lời giải Lasso khi λ tăng. Nó chạy dọc trục w₁ khi w₂ đã bằng 0.'))));
    view.append(el('dl', { class: 'answer' },
      el('div', {}, el('dt', {}, 'Ridge'), el('dd', {}, `w = (${fmt(ridgeW[0], 3)}, ${fmt(ridgeW[1], 3)})`)),
      el('div', {}, el('dt', {}, 'Lasso'), el('dd', {}, `w = (${fmt(lassoW[0], 3)}, ${fmt(lassoW[1], 3)})`, ' ', el('span', { class: `badge ${zero ? 'ok' : ''}` }, zero ? 'w₂ = 0 đúng tuyệt đối' : 'chưa thưa')))));
    view.append(el('p', { class: 'note' }, lambda === 0 ? 'λ = 0: không phạt, cả hai cho đúng w* = (2.2, 0.7).' : zero ? `Từ λ ≈ 1.01 trở đi, Lasso dừng ở đỉnh hình thoi trên trục w₁: w₂ bị loại khỏi mô hình. Ridge vẫn giữ w₂ = ${fmt(ridgeW[1], 3)}, nhỏ dần nhưng không bao giờ bằng 0.` : 'λ còn nhỏ: Lasso đã kéo w₂ về gần trục nhưng chưa chạm. Kéo λ vượt 1.01 để w₂ bằng đúng 0.'));
  }

  function drawShrink() {
    const lambda = lamShrink.value;
    const view1 = { x: [-3, 3] as Point, y: [-3, 3] as Point, xTicks: [-3, -2, -1, 0, 1, 2, 3], yTicks: [-3, -2, -1, 0, 1, 2, 3] };
    const f = frame({ width: 360, height: 320, ...view1, xLabel: 'β̂ (bình phương tối thiểu)', yLabel: 'β sau khi phạt', label: 'Hàm co hệ số của Ridge và Lasso' });
    axes(f);
    polyline(f, [[-3, -3], [3, 3]], 'margin');
    curve(f, beta => ridgeShrink(beta, lambda), -3, 3, 'path tone-1');
    curve(f, beta => lassoShrink(beta, lambda), -3, 3, 'path tone-2');
    if (lambda > 0) { polyline(f, [[-lambda, -3], [-lambda, 3]], 'vline'); polyline(f, [[lambda, -3], [lambda, 3]], 'vline'); }
    const g = frame({ width: 360, height: 320, x: [-3, 3], y: [-2.2, 2.2], xTicks: [-3, -2, -1, 0, 1, 2, 3], yTicks: [-2, -1, 0, 1, 2], xLabel: 'w', yLabel: 'lực kéo về 0 (đạo hàm của phạt)', label: 'Đạo hàm của phạt L1 và L2' });
    axes(g);
    curve(g, w => lambda * w, -3, 3, 'path tone-1');
    polyline(g, [[-3, -lambda], [0, -lambda]], 'path tone-2'); polyline(g, [[0, lambda], [3, lambda]], 'path tone-2');
    const sample = [0.3, 1, 2.5];
    view.append(el('div', { class: 'legend' }, el('span', {}, el('span', { class: 'key-line tone-1' }), 'Ridge (L2)'), el('span', {}, el('span', { class: 'key-line' }), 'Lasso (L1)'), el('span', {}, el('span', { class: 'key-line dashed', style: 'border-color: var(--color-text-2)' }), 'không phạt')));
    view.append(el('div', { class: 'duo' },
      el('figure', { class: 'opt-figure' }, el('h4', {}, 'Hệ số sau khi phạt'), f.root as unknown as Node, el('figcaption', { class: 'hint' }, `Vùng giữa hai vạch đứng (|β̂| ≤ λ = ${fmt(lambda, 2)}): Lasso trả về đúng 0.`)),
      el('figure', { class: 'opt-figure' }, el('h4', {}, 'Lực kéo về 0'), g.root as unknown as Node, el('figcaption', { class: 'hint' }, 'L2: lực tỉ lệ với w, gần 0 thì gần như không kéo. L1: lực luôn bằng λ dù w nhỏ đến đâu, đủ để đẩy w về đúng 0.'))));
    const table = el('table', { class: 'data compact' });
    table.append(el('thead', {}, el('tr', {}, ...['β̂', 'Ridge β̂/(1+λ)', 'Lasso max(|β̂| − λ, 0)'].map(text => el('th', { scope: 'col' }, text)))));
    table.append(el('tbody', {}, ...sample.map(beta => el('tr', {}, el('td', {}, fmt(beta, 1)), el('td', {}, fmt(ridgeShrink(beta, lambda), 3)), el('td', { class: lassoShrink(beta, lambda) === 0 ? 'zero' : '' }, fmt(lassoShrink(beta, lambda), 3))))));
    view.append(el('div', {}, el('h4', {}, `Ví dụ với λ = ${fmt(lambda, 2)}`), el('div', { class: 'table-scroll' }, table), el('p', { class: 'note' }, 'Ridge co hệ số lớn nhiều hơn theo giá trị tuyệt đối nhưng giữ nguyên tỉ lệ giữa chúng. Lasso trừ cùng một lượng λ cho mọi hệ số: hệ số lớn gần như giữ nguyên, hệ số nhỏ biến mất.')));
  }

  function drawPaths() {
    const index = lamIndex.value;
    const panel = (name: string, path: number[][]) => {
      const f = frame({ width: 380, height: 320, x: [-3, 1], y: [-2.6, 3.4], xTicks: [-3, -2, -1, 0, 1], yTicks: [-2, -1, 0, 1, 2, 3], xLabel: 'log₁₀ λ', yLabel: 'hệ số β', label: `Đường đi hệ số của ${name}`, format: value => (Math.abs(value) < 1e-9 ? '0' : String(value).replace('-', '−')) });
      polyline(f, [[-3, 0], [1, 0]], 'axis-zero');
      FEATURES.forEach((feature, j) => {
        const cls = j < 2 ? 'coef pair' : SIGNAL[j] ? 'coef signal' : 'coef';
        polyline(f, LAMBDAS.map((value, i) => [Math.log10(value), path[i][j]] as Point), cls);
        if (j < 3 || j === 5) label(f, [-3, path[0][j]], feature, 'end-label', 4, j === 0 ? 12 : -4);
      });
      const x = Math.log10(LAMBDAS[index]);
      polyline(f, [[x, -2.6], [x, 3.4]], 'vline');
      dots(f, path[index].map(value => [x, value] as Point), 'visit', 3);
      return f;
    };
    view.append(el('div', { class: 'legend' }, el('span', {}, el('span', { class: 'key-line tone-1' }), 'x₁, x₂ (gần trùng nhau)'), el('span', {}, el('span', { class: 'key-line' }), 'x₃, x₆ (có ích)'), el('span', {}, el('span', { class: 'key-line', style: 'border-color: var(--color-muted)' }), 'x₄, x₅, x₇, x₈ (vô ích, hệ số thật = 0)')));
    view.append(el('div', { class: 'duo' },
      el('figure', { class: 'opt-figure' }, el('h4', {}, 'Ridge (L2)'), panel('Ridge', ridgePath).root as unknown as Node),
      el('figure', { class: 'opt-figure' }, el('h4', {}, 'Lasso (L1)'), panel('Lasso', lassoPath).root as unknown as Node)));
    const table = el('table', { class: 'data compact' });
    table.append(el('thead', {}, el('tr', {}, el('th', { scope: 'col' }, 'Đặc trưng'), el('th', { scope: 'col' }, 'Hệ số thật'), el('th', { scope: 'col' }, 'Không phạt'), el('th', { scope: 'col' }, 'Ridge'), el('th', { scope: 'col' }, 'Lasso'))));
    table.append(el('tbody', {}, ...FEATURES.map((feature, j) => el('tr', {}, el('td', {}, feature), el('td', {}, fmt(TRUE_BETA[j], 1)), el('td', {}, fmt(ols[j])), el('td', {}, fmt(ridgePath[index][j])), el('td', { class: Math.abs(lassoPath[index][j]) < 1e-8 ? 'zero' : '' }, Math.abs(lassoPath[index][j]) < 1e-8 ? '0' : fmt(lassoPath[index][j]))))));
    view.append(el('div', { class: 'duo' }, el('div', { class: 'table-scroll' }, table), el('dl', { class: 'answer stack' },
      el('div', {}, el('dt', {}, `Hệ số khác 0 tại ${LABEL(index)}`), el('dd', {}, `Ridge ${nonZero(ridgePath[index])} / 8 · Lasso ${nonZero(lassoPath[index])} / 8`)),
      el('div', {}, el('dt', {}, 'Lỗi huấn luyện / kiểm tra (Ridge)'), el('dd', {}, `${fmt(mse(ridgePath[index], TRAIN))} / ${fmt(ridgeTest[index])}`)),
      el('div', {}, el('dt', {}, 'Lỗi huấn luyện / kiểm tra (Lasso)'), el('dd', {}, `${fmt(mse(lassoPath[index], TRAIN))} / ${fmt(lassoTest[index])}`)))));
  }

  function drawCorrelated() {
    const index = lamIndex.value, r = ridgePath[index], l = lassoPath[index];
    const rows: { name: string; beta: number[] }[] = [{ name: 'Không phạt', beta: ols }, { name: 'Ridge', beta: r }, { name: 'Lasso', beta: l }];
    const max = Math.max(3.2, ...rows.flatMap(row => [Math.abs(row.beta[0]), Math.abs(row.beta[1])]));
    view.append(el('div', { class: 'duo' },
      el('div', {}, el('h4', {}, `Hệ số của x₁ và x₂ tại ${LABEL(index)} (thật: 1.5 và 1.5)`), bars(rows.flatMap(row => [
        { label: `${row.name} · x₁`, value: Math.abs(row.beta[0]), text: fmt(row.beta[0]), tone: 1 as const },
        { label: `${row.name} · x₂`, value: Math.abs(row.beta[1]), text: fmt(row.beta[1]), tone: 2 as const },
      ]), max, 'Hệ số của hai đặc trưng tương quan'), el('p', { class: 'hint' }, 'Thanh xanh: x₁. Thanh cam: x₂. Độ dài là |β|, số bên phải có dấu.')),
      el('div', {}, el('dl', { class: 'answer stack' }, ...rows.map(row => el('div', {}, el('dt', {}, `${row.name}: β₁ + β₂`), el('dd', {}, `${fmt(row.beta[0] + row.beta[1])}  (β₁ = ${fmt(row.beta[0])}, β₂ = ${fmt(row.beta[1])})`)))),
        el('p', { class: 'note' }, 'Tổng β₁ + β₂ thay đổi ít hơn nhiều so với từng hệ số: dữ liệu xác định khá rõ tổng ảnh hưởng của cặp đặc trưng, nhưng gần như không xác định được cách chia. Không phạt chia rất lệch và sẽ đổi hẳn nếu đổi vài mẫu. Ridge ưu tiên chia đều vì 1.5² + 1.5² < 0² + 3². Lasso thấy |1.5| + |1.5| = |0| + |3| nên chọn lời giải thưa: giữ một đặc trưng.'))));
  }

  onResize(chartSvg.parentElement!, () => draw());
  draw();
}

function axes(f: Frame) {
  polyline(f, [[-10, 0], [10, 0]], 'axis-zero');
  polyline(f, [[0, -10], [0, 10]], 'axis-zero');
}

export const initRegularizationLab = () => boot('regularization', start);
