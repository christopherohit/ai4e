// DOM helpers shared by the sequence labs: lab boot, step bar, step card, heat tables, bar lists and a small line chart.
export type Root = HTMLElement;
export type StepText = readonly [title: string, description: string, formula: string];

export function boot(id: string, start: (root: Root) => void) {
  const run = () => document.querySelectorAll<HTMLElement>(`[data-seq-lab="${id}"]`).forEach(root => {
    if (root.dataset.initialized) return;
    root.dataset.initialized = 'true';
    try { start(root); } finally { root.setAttribute('aria-busy', 'false'); }
  });
  run();
  document.addEventListener('astro:page-load', run);
}

export const $ = <T extends Element = HTMLElement>(root: ParentNode, selector: string) => root.querySelector<T>(selector)!;
export const $$ = <T extends Element = HTMLElement>(root: ParentNode, selector: string) => [...root.querySelectorAll<T>(selector)];

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, attributes: Record<string, string> = {}, ...children: (Node | string | null | undefined | false)[]) {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (name === 'class') element.className = value; else element.setAttribute(name, value);
  }
  for (const child of children) if (child !== null && child !== undefined && child !== false) element.append(child);
  return element;
}

/** Step bar: returns a setter; clicking a step calls onChange. */
export function wireSteps(root: Root, onChange: (stage: number) => void) {
  const buttons = $$<HTMLButtonElement>(root, '.steps [data-stage]');
  const set = (stage: number) => buttons.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.stage) === stage)));
  buttons.forEach(button => button.addEventListener('click', () => { const stage = Number(button.dataset.stage); set(stage); onChange(stage); }));
  return set;
}

export function setCard(root: Root, stage: number, total: number, [title, description, formula]: StepText) {
  $(root, '[data-step-number]').textContent = `Bước ${stage + 1} / ${total}`;
  $(root, '[data-step-title]').textContent = title;
  $(root, '[data-step-description]').textContent = description;
  const box = $(root, '[data-formula]');
  box.textContent = formula; box.hidden = !formula;
}

/** Binds a range input to its <output data-value="name">; returns a reader. */
export function slider(root: Root, name: string, format: (value: number) => string, onInput: () => void) {
  const input = $<HTMLInputElement>(root, `input[name="${name}"]`);
  const output = root.querySelector<HTMLOutputElement>(`[data-value="${name}"]`);
  const show = () => { if (output) output.textContent = format(Number(input.value)); };
  input.addEventListener('input', () => { show(); onInput(); });
  show();
  return { input, get value() { return Number(input.value); }, set(value: number) { input.value = String(value); show(); } };
}

export const fmt = (value: number, digits = 2) => (Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value).toFixed(digits).replace('-', '−');
export const percent = (value: number) => `${Math.round(value * 100)}%`;

export interface HeatSpec {
  caption: string;
  note?: string;
  columns: string[];
  rows: { label: string; values: (number | null)[]; focus?: boolean; changed?: boolean[] }[];
  /** Symmetric colour range: |value| ≥ range is full strength. */
  range: number;
  digits?: number;
}

/** A matrix (or stacked vectors) as a table: every cell prints its value, colour encodes sign and magnitude. */
export function heatTable(spec: HeatSpec) {
  const table = el('table', { class: 'heat' });
  table.append(el('caption', {}, spec.caption, spec.note ? el('small', {}, spec.note) : null));
  const head = el('tr', {}, el('th', { scope: 'col' }, ''));
  for (const column of spec.columns) head.append(el('th', { scope: 'col', title: column }, column));
  table.append(el('thead', {}, head));
  const body = el('tbody');
  for (const row of spec.rows) {
    if (!row.values.length) { body.append(el('tr', { class: 'gap', 'aria-hidden': 'true' }, el('th'), el('td', { colspan: String(spec.columns.length) }))); continue; }
    const tr = el('tr', { class: row.focus ? 'focus' : '' }, el('th', { scope: 'row', title: row.label }, row.label));
    row.values.forEach((value, index) => {
      if (value === null) { tr.append(el('td', {}, '')); return; }
      const strength = Math.min(1, Math.abs(value) / spec.range);
      const cell = el('td', { class: [value >= 0 ? 'pos' : 'neg', strength > 0.55 ? 'strong' : '', row.changed?.[index] ? 'changed' : ''].join(' ').trim() }, fmt(value, spec.digits ?? 2));
      cell.style.setProperty('--w', `${Math.round(strength * 85)}%`);
      tr.append(cell);
    });
    body.append(tr);
  }
  table.append(body);
  return table;
}

export interface Bar { label: string; value: number; text?: string; tone?: 0 | 1 | 2 | 3; on?: boolean }
export function bars(items: Bar[], max = Math.max(1e-9, ...items.map(item => Math.abs(item.value))), label?: string) {
  const list = el('ul', { class: 'bars', ...(label ? { 'aria-label': label } : {}) });
  for (const item of items) {
    const fill = el('span', { class: `fill tone-${item.tone ?? 1}` });
    fill.style.width = `${Math.max(0, Math.min(1, item.value / max)) * 100}%`;
    list.append(el('li', { class: item.on ? 'on' : '' }, el('span', { class: 'label', title: item.label }, item.label), el('span', { class: 'track' }, fill), el('span', { class: 'value' }, item.text ?? fmt(item.value))));
  }
  return list;
}

const svgNS = 'http://www.w3.org/2000/svg';
export function svg<K extends keyof SVGElementTagNameMap>(tag: K, attributes: Record<string, string | number> = {}, text?: string) {
  const element = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value));
  if (text !== undefined) element.textContent = text;
  return element;
}

export interface LineSeries { label: string; values: number[]; tone: 1 | 2 | 3 }
export interface LineSpec {
  series: LineSeries[];
  xLabels: string[];
  yMax: number;
  yTicks: number[];
  yFormat: (value: number) => string;
  cursor?: number;
  onPick?: (index: number) => void;
  /** Log-free x positions in [0,1]; defaults to evenly spaced. */
  xPositions?: number[];
}

/** Discrete-point line chart with direct labels; values are also given in a table elsewhere. */
export function lineChart(target: SVGSVGElement, spec: LineSpec) {
  target.replaceChildren();
  const width = Math.max(320, Math.round(target.parentElement?.clientWidth || 900));
  const compact = width < 560;
  const height = compact ? 190 : 210;
  const margin = { top: 14, right: compact ? 112 : 160, bottom: 30, left: 52 };
  target.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const count = spec.xLabels.length;
  const position = (index: number) => spec.xPositions?.[index] ?? (count > 1 ? index / (count - 1) : 0.5);
  const x = (index: number) => margin.left + position(index) * (width - margin.left - margin.right);
  const y = (value: number) => margin.top + (1 - Math.max(0, Math.min(spec.yMax, value)) / spec.yMax) * (height - margin.top - margin.bottom);
  for (const tick of spec.yTicks) {
    target.append(svg('line', { x1: margin.left, x2: width - margin.right, y1: y(tick), y2: y(tick), class: 'grid' }));
    target.append(svg('text', { x: margin.left - 8, y: y(tick) + 4, class: 'tick', 'text-anchor': 'end' }, spec.yFormat(tick)));
  }
  const every = Math.max(1, Math.ceil(count / (compact ? 5 : 10)));
  spec.xLabels.forEach((label, index) => {
    if (index % every && index !== count - 1) return;
    target.append(svg('text', { x: x(index), y: height - 8, class: 'tick', 'text-anchor': 'middle' }, label));
  });
  if (spec.cursor !== undefined) target.append(svg('line', { x1: x(spec.cursor), x2: x(spec.cursor), y1: margin.top - 4, y2: height - margin.bottom, class: 'cursor' }));
  const ends: { y: number; label: string; tone: number }[] = [];
  for (const series of spec.series) {
    const points = series.values.map((value, index) => `${x(index)},${y(value)}`).join(' ');
    target.append(svg('polyline', { points, class: `series tone-${series.tone}` }));
    if (count <= 40) series.values.forEach((value, index) => target.append(svg('circle', { cx: x(index), cy: y(value), r: index === spec.cursor ? 4.5 : 3, class: `dot tone-${series.tone}` })));
    ends.push({ y: y(series.values.at(-1) ?? 0), label: series.label, tone: series.tone });
  }
  ends.sort((first, second) => first.y - second.y);
  for (let index = 1; index < ends.length; index++) ends[index].y = Math.max(ends[index].y, ends[index - 1].y + 16);
  for (const end of ends) {
    target.append(svg('line', { x1: width - margin.right + 8, x2: width - margin.right + 20, y1: end.y, y2: end.y, class: `key tone-${end.tone}` }));
    target.append(svg('text', { x: width - margin.right + 24, y: end.y + 4, class: 'direct-label' }, end.label));
  }
  if (spec.onPick) {
    const hit = svg('rect', { x: margin.left - 6, y: margin.top, width: width - margin.left - margin.right + 12, height: height - margin.top - margin.bottom, class: 'hit' });
    hit.addEventListener('click', event => {
      const bounds = target.getBoundingClientRect();
      const local = ((event as MouseEvent).clientX - bounds.left) / bounds.width * width;
      let best = 0;
      for (let index = 1; index < count; index++) if (Math.abs(x(index) - local) < Math.abs(x(best) - local)) best = index;
      spec.onPick!(best);
    });
    target.append(hit);
  }
}

export function fillTable(table: HTMLTableElement, headers: string[], rows: (string | number)[][]) {
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  for (const header of headers) head.append(el('th', { scope: 'col' }, header));
  const body = table.createTBody();
  for (const row of rows) { const tr = body.insertRow(); for (const value of row) tr.insertCell().textContent = String(value); }
}

/** Re-runs draw when the stage resizes (charts size their viewBox to the rendered width). */
export function onResize(element: Element, draw: () => void) {
  let last = 0;
  new ResizeObserver(entries => { const width = Math.round(entries[0].contentRect.width); if (Math.abs(width - last) > 24) { last = width; draw(); } }).observe(element);
}
