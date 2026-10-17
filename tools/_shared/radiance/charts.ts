import { scaleLinear } from 'd3-scale';
import { line, curveMonotoneX, curveStepAfter } from 'd3-shape';

// One chart, three series on a shared 0–1 axis. Colours come from CSS roles (--series-1..3) so light/dark swap in
// one place; identity is never colour-alone: every series is direct-labelled and the values are in the table below.
export interface Series { key: string; label: string; values: number[]; step?: boolean }
export interface ChartSpec {
  series: Series[];
  xLabels: [string, string];
  cursor?: number;
  caption: string;
  onPick?: (index: number) => void;
}

const namespace = 'http://www.w3.org/2000/svg';

export function drawChart(svg: SVGSVGElement, spec: ChartSpec) {
  svg.replaceChildren();
  // Size the viewBox to the rendered width so 12–13px text stays 12–13px on phones (no shrink-to-fit).
  const width = Math.max(320, Math.round(svg.parentElement?.clientWidth || 960));
  const compact = width < 560;
  const height = compact ? 200 : 220;
  const margin = { top: 14, right: compact ? 104 : 140, bottom: 30, left: 34 };
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const add = (tag: string, attributes: Record<string, string>, text?: string, parent: Element = svg) => {
    const element = document.createElementNS(namespace, tag);
    for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
    if (text !== undefined) element.textContent = text;
    parent.append(element); return element;
  };
  const count = Math.max(1, ...spec.series.map(series => series.values.length));
  const x = scaleLinear().domain([0, Math.max(1, count - 1)]).range([margin.left, width - margin.right]);
  const y = scaleLinear().domain([0, 1]).range([height - margin.bottom, margin.top]);
  for (const tick of [0, 0.5, 1]) {
    add('line', { x1: String(margin.left), x2: String(width - margin.right), y1: String(y(tick)), y2: String(y(tick)), class: 'grid' });
    add('text', { x: String(margin.left - 8), y: String(y(tick) + 4), class: 'tick', 'text-anchor': 'end' }, tick.toFixed(1));
  }
  add('text', { x: String(margin.left), y: String(height - 8), class: 'tick' }, spec.xLabels[0]);
  add('text', { x: String(width - margin.right), y: String(height - 8), class: 'tick', 'text-anchor': 'end' }, spec.xLabels[1]);
  const ends: { y: number; label: string; index: number }[] = [];
  spec.series.forEach((series, index) => {
    if (!series.values.length) return;
    const path = line<number>().x((_, position) => x(position)).y(value => y(Math.max(0, Math.min(1, value)))).curve(series.step ? curveStepAfter : curveMonotoneX)(series.values);
    if (path) add('path', { d: path, class: `series series-${index + 1}` });
    ends.push({ y: y(Math.max(0, Math.min(1, series.values.at(-1)!))), label: series.label, index });
  });
  // Direct labels at the right edge, nudged apart so they never collide.
  ends.sort((first, second) => first.y - second.y);
  for (let index = 1; index < ends.length; index++) ends[index].y = Math.max(ends[index].y, ends[index - 1].y + 16);
  for (const end of ends) {
    add('line', { x1: String(width - margin.right + 6), x2: String(width - margin.right + 18), y1: String(end.y), y2: String(end.y), class: `key series-${end.index + 1}` });
    add('text', { x: String(width - margin.right + 22), y: String(end.y + 4), class: 'direct-label' }, end.label);
  }
  if (spec.cursor !== undefined) add('line', { x1: String(x(spec.cursor)), x2: String(x(spec.cursor)), y1: String(margin.top - 4), y2: String(height - margin.bottom), class: 'cursor' });
  // Crosshair + tooltip layer: the whole plot is the hit target; the readout lists every series at that x.
  const hover = add('g', { class: 'hover', visibility: 'hidden' });
  const hairline = add('line', { y1: String(margin.top - 4), y2: String(height - margin.bottom), class: 'hairline' }, undefined, hover);
  const target = add('rect', { x: String(margin.left), y: String(margin.top), width: String(width - margin.left - margin.right), height: String(height - margin.top - margin.bottom), class: 'hit', tabindex: '-1' });
  const tooltip = svg.parentElement?.querySelector<HTMLElement>('[data-chart-tooltip]');
  const nearest = (event: PointerEvent) => {
    const bounds = svg.getBoundingClientRect();
    const local = (event.clientX - bounds.left) / bounds.width * width;
    return Math.max(0, Math.min(count - 1, Math.round(x.invert(local))));
  };
  target.addEventListener('pointermove', event => {
    const index = nearest(event as PointerEvent);
    hover.setAttribute('visibility', 'visible'); hairline.setAttribute('x1', String(x(index))); hairline.setAttribute('x2', String(x(index)));
    if (tooltip) {
      tooltip.hidden = false;
      tooltip.replaceChildren();
      for (const series of spec.series) {
        if (index >= series.values.length) continue;
        const row = document.createElement('div');
        const key = document.createElement('i'); key.className = `key series-${spec.series.indexOf(series) + 1}`;
        const value = document.createElement('strong'); value.textContent = series.values[index].toFixed(3);
        const label = document.createElement('span'); label.textContent = series.label;
        row.append(key, value, label); tooltip.append(row);
      }
      const bounds = svg.getBoundingClientRect();
      tooltip.style.left = `${Math.min(bounds.width - 160, Math.max(0, x(index) / width * bounds.width + 12))}px`;
    }
  });
  target.addEventListener('pointerleave', () => { hover.setAttribute('visibility', 'hidden'); if (tooltip) tooltip.hidden = true; });
  if (spec.onPick) target.addEventListener('click', event => spec.onPick!(nearest(event as PointerEvent)));
}

// Accessible table view of the same numbers (always available, not hidden behind hover).
export function fillTable(table: HTMLTableElement, headers: string[], rows: (string | number)[][]) {
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  for (const header of headers) { const cell = document.createElement('th'); cell.scope = 'col'; cell.textContent = header; head.append(cell); }
  const body = table.createTBody();
  for (const row of rows) {
    const element = body.insertRow();
    for (const value of row) element.insertCell().textContent = typeof value === 'number' ? value.toFixed(4) : value;
  }
}
