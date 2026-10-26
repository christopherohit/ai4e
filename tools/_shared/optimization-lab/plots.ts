// SVG plots shared by the optimization labs: a framed plot with ticks and a clipped data layer, plus curves, paths,
// arrows, the MSE contour map and the data scatter. Data coordinates go in; pixel mapping stays inside the frame.
import { svg } from '../sequence-lab/ui';
import { DATA, MIN_LOSS, levelSet, type Point } from './model';

export interface Frame {
  root: SVGSVGElement;
  /** Clipped layer for data marks. */
  plot: SVGGElement;
  /** Unclipped layer for labels drawn on top. */
  top: SVGGElement;
  x: (value: number) => number;
  y: (value: number) => number;
  /** Data coordinates under a click. */
  pick: (event: MouseEvent) => Point;
}

interface FrameSpec { width: number; height: number; x: Point; y: Point; xTicks: number[]; yTicks: number[]; xLabel: string; yLabel: string; label: string; format?: (value: number) => string }

let clipId = 0;
const decimal = (value: number) => String(Math.round(value * 100) / 100).replace('-', '−');

export function frame(spec: FrameSpec): Frame {
  const margin = { top: 10, right: 12, bottom: 38, left: 46 };
  const { width, height } = spec;
  const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, class: 'opt-plot', role: 'img', 'aria-label': spec.label });
  const x = (value: number) => margin.left + (value - spec.x[0]) / (spec.x[1] - spec.x[0]) * (width - margin.left - margin.right);
  const y = (value: number) => height - margin.bottom - (value - spec.y[0]) / (spec.y[1] - spec.y[0]) * (height - margin.top - margin.bottom);
  const format = spec.format ?? decimal;
  for (const tick of spec.xTicks) {
    root.append(svg('line', { x1: x(tick), x2: x(tick), y1: margin.top, y2: height - margin.bottom, class: 'grid' }));
    root.append(svg('text', { x: x(tick), y: height - margin.bottom + 15, class: 'tick', 'text-anchor': 'middle' }, format(tick)));
  }
  for (const tick of spec.yTicks) {
    root.append(svg('line', { x1: margin.left, x2: width - margin.right, y1: y(tick), y2: y(tick), class: 'grid' }));
    root.append(svg('text', { x: margin.left - 6, y: y(tick) + 4, class: 'tick', 'text-anchor': 'end' }, format(tick)));
  }
  root.append(svg('rect', { x: margin.left, y: margin.top, width: width - margin.left - margin.right, height: height - margin.top - margin.bottom, class: 'frame' }));
  root.append(svg('text', { x: (margin.left + width - margin.right) / 2, y: height - 4, class: 'axis-label', 'text-anchor': 'middle' }, spec.xLabel));
  root.append(svg('text', { x: 0, y: 0, class: 'axis-label', 'text-anchor': 'middle', transform: `translate(12, ${(margin.top + height - margin.bottom) / 2}) rotate(-90)` }, spec.yLabel));
  const id = `opt-clip-${++clipId}`;
  const defs = svg('defs'), clip = svg('clipPath', { id });
  clip.append(svg('rect', { x: margin.left, y: margin.top, width: width - margin.left - margin.right, height: height - margin.top - margin.bottom }));
  defs.append(clip);
  root.append(defs);
  const plot = svg('g', { 'clip-path': `url(#${id})` });
  const top = svg('g');
  root.append(plot, top);
  const pick = (event: MouseEvent): Point => {
    const matrix = root.getScreenCTM();
    if (!matrix) return [spec.x[0], spec.y[0]];
    const local = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    return [
      spec.x[0] + (local.x - margin.left) / (width - margin.left - margin.right) * (spec.x[1] - spec.x[0]),
      spec.y[0] + (height - margin.bottom - local.y) / (height - margin.top - margin.bottom) * (spec.y[1] - spec.y[0]),
    ];
  };
  return { root, plot, top, x, y, pick };
}

const finite = ([a, b]: Point) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a) < 1e4 && Math.abs(b) < 1e4;
const pixels = (f: Frame, points: readonly Point[]) => points.filter(finite).map(([a, b]) => `${f.x(a).toFixed(1)},${f.y(b).toFixed(1)}`).join(' ');

export function polyline(f: Frame, points: readonly Point[], cls: string) {
  const line = svg('polyline', { points: pixels(f, points), class: cls });
  f.plot.append(line);
  return line;
}

export function dots(f: Frame, points: readonly Point[], cls: string, radius = 3) {
  for (const point of points.filter(finite)) f.plot.append(svg('circle', { cx: f.x(point[0]), cy: f.y(point[1]), r: radius, class: cls }));
}

/** Function graph sampled on [from, to]. */
export function curve(f: Frame, fn: (value: number) => number, from: number, to: number, cls = 'curve', samples = 160) {
  return polyline(f, Array.from({ length: samples + 1 }, (_, index) => { const value = from + (to - from) * index / samples; return [value, fn(value)] as Point; }), cls);
}

/** Arrow drawn in pixel space so the head keeps its size whatever the axis scales are. */
export function arrow(f: Frame, from: Point, to: Point, cls: string, layer: SVGGElement = f.plot) {
  if (!finite(from) || !finite(to)) return;
  const [x1, y1, x2, y2] = [f.x(from[0]), f.y(from[1]), f.x(to[0]), f.y(to[1])];
  const length = Math.hypot(x2 - x1, y2 - y1);
  const group = svg('g', { class: `arrow ${cls}` });
  group.append(svg('line', { x1, y1, x2, y2 }));
  if (length > 4) {
    const [ux, uy] = [(x2 - x1) / length, (y2 - y1) / length], head = Math.min(9, length * 0.6);
    group.append(svg('path', { d: `M${x2},${y2} L${x2 - head * ux + head * 0.45 * uy},${y2 - head * uy - head * 0.45 * ux} L${x2 - head * ux - head * 0.45 * uy},${y2 - head * uy + head * 0.45 * ux} Z` }));
  }
  layer.append(group);
}

export function label(f: Frame, at: Point, text: string, cls = 'mark-label', dx = 6, dy = -6) {
  if (!finite(at)) return;
  f.top.append(svg('text', { x: f.x(at[0]) + dx, y: f.y(at[1]) + dy, class: cls }, text));
}

/** Contour levels of the MSE loss, as L* + c. Spacing grows so both the valley floor and the walls are visible. */
export const LEVELS = [0.03, 0.1, 0.25, 0.5, 0.9, 1.5, 2.4, 3.6, 5.2, 7.5];

export const PARAMETER_VIEW = { x: [-1, 3.5] as Point, y: [-1, 3.2] as Point, xTicks: [-1, 0, 1, 2, 3], yTicks: [-1, 0, 1, 2, 3] };

/** Loss landscape over (w, b): elliptic level sets around θ*, darker towards the minimum. */
export function lossMap(name: string, width = 400, height = 340) {
  const f = frame({ width, height, ...PARAMETER_VIEW, xLabel: 'w (độ dốc của đường thẳng)', yLabel: 'b (hệ số chặn)', label: name });
  LEVELS.slice().reverse().forEach((level, index) => {
    const shape = polyline(f, levelSet(level), 'contour');
    shape.setAttribute('style', `--level:${Math.round(6 + index * 4)}%`);
  });
  LEVELS.slice(0, 6).forEach(level => {
    const [w, b] = levelSet(level, 8)[1];
    if (w < PARAMETER_VIEW.x[0] || w > PARAMETER_VIEW.x[1] - 0.3 || b < PARAMETER_VIEW.y[0] + 0.1 || b > PARAMETER_VIEW.y[1]) return;
    f.top.append(svg('text', { x: f.x(w) + 2, y: f.y(b) - 2, class: 'contour-label' }, decimal(MIN_LOSS + level)));
  });
  return f;
}

export const DATA_VIEW = { x: [0, 2.4] as Point, y: [-0.5, 5.5] as Point, xTicks: [0, 0.5, 1, 1.5, 2], yTicks: [0, 1, 2, 3, 4, 5] };

/** Training data with the current line ŷ = w·x + b; highlighted samples are the ones the current update used. */
export function dataPlot(name: string, theta: Point, highlight: readonly number[] = [], width = 400, height = 340) {
  const f = frame({ width, height, ...DATA_VIEW, xLabel: 'x: diện tích (trăm m²)', yLabel: 'y: giá (tỷ đồng)', label: name });
  const on = new Set(highlight);
  if (on.size && on.size < DATA.length) for (const index of on) polyline(f, [[DATA[index].x, DATA[index].y], [DATA[index].x, theta[0] * DATA[index].x + theta[1]]], 'residual');
  DATA.forEach((sample, index) => f.plot.append(svg('circle', { cx: f.x(sample.x), cy: f.y(sample.y), r: on.has(index) && on.size < DATA.length ? 4.5 : 3, class: on.has(index) && on.size < DATA.length ? 'sample on' : 'sample' })));
  polyline(f, [[DATA_VIEW.x[0], theta[0] * DATA_VIEW.x[0] + theta[1]], [DATA_VIEW.x[1], theta[0] * DATA_VIEW.x[1] + theta[1]]], 'fit');
  return f;
}

/** Class marker: circles for class 0, squares for class 1, so the classes never rely on colour alone. */
export function marker(f: Frame, [a, b]: Point, cls: string, square = false, size = 4.5, layer: SVGGElement = f.plot) {
  if (!finite([a, b])) return;
  const shape = square
    ? svg('rect', { x: f.x(a) - size * 0.85, y: f.y(b) - size * 0.85, width: size * 1.7, height: size * 1.7, rx: 1, class: cls })
    : svg('circle', { cx: f.x(a), cy: f.y(b), r: size, class: cls });
  layer.append(shape);
  return shape;
}

/** Filled band between a lower and an upper curve sampled at the same x values. */
export function band(f: Frame, xs: readonly number[], lower: readonly number[], upper: readonly number[], cls = 'band') {
  const outline: Point[] = [...xs.map((x, index) => [x, upper[index]] as Point), ...xs.map((x, index) => [x, lower[index]] as Point).reverse()];
  const shape = svg('polygon', { points: outline.filter(finite).map(([a, b]) => `${f.x(a).toFixed(1)},${f.y(b).toFixed(1)}`).join(' '), class: cls });
  f.plot.append(shape);
  return shape;
}

/** Coloured cells of a scalar field on the plot area: --a carries |value| / max, the class picks the hue. */
export function field(f: Frame, x: Point, y: Point, cells: number, value: (a: number, b: number) => number, max: number) {
  const dx = (x[1] - x[0]) / cells, dy = (y[1] - y[0]) / cells;
  const group = svg('g', { class: 'field' });
  for (let i = 0; i < cells; i++) for (let j = 0; j < cells; j++) {
    const a = x[0] + (i + 0.5) * dx, b = y[0] + (j + 0.5) * dy, v = value(a, b);
    const rect = svg('rect', { x: f.x(x[0] + i * dx), y: f.y(y[0] + (j + 1) * dy), width: f.x(x[0] + dx) - f.x(x[0]) + 0.6, height: f.y(y[0]) - f.y(y[0] + dy) + 0.6, class: v >= 0 ? 'cell-1' : 'cell-0' });
    rect.setAttribute('style', `--a:${Math.round(Math.min(1, Math.abs(v) / max) * 100)}%`);
    group.append(rect);
  }
  f.plot.append(group);
  return group;
}
