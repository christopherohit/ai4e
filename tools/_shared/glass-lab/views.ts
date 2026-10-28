// Views shared by the regression-based labs (gradient descent, mini-batch SGD): parameter-space loss maps, animated
// trails and loss-vs-step charts.
import type { Frame } from './engine';
import { C, Plot, alpha, clamp, fitEqual, ramp, vn, type Point, type Rect } from './gfx';
import type { LinearProblem, Theta } from './regression';

export interface View { w: readonly [number, number]; b: readonly [number, number] }

/** Position along a path at a fractional step, so animations glide between iterations. */
export function at(path: readonly Theta[], k: number): Theta {
  const i = Math.min(path.length - 1, Math.max(0, Math.floor(k))), j = Math.min(path.length - 1, i + 1), f = k - Math.floor(k);
  return [path[i][0] + (path[j][0] - path[i][0]) * f, path[i][1] + (path[j][1] - path[i][1]) * f];
}

/** Loss → [0, 1] height with a log curve, so the valley floor and the steep walls are both visible. */
export const heightOf = (p: LinearProblem, scale: number, cap: number) => (theta: Theta) => clamp(Math.log1p((p.loss(theta) - p.minLoss) / scale) / Math.log1p(cap / scale));

export function ticks([a, b]: readonly [number, number], count = 5) {
  const raw = (b - a) / count, power = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 2.5, 5, 10].map(m => m * power).find(s => s >= raw)!;
  const out: number[] = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
}
const tickText = (step: number) => (v: number) => vn(v, step >= 1 ? 0 : step >= 0.1 ? 1 : 2);

/** Parameter-space map: dark heat field, glowing iso-lines and the optimum marker. */
export function lossMap(f: Frame, rect: Rect, p: LinearProblem, view: View, height: (theta: Theta) => number, title: string, labels: [string, string], equal = false) {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, title, C.violet);
  const pad = { l: 46, r: 16, t: 40, b: 36 };
  let xr: Point = [view.w[0], view.w[1]], yr: Point = [view.b[0], view.b[1]];
  // Same data units per pixel on both axes, so a round bowl looks round.
  if (equal) [xr, yr] = fitEqual(rect, pad, xr, yr);
  const plot = new Plot(g, rect, xr, yr, pad);
  plot.heat((w, b) => height([w, b]), v => { const m = ramp(v).match(/\d+/g)!.map(Number); return [m[0] * 0.55, m[1] * 0.55, m[2] * 0.6, 235]; }, 80);
  plot.contours((w, b) => height([w, b]), [0.06, 0.14, 0.24, 0.35, 0.47, 0.6, 0.73, 0.86], v => ramp(v + 0.12, 0.75), { width: 1.1 });
  const xt = ticks(xr), yt = ticks(yr);
  plot.axes({ xTicks: xt, yTicks: yt, xLabel: labels[0], yLabel: labels[1], fx: tickText(xt[1] - xt[0]), fy: tickText(yt[1] - yt[0]) });
  const [ox, oy] = plot.p(p.optimum);
  g.circle(ox, oy, 6, null, { stroke: C.white, width: 2 });
  g.circle(ox, oy, 2, C.white);
  return plot;
}

/** Animated path up to fractional step k with a glowing head. A run that leaves the map is cut at the exit. */
export function trail(plot: Plot, path: readonly Theta[], k: number, color: string, options: { dots?: boolean; width?: number; head?: number } = {}) {
  let shown = [...path.slice(0, Math.floor(k) + 1), at(path, k)];
  const out = shown.findIndex(p => !plot.inside(plot.px(p[0]), plot.py(p[1])));
  if (out > 0) {
    shown = shown.slice(0, out + 1);
    const [x, y] = plot.p(shown[out - 1]);
    plot.clip(() => plot.line(shown, color, { width: 2, glow: 10 }));
    plot.g.tag('văng ra ngoài', clamp(x, plot.box.x + 50, plot.box.x + plot.box.w - 50), clamp(y, plot.box.y + 30, plot.box.y + plot.box.h - 14), color);
    return;
  }
  plot.clip(() => {
    plot.line(shown, color, { width: options.width ?? 2.2, glow: 10 });
    if (options.dots ?? true) shown.slice(0, -1).forEach(p => plot.g.circle(plot.px(p[0]), plot.py(p[1]), 2.4, color, { alpha: 0.85 }));
    const [x, y] = plot.p(at(path, k));
    plot.g.circle(x, y, options.head ?? 7, color, { glow: 22 });
    plot.g.circle(x, y, 3, C.white);
  });
}

export interface Series { values: readonly number[]; color: string; x?: readonly number[] }
/** Loss chart with a cursor; values above yMax are clipped (divergence). x defaults to the step index. */
export function lossChart(f: Frame, rect: Rect, series: Series[], cursor: number, yMax: number, minLoss: number, title = 'MSE theo từng bước', xLabel = 'bước', xMax?: number, xFormat?: (v: number) => string) {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, title, C.amber);
  const steps = xMax ?? Math.max(...series.map(s => (s.x ? s.x.at(-1)! : s.values.length - 1)));
  const plot = new Plot(g, rect, [0, steps], [0, yMax], { l: 42, r: 14, t: 40, b: 30 });
  const xt = ticks([0, steps]), yt = ticks([0, yMax], 4);
  plot.axes({ xTicks: xt, yTicks: yt, xLabel, fx: xFormat ?? tickText(xt[1] - xt[0]), fy: tickText(yt[1] - yt[0]) });
  plot.line([[0, minLoss], [steps, minLoss]], alpha(C.white, 0.35), { width: 1, dash: [4, 4] });
  g.text(`tối thiểu ${vn(minLoss, minLoss < 10 ? 3 : 1)}`, plot.box.x + plot.box.w - 4, plot.py(minLoss) - 9, { size: 10.5, color: C.muted, align: 'right' });
  plot.clip(() => {
    for (const s of series) {
      const xs = s.x ?? s.values.map((_, i) => i);
      const shown: Point[] = [];
      for (let i = 0; i < s.values.length && xs[i] <= cursor + 1e-9; i++) shown.push([xs[i], Math.min(s.values[i], yMax * 1.5)]);
      plot.line(shown, s.color, { width: 2, glow: 8 });
      const last = shown.at(-1);
      if (last) g.circle(plot.px(last[0]), plot.py(Math.min(last[1], yMax)), 3.5, s.color, { glow: 10 });
    }
  });
  return plot;
}

export interface CategorySeries { label: string; values: readonly number[]; color: string; format?: (v: number) => string }
/** Lines over categorical x values (a parameter grid), with a highlighted current category and direct labels. */
export function categoryChart(f: Frame, rect: Rect, title: string, categories: readonly string[], series: CategorySeries[], y: [number, number], current: number | null, reveal = 1, yFormat: (v: number) => string = v => vn(v, 0), xLabel = '') {
  const { g } = f;
  g.panel(rect.x, rect.y, rect.w, rect.h, title, C.amber);
  const n = categories.length;
  const plot = new Plot(g, rect, [-0.4, n - 0.6], y, { l: 44, r: 16, t: 40, b: 30 });
  plot.axes({ yTicks: ticks(y, 4), fy: yFormat, xLabel });
  categories.forEach((label, i) => g.text(label, plot.px(i), plot.box.y + plot.box.h + 13, { size: 10.5, color: i === current ? C.ink : C.muted, align: 'center', weight: i === current ? 650 : 500 }));
  if (current !== null) g.line([[plot.px(current), plot.box.y], [plot.px(current), plot.box.y + plot.box.h]], alpha(C.white, 0.35), { width: 1, dash: [3, 3] });
  const shownCount = Math.max(1, Math.round(n * clamp(reveal)));
  series.forEach((s, k) => {
    const pts = s.values.slice(0, shownCount).map((v, i) => [i, v] as Point);
    plot.clip(() => plot.line(pts, s.color, { width: 2.2, glow: 8 }));
    pts.forEach(([i, v]) => g.circle(plot.px(i), plot.py(v), i === current ? 5 : 3, s.color, { glow: i === current ? 14 : 0 }));
    const last = pts.at(-1)!;
    g.text(s.label, plot.box.x + 8, plot.box.y + 12 + k * 16, { size: 11, color: s.color, weight: 650 });
    if (current !== null && current < shownCount) g.text((s.format ?? yFormat)(s.values[current]), plot.px(current) + 7, plot.py(s.values[current]) - 9, { size: 11, color: s.color, mono: true });
    void last;
  });
  return plot;
}
