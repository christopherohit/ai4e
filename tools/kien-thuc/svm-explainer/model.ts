// Support vector machine lab model: three small 2D datasets and an exact SMO solver (Platt 1998) for the soft-margin
// dual, with linear and RBF kernels. Labels are −1 / +1; points are in a 0–10 box.
import { seeded } from '../../_shared/sequence-lab/math';

export type Point = readonly [number, number];
export interface Sample { x: Point; y: -1 | 1 }

const random = seeded(31);
const gaussian = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
const blob = (cx: number, cy: number, spread: number, count: number, y: -1 | 1): Sample[] => Array.from({ length: count }, () => ({ x: [round(cx + spread * gaussian()), round(cy + spread * gaussian())] as Point, y }));
function round(value: number) { return Math.round(value * 100) / 100; }

/** Linearly separable: two clusters with a clear corridor between them. */
const labelled = (y: -1 | 1, points: Point[]): Sample[] => points.map(x => ({ x, y }));
export const SEPARABLE: Sample[] = [
  ...labelled(-1, [[2, 6.5], [3, 8], [1.5, 8.5], [3.5, 6], [2.5, 4.8], [1, 6], [4, 8.6], [2.2, 3.6]]),
  ...labelled(1, [[6.5, 3], [7.5, 4.5], [8, 2], [6, 1.5], [8.8, 5.5], [5.6, 3.6], [7.2, 6.2], [9, 3.4]]),
];

/** Overlapping clusters: no line separates them, so the margin has to be soft. */
export const OVERLAP: Sample[] = [...blob(4, 5.6, 1.35, 18, -1), ...blob(6, 4.4, 1.35, 18, 1)];

/** A ring around a core: no line works at all, an RBF kernel does. */
function ring(count: number, radius: number, noise: number, y: -1 | 1, offset: number): Sample[] {
  return Array.from({ length: count }, (_, index) => {
    const angle = (index + offset * random()) / count * 2 * Math.PI, r = radius + noise * gaussian();
    return { x: [round(5 + r * Math.cos(angle)), round(5 + r * Math.sin(angle))] as Point, y };
  });
}
export const RINGS: Sample[] = [...ring(16, 1.4, 0.6, -1, 0.8), ...ring(26, 3.4, 0.55, 1, 0.8)];
/** A second draw from the same ring distribution, used as a test set for the kernel step. */
export const RINGS_TEST: Sample[] = [...ring(40, 1.4, 0.6, -1, 1), ...ring(60, 3.4, 0.55, 1, 1)];

export type Kernel = { type: 'linear' } | { type: 'rbf'; gamma: number };
export function kernel(k: Kernel, a: Point, b: Point) {
  if (k.type === 'linear') return a[0] * b[0] + a[1] * b[1];
  return Math.exp(-k.gamma * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2));
}

export interface Model { data: Sample[]; alpha: number[]; b: number; C: number; kernel: Kernel; iterations: number }

/**
 * SMO for  max Σα − ½ΣΣ αᵢαⱼ yᵢyⱼ K(xᵢ, xⱼ)  s.t. 0 ≤ αᵢ ≤ C, Σ αᵢyᵢ = 0. Each step optimises one pair analytically;
 * the second index is the one with the largest |Eᵢ − Eⱼ|, falling back to every other index. Deterministic.
 */
export function train(data: Sample[], C: number, k: Kernel = { type: 'linear' }, tol = 1e-4, maxSweeps = 400): Model {
  const n = data.length, alpha = Array(n).fill(0);
  let b = 0;
  const K = data.map(p => data.map(q => kernel(k, p.x, q.x)));
  const f = (i: number) => alpha.reduce((sum, a, j) => sum + a * data[j].y * K[j][i], 0) + b;
  const step = (i: number, j: number) => {
    if (i === j) return false;
    const [yi, yj] = [data[i].y, data[j].y];
    const [Ei, Ej] = [f(i) - yi, f(j) - yj];
    const [ai, aj] = [alpha[i], alpha[j]];
    const [L, H] = yi !== yj ? [Math.max(0, aj - ai), Math.min(C, C + aj - ai)] : [Math.max(0, ai + aj - C), Math.min(C, ai + aj)];
    if (H - L < 1e-12) return false;
    const eta = 2 * K[i][j] - K[i][i] - K[j][j];
    if (eta >= -1e-12) return false;
    const ajNew = Math.min(H, Math.max(L, aj - yj * (Ei - Ej) / eta));
    if (Math.abs(ajNew - aj) < 1e-9 * (ajNew + aj + 1e-9)) return false;
    const aiNew = ai + yi * yj * (aj - ajNew);
    const b1 = b - Ei - yi * (aiNew - ai) * K[i][i] - yj * (ajNew - aj) * K[i][j];
    const b2 = b - Ej - yi * (aiNew - ai) * K[i][j] - yj * (ajNew - aj) * K[j][j];
    alpha[i] = aiNew; alpha[j] = ajNew;
    b = aiNew > 1e-9 && aiNew < C - 1e-9 ? b1 : ajNew > 1e-9 && ajNew < C - 1e-9 ? b2 : (b1 + b2) / 2;
    return true;
  };
  let sweeps = 0;
  for (; sweeps < maxSweeps; sweeps++) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const Ei = f(i) - data[i].y, r = Ei * data[i].y;
      if (!((r < -tol && alpha[i] < C) || (r > tol && alpha[i] > 0))) continue;
      const order = [...Array(n).keys()].map(j => ({ j, gap: Math.abs(Ei - (f(j) - data[j].y)) })).sort((p, q) => q.gap - p.gap || p.j - q.j);
      if (order.some(({ j }) => step(i, j))) changed++;
    }
    if (!changed) break;
  }
  return { data, alpha, b, C, kernel: k, iterations: sweeps };
}

export const decision = (m: Model, x: Point) => m.alpha.reduce((sum, a, i) => (a > 1e-9 ? sum + a * m.data[i].y * kernel(m.kernel, m.data[i].x, x) : sum), 0) + m.b;

/** Linear kernel only: w = Σ αᵢ yᵢ xᵢ. */
export const weights = (m: Model): Point => m.alpha.reduce<[number, number]>((w, a, i) => [w[0] + a * m.data[i].y * m.data[i].x[0], w[1] + a * m.data[i].y * m.data[i].x[1]], [0, 0]);
/** Width of the band |f(x)| ≤ 1 for a linear model: 2 / ‖w‖. */
export const marginWidth = (m: Model) => 2 / Math.hypot(...weights(m));

export type Role = 'free' | 'margin' | 'bound';
/** free: αᵢ = 0 (ignored); margin: 0 < αᵢ < C, exactly on the margin; bound: αᵢ = C, inside the margin or misclassified. */
export function roles(m: Model): Role[] {
  return m.alpha.map(a => (a < 1e-6 ? 'free' : a > m.C * (1 - 1e-6) ? 'bound' : 'margin'));
}
export const supportCount = (m: Model) => roles(m).filter(role => role !== 'free').length;
export const errors = (m: Model, data: Sample[] = m.data) => data.filter(p => Math.sign(decision(m, p.x)) !== p.y).length;
/** Points with y·f(x) < 1: inside the margin or on the wrong side. */
export const violations = (m: Model) => m.data.filter(p => p.y * decision(m, p.x) < 1 - 1e-3).length;

/**
 * Step 1: for a line whose normal points at angle θ, the widest corridor between the classes is the gap between the
 * lowest projection of class +1 and the highest of class −1. Negative gap = this direction cannot separate them.
 */
export function corridor(data: Sample[], theta: number) {
  const normal: Point = [Math.cos(theta), Math.sin(theta)];
  const projection = (p: Sample) => p.x[0] * normal[0] + p.x[1] * normal[1];
  const low = Math.min(...data.filter(p => p.y === 1).map(projection));
  const high = Math.max(...data.filter(p => p.y === -1).map(projection));
  return { normal, offset: (low + high) / 2, width: low - high };
}

export const C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100] as const;
export const GAMMA_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30] as const;
export const HARD_C = 1e4;
