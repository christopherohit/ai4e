// SVM lab model on the Iris flowers (Fisher 1936): three two-feature problems, standardized like scikit-learn's
// StandardScaler, and an exact SMO solver (Platt 1998) for the soft-margin dual with linear and RBF kernels.
import data from './data/iris.json';

export type Point = readonly [number, number];
export interface Sample { x: Point; y: -1 | 1; raw: Point; species: number }
export const SOURCE = data.source;
export const FEATURES = data.features;
export const SPECIES = data.species;

export interface Problem { name: string; columns: [number, number]; data: Sample[]; mean: Point; sd: Point; positive: string; negative: string }
/** Rows where keep(species) holds; class +1 = `positive` species; features standardized on those rows. */
function problem(name: string, columns: [number, number], keep: (species: number) => boolean, positive: number): Problem {
  const rows = data.X.map((row, i) => ({ raw: [row[columns[0]], row[columns[1]]] as Point, species: data.y[i] })).filter(r => keep(r.species));
  const mean = [0, 1].map(k => rows.reduce((s, r) => s + r.raw[k], 0) / rows.length) as unknown as Point;
  const sd = [0, 1].map(k => Math.sqrt(rows.reduce((s, r) => s + (r.raw[k] - mean[k]) ** 2, 0) / rows.length)) as unknown as Point;
  const negativeNames = [...new Set(rows.filter(r => r.species !== positive).map(r => SPECIES[r.species]))].join(' + ');
  return { name, columns, mean, sd, positive: SPECIES[positive], negative: negativeNames, data: rows.map(r => ({ raw: r.raw, species: r.species, y: r.species === positive ? 1 : -1, x: [(r.raw[0] - mean[0]) / sd[0], (r.raw[1] - mean[1]) / sd[1]] as Point })) };
}
/** Setosa vs versicolor on the sepals: separable, with a narrow corridor. */
export const SEPARABLE = problem('separable', [0, 1], s => s < 2, 1);
/** Versicolor vs virginica on the petals: a few flowers overlap. */
export const OVERLAP = problem('overlap', [2, 3], s => s > 0, 2);
/** Versicolor vs the other two species on the petals: versicolor sits between them, no line can work. */
export const MIDDLE = problem('middle', [2, 3], () => true, 1);
export const toRaw = (p: Problem, [a, b]: Point): Point => [a * p.sd[0] + p.mean[0], b * p.sd[1] + p.mean[1]];
export const toStd = (p: Problem, [a, b]: Point): Point => [(a - p.mean[0]) / p.sd[0], (b - p.mean[1]) / p.sd[1]];
export const REFERENCE = data.reference;

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
  const n = data.length, alpha = new Array<number>(n).fill(0), y = data.map(p => p.y);
  const K = data.map(p => data.map(q => kernel(k, p.x, q.x)));
  // Error cache E_i = f(x_i) − y_i, updated in O(n) after every pair step instead of recomputing f from scratch.
  const E = y.map(v => -v);
  let b = 0;
  const step = (i: number, j: number) => {
    if (i === j) return false;
    const [yi, yj, Ei, Ej, ai, aj] = [y[i], y[j], E[i], E[j], alpha[i], alpha[j]];
    const [L, H] = yi !== yj ? [Math.max(0, aj - ai), Math.min(C, C + aj - ai)] : [Math.max(0, ai + aj - C), Math.min(C, ai + aj)];
    if (H - L < 1e-12) return false;
    const eta = 2 * K[i][j] - K[i][i] - K[j][j];
    if (eta >= -1e-12) return false;
    const ajNew = Math.min(H, Math.max(L, aj - yj * (Ei - Ej) / eta));
    if (Math.abs(ajNew - aj) < 1e-9 * (ajNew + aj + 1e-9)) return false;
    const aiNew = ai + yi * yj * (aj - ajNew);
    const b1 = b - Ei - yi * (aiNew - ai) * K[i][i] - yj * (ajNew - aj) * K[i][j];
    const b2 = b - Ej - yi * (aiNew - ai) * K[i][j] - yj * (ajNew - aj) * K[j][j];
    const bNew = aiNew > 1e-9 && aiNew < C - 1e-9 ? b1 : ajNew > 1e-9 && ajNew < C - 1e-9 ? b2 : (b1 + b2) / 2;
    const di = yi * (aiNew - ai), dj = yj * (ajNew - aj), db = bNew - b;
    for (let m = 0; m < n; m++) E[m] += di * K[i][m] + dj * K[j][m] + db;
    alpha[i] = aiNew; alpha[j] = ajNew; b = bNew;
    return true;
  };
  let sweeps = 0;
  const order = [...Array(n).keys()];
  for (; sweeps < maxSweeps; sweeps++) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const r = E[i] * y[i];
      if (!((r < -tol && alpha[i] < C) || (r > tol && alpha[i] > 0))) continue;
      const Ei = E[i];
      order.sort((p, q) => Math.abs(Ei - E[q]) - Math.abs(Ei - E[p]) || p - q);
      if (order.some(j => step(i, j))) changed++;
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
export const GAMMA_GRID = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100] as const;
export const HARD_C = 1e4;

/** k-fold cross-validated accuracy with a fixed interleaved split (fold = index mod k). */
export function crossValidate(data: Sample[], C: number, k: Kernel, folds = 5) {
  let correct = 0;
  for (let f = 0; f < folds; f++) {
    const train_ = data.filter((_, i) => i % folds !== f), test = data.filter((_, i) => i % folds === f);
    const m = train(train_, C, k);
    correct += test.filter(p => Math.sign(decision(m, p.x)) === p.y).length;
  }
  return correct / data.length;
}
