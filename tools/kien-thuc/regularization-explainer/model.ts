// L1 (Lasso) vs L2 (Ridge) lab model. Two problems: a 2D quadratic loss where the geometry of the penalty is visible,
// and a linear regression with 8 features (two of them nearly identical, three irrelevant) for coefficient paths.
import { seeded } from '../../_shared/sequence-lab/math';

export type Point = readonly [number, number];

// ---------- 2D geometry: L(w) = (w − w*)ᵀ A (w − w*) ----------
export const A2 = [[1, 0.45], [0.45, 0.6]] as const;
export const W_STAR: Point = [2.2, 0.7];
export const quad = (w: Point) => { const d = [w[0] - W_STAR[0], w[1] - W_STAR[1]]; return A2[0][0] * d[0] ** 2 + 2 * A2[0][1] * d[0] * d[1] + A2[1][1] * d[1] ** 2; };

/** argmin L(w) + λ‖w‖²: (A + λI) w = A w*. */
export function ridge2D(lambda: number): Point {
  const [a, b, c] = [A2[0][0] + lambda, A2[0][1], A2[1][1] + lambda];
  const r = [A2[0][0] * W_STAR[0] + A2[0][1] * W_STAR[1], A2[0][1] * W_STAR[0] + A2[1][1] * W_STAR[1]];
  const det = a * c - b * b;
  return [(c * r[0] - b * r[1]) / det, (a * r[1] - b * r[0]) / det];
}

/** argmin L(w) + λ‖w‖₁ by exact coordinate minimisation (each 1D problem is a soft-threshold). */
export function lasso2D(lambda: number): Point {
  const w = [0, 0];
  const r = [A2[0][0] * W_STAR[0] + A2[0][1] * W_STAR[1], A2[0][1] * W_STAR[0] + A2[1][1] * W_STAR[1]];
  for (let sweep = 0; sweep < 500; sweep++) {
    let change = 0;
    for (let j = 0; j < 2; j++) {
      const k = 1 - j;
      // d/dw_j: 2A_jj w_j + 2A_jk w_k − 2r_j + λ·sign(w_j) = 0
      const next = softThreshold(r[j] - A2[j][k] * w[k], lambda / 2) / A2[j][j];
      change = Math.max(change, Math.abs(next - w[j]));
      w[j] = next;
    }
    if (change < 1e-12) break;
  }
  return [w[0], w[1]];
}

export const softThreshold = (value: number, threshold: number) => (Math.abs(value) <= threshold ? 0 : Math.sign(value) * (Math.abs(value) - threshold));
/** Orthonormal design: Ridge divides the least-squares coefficient, Lasso subtracts and clips at zero. */
export const ridgeShrink = (beta: number, lambda: number) => beta / (1 + lambda);
export const lassoShrink = (beta: number, lambda: number) => softThreshold(beta, lambda);

/** Points of the ellipse L(w) = c around w*. */
export function ellipse(c: number, count = 96): Point[] {
  const tr = A2[0][0] + A2[1][1], det = A2[0][0] * A2[1][1] - A2[0][1] ** 2, s = Math.sqrt(tr * tr / 4 - det);
  const lambdas = [tr / 2 + s, tr / 2 - s];
  const vectors = lambdas.map(l => { const v = [A2[0][1], l - A2[0][0]]; const n = Math.hypot(v[0], v[1]); return [v[0] / n, v[1] / n]; });
  return Array.from({ length: count + 1 }, (_, i) => {
    const t = i / count * 2 * Math.PI, a = Math.sqrt(c / lambdas[0]) * Math.cos(t), b = Math.sqrt(c / lambdas[1]) * Math.sin(t);
    return [W_STAR[0] + a * vectors[0][0] + b * vectors[1][0], W_STAR[1] + a * vectors[0][1] + b * vectors[1][1]] as Point;
  });
}

// ---------- 8-feature regression ----------
export const FEATURES = ['x₁', 'x₂ ≈ x₁', 'x₃', 'x₄', 'x₅', 'x₆', 'x₇', 'x₈'];
export const TRUE_BETA = [1.5, 1.5, -2, 0, 0, 1, 0, 0];
export const SIGNAL = TRUE_BETA.map(value => value !== 0);
export const N_TRAIN = 25, N_TEST = 500, NOISE = 1.5;

export interface Data { X: number[][]; y: number[] }
function generate(n: number, seed: number): Data {
  const random = seeded(seed);
  const gaussian = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
  const X = Array.from({ length: n }, () => {
    const row = Array.from({ length: 8 }, gaussian);
    row[1] = row[0] + 0.25 * gaussian();
    return row;
  });
  return { X, y: X.map(row => row.reduce((sum, value, j) => sum + value * TRUE_BETA[j], 0) + NOISE * gaussian()) };
}
const rawTrain = generate(N_TRAIN, 11), rawTest = generate(N_TEST, 78);
// Centre with training means so the intercept drops out (equivalent to fitting an unpenalised intercept).
const means = Array.from({ length: 8 }, (_, j) => rawTrain.X.reduce((sum, row) => sum + row[j], 0) / N_TRAIN);
const yMean = rawTrain.y.reduce((sum, value) => sum + value, 0) / N_TRAIN;
const centre = (data: Data): Data => ({ X: data.X.map(row => row.map((value, j) => value - means[j])), y: data.y.map(value => value - yMean) });
export const TRAIN = centre(rawTrain), TEST = centre(rawTest);

const gram = TRAIN.X[0].map((_, j) => TRAIN.X[0].map((__, k) => TRAIN.X.reduce((sum, row) => sum + row[j] * row[k], 0) / N_TRAIN));
const xty = TRAIN.X[0].map((_, j) => TRAIN.X.reduce((sum, row, i) => sum + row[j] * TRAIN.y[i], 0) / N_TRAIN);
export const correlation12 = gram[0][1] / Math.sqrt(gram[0][0] * gram[1][1]);

function solve(M: number[][], b: number[]) {
  const n = b.length, A = M.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let p = col; for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[p][col])) p = r;
    [A[col], A[p]] = [A[p], A[col]];
    for (let r = col + 1; r < n; r++) { const f = A[r][col] / A[col][col]; for (let k = col; k <= n; k++) A[r][k] -= f * A[col][k]; }
  }
  const x = Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = A[r][n]; for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k]; x[r] = s / A[r][r]; }
  return x;
}

/** Ridge: min (1/2n)‖y − Xβ‖² + (λ/2)‖β‖²  ⇒  (XᵀX/n + λI) β = Xᵀy/n. λ = 0 is ordinary least squares. */
export const ridge = (lambda: number) => solve(gram.map((row, j) => row.map((value, k) => value + (j === k ? lambda : 0))), xty);

/** Lasso: min (1/2n)‖y − Xβ‖² + λ‖β‖₁ by cyclic coordinate descent (warm start optional). */
export function lasso(lambda: number, warm?: number[]) {
  const beta = warm ? warm.slice() : Array(8).fill(0);
  for (let sweep = 0; sweep < 5000; sweep++) {
    let change = 0;
    for (let j = 0; j < 8; j++) {
      const rho = xty[j] - gram[j].reduce((sum, value, k) => (k === j ? sum : sum + value * beta[k]), 0);
      const next = softThreshold(rho, lambda) / gram[j][j];
      change = Math.max(change, Math.abs(next - beta[j]));
      beta[j] = next;
    }
    if (change < 1e-10) break;
  }
  return beta;
}

export const mse = (beta: number[], data: Data) => data.X.reduce((sum, row, i) => sum + (row.reduce((s, value, j) => s + value * beta[j], 0) - data.y[i]) ** 2, 0) / data.y.length;

/** λ grid shared by both methods, log-spaced from 10⁻³ to 10. */
export const LAMBDAS = Array.from({ length: 25 }, (_, i) => 10 ** (-3 + 4 * i / 24));
export function paths() {
  const ridgePath = LAMBDAS.map(ridge);
  const lassoPath: number[][] = [];
  // Warm start from large λ (all zero) downwards, as glmnet does.
  let warm = Array(8).fill(0);
  for (let i = LAMBDAS.length - 1; i >= 0; i--) { warm = lasso(LAMBDAS[i], warm); lassoPath[i] = warm; }
  return { ridgePath, lassoPath, ols: ridge(0) };
}
export const nonZero = (beta: number[]) => beta.filter(value => Math.abs(value) > 1e-8).length;
