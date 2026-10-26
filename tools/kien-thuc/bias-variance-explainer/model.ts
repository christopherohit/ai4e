// Bias–variance lab model: fit polynomials of degree d to n noisy samples of f(x) = sin(2πx) on [0, 1], repeated over
// many independent training sets so bias² and variance can be measured directly instead of assumed.
import { seeded } from '../../_shared/sequence-lab/math';

export const truth = (x: number) => Math.sin(2 * Math.PI * x);
export const NOISE = 0.3;
export const MAX_DEGREE = 12;
export const DEGREES = Array.from({ length: MAX_DEGREE + 1 }, (_, degree) => degree);
/** Evaluation grid: midpoints of 120 slices of [0.02, 0.98], so averages over it approximate averages over x. */
export const GRID = Array.from({ length: 120 }, (_, index) => 0.02 + 0.96 * (index + 0.5) / 120);

export interface Dataset { x: number[]; y: number[] }

const gaussian = (random: () => number) => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
/**
 * Training set number `index` of size n: one x per slice [i/n, (i+1)/n) (stratified, so no set leaves a large gap that
 * a high-degree polynomial would extrapolate wildly across), y = f(x) + N(0, σ²). Same index → same data.
 */
export function dataset(n: number, index: number): Dataset {
  const random = seeded(9001 + index * 7919 + n * 104729);
  const x = Array.from({ length: n }, (_, i) => (i + random()) / n);
  return { x, y: x.map(value => truth(value) + NOISE * gaussian(random)) };
}

/** Legendre polynomials P₀..P_d at t = 2x − 1: far better conditioned than 1, x, x², … on [0, 1]. */
function basis(x: number, degree: number) {
  const t = 2 * x - 1, values = [1, t];
  for (let k = 1; k < degree; k++) values.push(((2 * k + 1) * t * values[k] - k * values[k - 1]) / (k + 1));
  return values.slice(0, degree + 1);
}

/** Least squares with a tiny ridge (1e-8) so under-determined fits (d + 1 > n) stay solvable. */
export function fit(data: Dataset, degree: number): number[] {
  const p = degree + 1;
  const A = Array.from({ length: p }, () => Array(p).fill(0)), rhs = Array(p).fill(0);
  data.x.forEach((x, i) => {
    const phi = basis(x, degree);
    for (let r = 0; r < p; r++) { rhs[r] += phi[r] * data.y[i]; for (let c = 0; c < p; c++) A[r][c] += phi[r] * phi[c]; }
  });
  for (let r = 0; r < p; r++) A[r][r] += 1e-8;
  return solve(A, rhs);
}

function solve(A: number[][], b: number[]) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / M[col][col];
      for (let k = col; k <= n; k++) M[row][k] -= factor * M[col][k];
    }
  }
  const x = Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) x[row] = (M[row][n] - M[row].slice(row + 1, n).reduce((sum, value, k) => sum + value * x[row + 1 + k], 0)) / M[row][row];
  return x;
}

export const predict = (coef: number[], x: number) => basis(x, coef.length - 1).reduce((sum, value, k) => sum + value * coef[k], 0);
const mse = (coef: number[], data: Dataset) => data.x.reduce((sum, x, i) => sum + (predict(coef, x) - data.y[i]) ** 2, 0) / data.x.length;

/** A large fresh test set from the same distribution (fixed, shared by all fits). */
export const TEST: Dataset = dataset(3000, -1);
export const trainError = (coef: number[], data: Dataset) => mse(coef, data);
export const testError = (coef: number[]) => mse(coef, TEST);

export interface Decomposition { degree: number; n: number; bias2: number; variance: number; noise: number; expected: number; train: number; test: number; mean: number[]; spread: number[] }

/**
 * Fit `runs` independent training sets. At each grid point: bias = mean prediction − f(x), variance = spread of the
 * predictions. Averaged over x: expected test MSE = bias² + variance + σ².
 */
export function decompose(degree: number, n: number, runs = 200): Decomposition {
  const predictions = Array.from({ length: runs }, (_, run) => { const coef = fit(dataset(n, run), degree); return { grid: GRID.map(x => predict(coef, x)), coef, run }; });
  const mean = GRID.map((_, g) => predictions.reduce((sum, p) => sum + p.grid[g], 0) / runs);
  const varianceAt = GRID.map((_, g) => predictions.reduce((sum, p) => sum + (p.grid[g] - mean[g]) ** 2, 0) / runs);
  const bias2 = GRID.reduce((sum, x, g) => sum + (mean[g] - truth(x)) ** 2, 0) / GRID.length;
  const variance = varianceAt.reduce((sum, value) => sum + value, 0) / GRID.length;
  const train = predictions.reduce((sum, p) => sum + trainError(p.coef, dataset(n, p.run)), 0) / runs;
  // Measured on fresh points restricted to the grid's x-range, so it is comparable with bias² + variance + σ².
  const keep = TEST.x.map(x => x >= 0.02 && x <= 0.98);
  const inside = { x: TEST.x.filter((_, i) => keep[i]), y: TEST.y.filter((_, i) => keep[i]) };
  const test = predictions.reduce((sum, p) => sum + mse(p.coef, inside), 0) / runs;
  return { degree, n, bias2, variance, noise: NOISE ** 2, expected: bias2 + variance + NOISE ** 2, train, test, mean, spread: varianceAt.map(Math.sqrt) };
}

export const curve = (degree: number, n: number, run: number) => { const coef = fit(dataset(n, run), degree); return GRID.map(x => predict(coef, x)); };
