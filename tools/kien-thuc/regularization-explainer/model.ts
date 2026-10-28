// L1 / L2 regularization lab model on the Diabetes data (Efron et al. 2004): 442 patients, 10 baseline measurements,
// target = disease progression one year later. Features are standardized, the target centred; Lasso uses
// scikit-learn's objective (1/2n)‖y − Xβ‖² + λ‖β‖₁ and Ridge (1/2n)‖y − Xβ‖² + (λ/2)‖β‖².
import data from './data/diabetes.json';

export const SOURCE = data.source;
export const NAMES = ['tuổi', 'giới tính', 'BMI', 'huyết áp', 's1 · cholesterol TP', 's2 · LDL', 's3 · HDL', 's4 · TC/HDL', 's5 · triglyceride', 's6 · đường huyết'];
export const SHORT = ['tuổi', 'giới', 'BMI', 'HA', 's1', 's2', 's3', 's4', 's5', 's6'];
export const P = 10;
export const REFERENCE = data.reference;
const RAW = data.X, Y = data.y;
export const N = Y.length;

export interface Design { X: number[][]; y: number[]; gram: number[][]; xty: number[]; mean: number[]; sd: number[]; yMean: number }
/** Standardize the given rows with their own means and standard deviations; centre y. */
export function design(rows: readonly number[], columns: readonly number[] = [...Array(P).keys()]): Design {
  const n = rows.length, mean = columns.map(j => rows.reduce((s, i) => s + RAW[i][j], 0) / n);
  const sd = columns.map((j, k) => Math.sqrt(rows.reduce((s, i) => s + (RAW[i][j] - mean[k]) ** 2, 0) / n));
  const yMean = rows.reduce((s, i) => s + Y[i], 0) / n;
  const X = rows.map(i => columns.map((j, k) => (RAW[i][j] - mean[k]) / sd[k])), y = rows.map(i => Y[i] - yMean);
  const gram = columns.map((_, a) => columns.map((__, b) => X.reduce((s, row) => s + row[a] * row[b], 0) / n));
  const xty = columns.map((_, a) => X.reduce((s, row, r) => s + row[a] * y[r], 0) / n);
  return { X, y, gram, xty, mean, sd, yMean };
}
export const ALL = [...Array(N).keys()];
export const FULL = design(ALL);

function solve(M: number[][], b: number[]) {
  const n = b.length, A = M.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let m = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[m][c])) m = r;
    [A[c], A[m]] = [A[m], A[c]];
    for (let r = c + 1; r < n; r++) { const f = A[r][c] / A[c][c]; for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k]; }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = A[r][n]; for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k]; x[r] = s / A[r][r]; }
  return x;
}
export const ols = (d: Design) => solve(d.gram, d.xty);
export const ridge = (d: Design, lambda: number) => solve(d.gram.map((row, j) => row.map((v, k) => v + (j === k ? lambda : 0))), d.xty);

export const softThreshold = (value: number, threshold: number) => (Math.abs(value) <= threshold ? 0 : Math.sign(value) * (Math.abs(value) - threshold));
export function lasso(d: Design, lambda: number, warm?: number[]) {
  const p = d.xty.length, beta = warm ? warm.slice() : new Array<number>(p).fill(0);
  for (let sweep = 0; sweep < 10000; sweep++) {
    let change = 0;
    for (let j = 0; j < p; j++) {
      let rho = d.xty[j];
      for (let k = 0; k < p; k++) if (k !== j) rho -= d.gram[j][k] * beta[k];
      const next = softThreshold(rho, lambda) / d.gram[j][j];
      change = Math.max(change, Math.abs(next - beta[j])); beta[j] = next;
    }
    if (change < 1e-9) break;
  }
  return beta;
}

/** λ grid shared by both penalties, ascending, the same as scikit-learn's reference path (10⁻² … 10²). */
export const LAMBDAS = REFERENCE.lassoAlphas.slice().sort((a, b) => a - b);
export function lassoPath(d: Design) {
  const path: number[][] = new Array(LAMBDAS.length);
  let warm = new Array<number>(d.xty.length).fill(0);
  for (let i = LAMBDAS.length - 1; i >= 0; i--) { warm = lasso(d, LAMBDAS[i], warm); path[i] = warm; }
  return path;
}
export const ridgePath = (d: Design) => LAMBDAS.map(l => ridge(d, l));

export const mse = (d: Design, beta: number[]) => d.X.reduce((s, row, i) => s + (row.reduce((a, v, j) => a + v * beta[j], 0) - d.y[i]) ** 2, 0) / d.y.length;
/** Prediction error on rows `test` for a model fitted on design `train` (test rows standardized with train statistics). */
function testError(train: Design, beta: number[], test: readonly number[], columns: readonly number[]) {
  return test.reduce((s, i) => { const pred = columns.reduce((a, j, k) => a + beta[k] * (RAW[i][j] - train.mean[k]) / train.sd[k], 0) + train.yMean; return s + (pred - Y[i]) ** 2; }, 0) / test.length;
}

/** 10-fold cross-validated MSE along the λ grid (fold = row index mod 10, standardization inside each fold). */
export function crossValidate(folds = 10) {
  const ridgeErr = new Array<number>(LAMBDAS.length).fill(0), lassoErr = new Array<number>(LAMBDAS.length).fill(0);
  let olsErr = 0;
  const columns = [...Array(P).keys()];
  for (let f = 0; f < folds; f++) {
    const train = ALL.filter(i => i % folds !== f), test = ALL.filter(i => i % folds === f), d = design(train);
    const share = test.length / N;
    olsErr += testError(d, ols(d), test, columns) * share;
    lassoPath(d).forEach((beta, i) => { lassoErr[i] += testError(d, beta, test, columns) * share; });
    ridgePath(d).forEach((beta, i) => { ridgeErr[i] += testError(d, beta, test, columns) * share; });
  }
  return { ridge: ridgeErr, lasso: lassoErr, ols: olsErr };
}

/** Bootstrap resamples (seeded): how much the s1 / s2 coefficients move from one sample of patients to another. */
export function bootstrap(count: number, lambda: { ridge: number; lasso: number }) {
  let s = 99;
  const next = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return Array.from({ length: count }, () => {
    const rows = Array.from({ length: N }, () => Math.floor(next() * N)), d = design(rows);
    return { ols: ols(d), ridge: ridge(d, lambda.ridge), lasso: lasso(d, lambda.lasso) };
  });
}

/** The 2D problem of chapter 1: only BMI and s5, the two strongest predictors. */
export const PAIR = [2, 8] as const;
export const PAIR_DESIGN = design(ALL, PAIR);
