// Bias–variance lab model on Auto MPG: predict mpg from horsepower with polynomials of degree 1–10 fitted to small
// random subsets of the 392 cars. The unknown true curve is estimated by a local-linear smoother on all cars, so bias²
// and variance are estimates; the held-out test error is measured directly on the cars left out of each subset.
import data from '../../_shared/glass-lab/data/auto-mpg.json';

export const SOURCE = data.source;
export const HP = data.horsepower, MPG = data.mpg, NAME = data.name;
export const N = HP.length;
export const DEGREES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] as const;
export const SIZES = [15, 20, 30, 50, 100, 200, 300] as const;
export const RANGE = [40, 240] as const;

function seeded(seed: number) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
/** Training subset number `run` of size n: n distinct cars, seeded so every reader sees the same draws. */
export function subset(n: number, run: number) {
  const next = seeded(1234 + run * 7919 + n * 104729), order = Array.from({ length: N }, (_, i) => i);
  for (let i = 0; i < n; i++) { const j = i + Math.floor(next() * (N - i)); [order[i], order[j]] = [order[j], order[i]]; }
  return { train: order.slice(0, n), test: order.slice(n) };
}

/** Legendre basis on t = 2(hp − 40)/200 − 1 ∈ [−1, 1]: well conditioned up to degree 10. */
function basis(hp: number, degree: number) {
  const t = 2 * (hp - RANGE[0]) / (RANGE[1] - RANGE[0]) - 1, v = [1, t];
  for (let k = 1; k < degree; k++) v.push(((2 * k + 1) * t * v[k] - k * v[k - 1]) / (k + 1));
  return v.slice(0, degree + 1);
}
export function fit(rows: readonly number[], degree: number) {
  const p = degree + 1, A = Array.from({ length: p }, () => new Array<number>(p).fill(0)), b = new Array<number>(p).fill(0);
  for (const i of rows) { const phi = basis(HP[i], degree); for (let r = 0; r < p; r++) { b[r] += phi[r] * MPG[i]; for (let c = 0; c < p; c++) A[r][c] += phi[r] * phi[c]; } }
  for (let r = 0; r < p; r++) A[r][r] += 1e-9;
  for (let c = 0; c < p; c++) {
    let m = c; for (let r = c + 1; r < p; r++) if (Math.abs(A[r][c]) > Math.abs(A[m][c])) m = r;
    [A[c], A[m]] = [A[m], A[c]]; [b[c], b[m]] = [b[m], b[c]];
    for (let r = c + 1; r < p; r++) { const f = A[r][c] / A[c][c]; for (let k = c; k < p; k++) A[r][k] -= f * A[c][k]; b[r] -= f * b[c]; }
  }
  const x = new Array<number>(p).fill(0);
  for (let r = p - 1; r >= 0; r--) { let s = b[r]; for (let k = r + 1; k < p; k++) s -= A[r][k] * x[k]; x[r] = s / A[r][r]; }
  return x;
}
export const predict = (coef: number[], hp: number) => basis(hp, coef.length - 1).reduce((s, v, k) => s + v * coef[k], 0);
export const mse = (coef: number[], rows: readonly number[]) => rows.reduce((s, i) => s + (predict(coef, HP[i]) - MPG[i]) ** 2, 0) / rows.length;

/** Local linear regression with a Gaussian kernel (bandwidth in hp): the stand-in for the unknown true curve. */
export const BANDWIDTH = 14;
export function reference(hp: number) {
  let s0 = 0, s1 = 0, s2 = 0, t0 = 0, t1 = 0;
  for (let i = 0; i < N; i++) { const d = HP[i] - hp, w = Math.exp(-0.5 * (d / BANDWIDTH) ** 2); s0 += w; s1 += w * d; s2 += w * d * d; t0 += w * MPG[i]; t1 += w * d * MPG[i]; }
  return (s2 * t0 - s1 * t1) / (s0 * s2 - s1 * s1);
}
const REF = HP.map(reference);
/** Noise estimate: mean squared residual of the reference curve over all cars. */
export const NOISE = HP.reduce((s, _, i) => s + (MPG[i] - REF[i]) ** 2, 0) / N;

export interface Decomposition { degree: number; n: number; bias2: number; variance: number; expected: number; test: number; train: number }
/** Average over `runs` subsets of size n. Bias² and variance are evaluated at the 392 real horsepower values. */
export function decompose(degree: number, n: number, runs = 150): Decomposition {
  const preds = new Float64Array(N);
  const sq = new Float64Array(N);
  let test = 0, train = 0;
  for (let r = 0; r < runs; r++) {
    const { train: tr, test: te } = subset(n, r), coef = fit(tr, degree);
    for (let i = 0; i < N; i++) { const p = predict(coef, HP[i]); preds[i] += p; sq[i] += p * p; }
    test += mse(coef, te); train += mse(coef, tr);
  }
  let bias2 = 0, variance = 0;
  for (let i = 0; i < N; i++) { const m = preds[i] / runs; bias2 += (m - REF[i]) ** 2; variance += Math.max(0, sq[i] / runs - m * m); }
  bias2 /= N; variance /= N;
  return { degree, n, bias2, variance, expected: bias2 + variance + NOISE, test: test / runs, train: train / runs };
}
