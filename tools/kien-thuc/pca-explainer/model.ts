// PCA lab data: Auto MPG weight/horsepower (2-D geometry) and the 8×8 digits (64-D), with reference values computed
// by scikit-learn in scripts/ml-datasets/prepare_nn_pca.py. The 2-D maths is redone here so the lab can sweep angles.
import mpgRaw from '../../_shared/glass-lab/data/auto-mpg.json';
import data from './data/pca.json';

export type Vec = [number, number];
export const MPG = data.mpg as { source: string; mean: Vec; std: Vec; cov: [Vec, Vec]; eigenvalues: Vec; eigenvectors: [Vec, Vec]; explained: Vec };
export const DIGITS = data.digits as { source: string; mean: number[]; components: number[][]; explained: number[]; coords: Vec[]; labels: number[]; k95: number; samples: { index: number; label: number; pixels: number[]; coef: number[] }[] };

/** Raw points: weight in thousands of lb, horsepower. */
export const RAW: Vec[] = mpgRaw.weight.map((w, i) => [w / 1000, mpgRaw.horsepower[i]]);
export const NAMES = mpgRaw.name as string[];
export const STD: Vec[] = RAW.map(([a, b]) => [(a - MPG.mean[0]) / MPG.std[0], (b - MPG.mean[1]) / MPG.std[1]]);

export function covariance(points: Vec[]): [Vec, Vec] {
  const n = points.length, m = [0, 1].map(k => points.reduce((s, p) => s + p[k], 0) / n);
  let a = 0, b = 0, c = 0;
  for (const [x, y] of points) { a += (x - m[0]) ** 2; b += (x - m[0]) * (y - m[1]); c += (y - m[1]) ** 2; }
  return [[a / n, b / n], [b / n, c / n]];
}
/** Symmetric 2×2 eigen-decomposition, largest eigenvalue first, first component of each vector ≥ 0. */
export function eigen([[a, b], [, c]]: [Vec, Vec]) {
  const tr = a + c, det = a * c - b * b, d = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 + d, l2 = tr / 2 - d;
  const v = (l: number): Vec => { const x = Math.abs(b) > 1e-12 ? [b, l - a] : (a >= c) === (l === l1) ? [1, 0] : [0, 1]; const n = Math.hypot(x[0], x[1]); const s = x[0] < 0 ? -1 : 1; return [s * x[0] / n, s * x[1] / n]; };
  return { values: [l1, l2] as Vec, vectors: [v(l1), v(l2)] as [Vec, Vec] };
}
export const COV = covariance(STD);
export const EIG = eigen(COV);
export const unit = (deg: number): Vec => [Math.cos(deg * Math.PI / 180), Math.sin(deg * Math.PI / 180)];
/** Variance of the projections on direction u: uᵀ S u. */
export const varianceAlong = (u: Vec, S: [Vec, Vec] = COV) => u[0] * (S[0][0] * u[0] + S[0][1] * u[1]) + u[1] * (S[1][0] * u[0] + S[1][1] * u[1]);
export const times = (S: [Vec, Vec], u: Vec): Vec => [S[0][0] * u[0] + S[0][1] * u[1], S[1][0] * u[0] + S[1][1] * u[1]];
export const dot = (a: Vec, b: Vec) => a[0] * b[0] + a[1] * b[1];
export const PC1_ANGLE = Math.atan2(EIG.vectors[0][1], EIG.vectors[0][0]) * 180 / Math.PI;

/** Reconstruct a digit from its first k PCA coefficients. */
export function reconstruct(coef: number[], k: number) {
  return DIGITS.mean.map((m, p) => { let v = m; for (let j = 0; j < k; j++) v += coef[j] * DIGITS.components[j][p]; return v; });
}
export const cumulative = DIGITS.explained.reduce<number[]>((acc, v) => { acc.push((acc.at(-1) ?? 0) + v); return acc; }, []);
export const mse = (a: number[], b: number[]) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length;
