// Small dense linear algebra for the sequence labs. Vectors are plain arrays; a matrix is an array of rows.
export type Vector = number[];
export type Matrix = number[][];

export const zeros = (rows: number, columns: number): Matrix => Array.from({ length: rows }, () => Array(columns).fill(0));
export const dot = (a: Vector, b: Vector) => a.reduce((sum, value, index) => sum + value * b[index], 0);
export const norm = (a: Vector) => Math.sqrt(dot(a, a));
export const scale = (a: Vector, factor: number) => a.map(value => value * factor);
export const add = (a: Vector, b: Vector) => a.map((value, index) => value + b[index]);
export const subtract = (a: Vector, b: Vector) => a.map((value, index) => value - b[index]);
export const l2normalize = (a: Vector) => { const length = norm(a); return length ? scale(a, 1 / length) : a.slice(); };
export const rmsnormalize = (a: Vector) => { const rms = Math.sqrt(dot(a, a) / a.length); return rms ? scale(a, 1 / rms) : a.slice(); };
export const mean = (vectors: Vector[]) => scale(vectors.reduce((sum, vector) => add(sum, vector)), 1 / vectors.length);

export const scaleMatrix = (m: Matrix, factor: number) => m.map(row => scale(row, factor));
export const addMatrix = (a: Matrix, b: Matrix) => a.map((row, index) => add(row, b[index]));
/** k vᵀ: rows follow k, columns follow v. */
export const outer = (k: Vector, v: Vector): Matrix => k.map(value => scale(v, value));
/** Sᵀ q for S ∈ R^{dk×dv}: a dv-vector. */
export const readOut = (s: Matrix, q: Vector): Vector => s[0].map((_, column) => s.reduce((sum, row, index) => sum + row[column] * q[index], 0));
export const multiply = (m: Matrix, v: Vector): Vector => m.map(row => dot(row, v));

export function softmax(scores: number[]) {
  const finite = scores.filter(Number.isFinite);
  if (!finite.length) return scores.map(() => 0);
  const top = Math.max(...finite);
  const exps = scores.map(score => (Number.isFinite(score) ? Math.exp(score - top) : 0));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map(value => value / total);
}

export const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));
export const maxAbs = (m: Matrix | Vector) => Math.max(1e-9, ...(m as (number | number[])[]).flat().map(Math.abs));

/** Deterministic pseudo-random numbers so every reader sees the same toy example. */
export function seeded(seed: number) {
  let state = seed >>> 0;
  return () => { state = (state + 0x6d2b79f5) >>> 0; let t = state; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
