// Shared toy problem for the optimization labs (gradient descent, batch / mini-batch / SGD): fit ŷ = w·x + b to 64
// synthetic points with mean squared error. Everything is small and closed-form so the article can quote exact numbers.
import { seeded } from '../sequence-lab/math';

export type Point = readonly [number, number];
export interface Sample { x: number; y: number }

export const N = 64;
export const TRUE_W = 1.5, TRUE_B = 0.8;

const random = seeded(2026);
const gaussian = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
/** x: diện tích (trăm m²), y: giá (tỷ đồng). Synthetic: y = 1.5x + 0.8 + noise. */
export const DATA: Sample[] = Array.from({ length: N }, (_, index) => {
  const x = 0.2 + 2 * (index + random()) / N;
  return { x: round(x), y: round(TRUE_W * x + TRUE_B + 0.45 * gaussian()) };
});
function round(value: number) { return Math.round(value * 1000) / 1000; }

const all = DATA.map((_, index) => index);
export const ALL = all;

/** L(w,b) = mean over the batch of (w·x + b − y)². */
export function loss([w, b]: Point, batch: readonly number[] = all) {
  return batch.reduce((sum, index) => sum + (w * DATA[index].x + b - DATA[index].y) ** 2, 0) / batch.length;
}

/** ∇L = (2/|B|) Σ r_i · (x_i, 1), with residual r_i = w·x_i + b − y_i. */
export function gradient([w, b]: Point, batch: readonly number[] = all): Point {
  let gw = 0, gb = 0;
  for (const index of batch) {
    const residual = w * DATA[index].x + b - DATA[index].y;
    gw += residual * DATA[index].x; gb += residual;
  }
  return [2 * gw / batch.length, 2 * gb / batch.length];
}

const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
const Sx = mean(DATA.map(d => d.x)), Sxx = mean(DATA.map(d => d.x * d.x));
const Sy = mean(DATA.map(d => d.y)), Sxy = mean(DATA.map(d => d.x * d.y));

/** Least-squares optimum θ* (where the full gradient is exactly zero). */
export const OPTIMUM: Point = (() => { const w = (Sxy - Sx * Sy) / (Sxx - Sx * Sx); return [w, Sy - w * Sx]; })();
export const MIN_LOSS = loss(OPTIMUM);

/** L is quadratic: L(θ) = L* + (θ−θ*)ᵀ A (θ−θ*) with A = [[E x², E x], [E x, 1]]; the Hessian is 2A. */
const A = [[Sxx, Sx], [Sx, 1]] as const;
const trace = A[0][0] + A[1][1], det = A[0][0] * A[1][1] - A[0][1] ** 2;
const spread = Math.sqrt(trace * trace / 4 - det);
/** Eigenvalues of the Hessian 2A, largest first. */
export const CURVATURES = [2 * (trace / 2 + spread), 2 * (trace / 2 - spread)] as const;
/** Full-batch gradient descent on a quadratic converges iff η < 2 / λ_max(H). */
export const MAX_STABLE_LR = 2 / CURVATURES[0];

/** Points on the level set L(θ) = L* + c: an ellipse around θ* along the eigenvectors of A. */
export function levelSet(c: number, count = 96): Point[] {
  const lambdas = [trace / 2 + spread, trace / 2 - spread];
  const vectors = lambdas.map(lambda => { const v = [A[0][1], lambda - A[0][0]]; const length = Math.hypot(v[0], v[1]); return [v[0] / length, v[1] / length]; });
  return Array.from({ length: count + 1 }, (_, index) => {
    const t = (index / count) * 2 * Math.PI;
    const a = Math.sqrt(c / lambdas[0]) * Math.cos(t), s = Math.sqrt(c / lambdas[1]) * Math.sin(t);
    return [OPTIMUM[0] + a * vectors[0][0] + s * vectors[1][0], OPTIMUM[1] + a * vectors[0][1] + s * vectors[1][1]] as Point;
  });
}

/** Seeded Fisher–Yates permutation of 0..N−1, one per epoch so every reader sees the same run. */
export function shuffled(seed: number) {
  const next = seeded(seed);
  const order = all.slice();
  for (let index = order.length - 1; index > 0; index--) {
    const pick = Math.floor(next() * (index + 1));
    [order[index], order[pick]] = [order[pick], order[index]];
  }
  return order;
}

/** Split one epoch into consecutive mini-batches of size B; the last one is smaller when B does not divide N. */
export function epochBatches(batchSize: number, seed: number) {
  const order = batchSize >= N ? all.slice() : shuffled(seed);
  const batches: number[][] = [];
  for (let start = 0; start < N; start += batchSize) batches.push(order.slice(start, start + batchSize));
  return batches;
}

export interface Update { epoch: number; batch: number[]; before: Point; after: Point; step: Point; full: Point }
export interface Run { points: Point[]; updates: Update[]; batchSize: number; lr: number }

/** θ ← θ − η ∇L_B(θ) over shuffled mini-batches. B = N is batch GD, B = 1 is SGD. Stops early if θ diverges. */
export function train({ batchSize, lr, epochs, start, seed = 1 }: { batchSize: number; lr: number; epochs: number; start: Point; seed?: number }): Run {
  let theta = start;
  const points: Point[] = [start], updates: Update[] = [];
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (const batch of epochBatches(batchSize, seed * 1000 + epoch)) {
      const step = gradient(theta, batch);
      const after: Point = [theta[0] - lr * step[0], theta[1] - lr * step[1]];
      updates.push({ epoch, batch, before: theta, after, step, full: gradient(theta) });
      points.push(after);
      theta = after;
      if (!Number.isFinite(theta[0]) || Math.hypot(theta[0], theta[1]) > 1e6) return { points, updates, batchSize, lr };
    }
  }
  return { points, updates, batchSize, lr };
}

/** Full-dataset loss after every epoch (index 0 = before training). */
export function lossPerEpoch(run: Run, epochs: number) {
  const values = [loss(run.points[0])];
  for (let epoch = 0; epoch < epochs; epoch++) {
    const last = run.updates.findLast(update => update.epoch === epoch);
    values.push(last ? loss(last.after) : Infinity);
  }
  return values;
}

/**
 * Per-coordinate spread of the mini-batch gradient around the full gradient when B samples are drawn without
 * replacement: Cov = (N − B) / (B (N − 1)) · Σ, with Σ the covariance of per-sample gradients. Returns √trace(Cov).
 */
export function gradientNoise(theta: Point, batchSize: number) {
  const full = gradient(theta);
  const sigma = all.reduce((sum, index) => { const g = gradient(theta, [index]); return sum + (g[0] - full[0]) ** 2 + (g[1] - full[1]) ** 2; }, 0) / N;
  return Math.sqrt(sigma * (N - batchSize) / (batchSize * (N - 1)));
}

/** A fixed set of random mini-batch gradients at θ, drawn the way training draws them (for the gradient cloud). */
export function sampleGradients(theta: Point, batchSize: number, count: number, seed = 5): Point[] {
  return Array.from({ length: count }, (_, index) => gradient(theta, shuffled(seed * 7919 + index).slice(0, batchSize)));
}
