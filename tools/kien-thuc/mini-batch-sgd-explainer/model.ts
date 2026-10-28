// Batch / mini-batch / stochastic gradient descent on California Housing: predict the median house value of a census
// block group ($100k) from its median income ($10k) over all 20 640 block groups. Shuffles are seeded.
import data from './data/california-housing.json';
import { LinearProblem, type Theta } from '../../_shared/glass-lab/regression';

export const SOURCE = data.source;
export const INCOME = data.income.map(v => v / 1e4);
export const VALUE = data.value.map(v => v / 1000);
export const problem = new LinearProblem(INCOME, VALUE);
export const N = problem.n;
export const START: Theta = [-0.2, 3.4];
export const VIEW = { w: [-0.45, 0.85] as const, b: [-0.8, 3.8] as const };
export const BATCHES = [1, 8, 64, 512, 4096, N] as const;
/** η chosen per method near its own stability limit (batch GD tolerates the largest step). */
export const LR = { batch: 0.05, mini: 0.02, sgd: 0.002 } as const;

function seeded(seed: number) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function permutation(n: number, seed: number) {
  const next = seeded(seed), order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(next() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
  return order;
}

export interface Run { batch: number; lr: number; points: Theta[]; samples: number[]; lastBatch: Uint32Array[] }
/**
 * θ ← θ − η ∇L_B(θ) over shuffled mini-batches. Records every `every`-th point (and the last) so a 60 000-step SGD run
 * stays light, together with the number of samples processed so far and the most recent batches.
 */
export function train(batch: number, lr: number, epochs: number, seed = 7, every = 1, keep = 0): Run {
  let theta = START;
  const points: Theta[] = [theta], samples = [0], lastBatch: Uint32Array[] = [];
  let processed = 0, step = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    const order = batch >= N ? null : permutation(N, seed * 1000 + epoch);
    for (let start = 0; start < N; start += batch) {
      const rows = order ? order.subarray(start, Math.min(N, start + batch)) : null;
      const [gw, gb] = rows ? problem.batchGradient(theta, rows) : problem.gradient(theta);
      theta = [theta[0] - lr * gw, theta[1] - lr * gb];
      processed += rows ? rows.length : N; step++;
      if (rows && lastBatch.length < keep) lastBatch.push(rows.slice());
      if (step % every === 0 || start + batch >= N) { points.push(theta); samples.push(processed); }
    }
  }
  return { batch, lr, points, samples, lastBatch };
}

/** √trace Cov(ĝ_B) at θ for B rows drawn without replacement: σ̄ · √((N − B) / (B (N − 1))). Exact, O(N). */
export function gradientNoise(theta: Theta, batch: number) {
  const [w, b] = theta, [gw, gb] = problem.gradient(theta);
  let sum = 0;
  for (let i = 0; i < N; i++) { const r = 2 * (w * INCOME[i] + b - VALUE[i]); sum += (r * INCOME[i] - gw) ** 2 + (r - gb) ** 2; }
  return Math.sqrt(sum / N * (N - batch) / (batch * (N - 1)));
}

/** A fan of `count` mini-batch gradients at θ, the way training would draw them. */
export function sampleGradients(theta: Theta, batch: number, count: number, seed = 3): Theta[] {
  const next = seeded(seed);
  return Array.from({ length: count }, () => {
    const rows = new Uint32Array(Math.min(batch, N));
    if (batch >= N) return problem.gradient(theta);
    for (let k = 0; k < rows.length; k++) rows[k] = Math.floor(next() * N);
    return problem.batchGradient(theta, rows);
  });
}
