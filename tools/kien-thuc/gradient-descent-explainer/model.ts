// Gradient descent lab model: the same rule θ ← θ − η∇L on three toy losses. A 1D slice of the regression MSE (b held
// at its optimum), the full 2D MSE over (w, b) and a non-convex 1D function with two valleys.
import { DATA, OPTIMUM, gradient, loss, type Point } from '../../_shared/optimization-lab/model';

export interface Problem1D { f: (w: number) => number; df: (w: number) => number }

/** L(w) = MSE(w, b*): a parabola whose curvature is 2·E[x²]. */
export const slice: Problem1D = { f: w => loss([w, OPTIMUM[1]]), df: w => gradient([w, OPTIMUM[1]])[0] };
export const SLICE_CURVATURE = 2 * DATA.reduce((sum, d) => sum + d.x * d.x, 0) / DATA.length;
/** On a parabola each step multiplies the distance to the minimum by (1 − η·L''). */
export const contraction = (lr: number) => 1 - lr * SLICE_CURVATURE;

/** f(w) = w⁴/4 − w² + 0.3w + 1.5: a deep valley near −1.48, a shallow one near 1.33 and a hump near 0.15. */
export const bumpy: Problem1D = { f: w => w ** 4 / 4 - w * w + 0.3 * w + 1.5, df: w => w ** 3 - 2 * w + 0.3 };
/** Critical points of bumpy (roots of w³ − 2w + 0.3), found by Newton's method from rough guesses. */
export const BUMPY_CRITICAL = [-1.5, 0.15, 1.35].map(guess => { let w = guess; for (let i = 0; i < 30; i++) w -= bumpy.df(w) / (3 * w * w - 2); return w; });
export const [GLOBAL_MIN, HUMP, LOCAL_MIN] = BUMPY_CRITICAL;

const diverged = (value: number) => !Number.isFinite(value) || Math.abs(value) > 1e6;

/** w₀, w₁, … with w_{k+1} = w_k − η f'(w_k). */
export function descend1D(problem: Problem1D, start: number, lr: number, steps: number) {
  const path = [start];
  for (let k = 0; k < steps && !diverged(path[k]); k++) path.push(path[k] - lr * problem.df(path[k]));
  return path;
}

/** θ₀, θ₁, … on the full 2D MSE. */
export function descend2D(start: Point, lr: number, steps: number) {
  const path: Point[] = [start];
  for (let k = 0; k < steps && !diverged(path[k][0]) && !diverged(path[k][1]); k++) {
    const [gw, gb] = gradient(path[k]);
    path.push([path[k][0] - lr * gw, path[k][1] - lr * gb]);
  }
  return path;
}

export type Regime = 'slow' | 'smooth' | 'oscillating' | 'diverging';
/** Behaviour of gradient descent on a parabola with curvature c, from the factor 1 − ηc. */
export function regime(lr: number, curvature = SLICE_CURVATURE): Regime {
  const factor = 1 - lr * curvature;
  if (Math.abs(factor) >= 1) return 'diverging';
  if (factor < 0) return 'oscillating';
  return factor > 0.75 ? 'slow' : 'smooth';
}
