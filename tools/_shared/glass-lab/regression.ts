// Two-parameter least squares ŷ = w·x + b on real data: loss, gradient, closed-form optimum, Hessian eigenvalues and
// plain gradient descent. Shared by the gradient-descent and mini-batch labs.
export type Theta = readonly [number, number];

export class LinearProblem {
  readonly n: number;
  readonly mx: number; readonly my: number; readonly mxx: number; readonly mxy: number; readonly myy: number;
  constructor(readonly x: readonly number[], readonly y: readonly number[]) {
    this.n = x.length;
    const mean = (f: (i: number) => number) => { let s = 0; for (let i = 0; i < this.n; i++) s += f(i); return s / this.n; };
    this.mx = mean(i => x[i]); this.my = mean(i => y[i]);
    this.mxx = mean(i => x[i] * x[i]); this.mxy = mean(i => x[i] * y[i]); this.myy = mean(i => y[i] * y[i]);
  }
  /** Mean squared error, expanded from the five moments so it costs O(1). */
  loss([w, b]: Theta) { return w * w * this.mxx + 2 * w * b * this.mx + b * b - 2 * w * this.mxy - 2 * b * this.my + this.myy; }
  gradient([w, b]: Theta): Theta { return [2 * (w * this.mxx + b * this.mx - this.mxy), 2 * (w * this.mx + b - this.my)]; }
  /** Gradient on a subset of rows (mini-batch). */
  batchGradient([w, b]: Theta, rows: ArrayLike<number>): Theta {
    let gw = 0, gb = 0;
    for (let k = 0; k < rows.length; k++) { const i = rows[k]; const r = w * this.x[i] + b - this.y[i]; gw += r * this.x[i]; gb += r; }
    return [2 * gw / rows.length, 2 * gb / rows.length];
  }
  get optimum(): Theta { const w = (this.mxy - this.mx * this.my) / (this.mxx - this.mx * this.mx); return [w, this.my - w * this.mx]; }
  get minLoss() { return this.loss(this.optimum); }
  /** Eigenvalues of the Hessian 2·[[E x², E x], [E x, 1]], largest first. */
  get curvatures(): [number, number] {
    const tr = this.mxx + 1, det = this.mxx - this.mx * this.mx, s = Math.sqrt(tr * tr / 4 - det);
    return [2 * (tr / 2 + s), 2 * (tr / 2 - s)];
  }
  get maxStableLr() { return 2 / this.curvatures[0]; }
  get condition() { const [a, b] = this.curvatures; return a / b; }
  descend(start: Theta, lr: number, steps: number): Theta[] {
    const path: Theta[] = [start];
    for (let k = 0; k < steps; k++) {
      const p = path[k], [gw, gb] = this.gradient(p), next: Theta = [p[0] - lr * gw, p[1] - lr * gb];
      path.push(next);
      if (!Number.isFinite(next[0]) || Math.abs(next[0]) + Math.abs(next[1]) > 1e9) break;
    }
    return path;
  }
  /** Number of steps until the loss is within `tolerance` (relative) of the minimum, or Infinity. */
  stepsTo(start: Theta, lr: number, tolerance = 0.01, cap = 100000) {
    let p = start;
    const target = this.minLoss * (1 + tolerance);
    for (let k = 0; k <= cap; k++) {
      if (this.loss(p) <= target) return k;
      const [gw, gb] = this.gradient(p);
      p = [p[0] - lr * gw, p[1] - lr * gb];
      if (!Number.isFinite(p[0]) || Math.abs(p[0]) > 1e9) return Infinity;
    }
    return Infinity;
  }
}

export const standardize = (values: readonly number[]) => {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
  return { mean, sd, z: values.map(v => (v - mean) / sd) };
};
