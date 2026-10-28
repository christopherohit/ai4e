import { describe, expect, it } from 'vitest';
import { LAYER1, LAYER2, MERGED, NETS, POINTS, XOR, accuracy, apply, collapse, logit, xorHidden, xorOut } from '../../../tools/kien-thuc/activation-explainer/model';
import { COV, DIGITS, EIG, MPG, cumulative, reconstruct, varianceAlong } from '../../../tools/kien-thuc/pca-explainer/model';

describe('activation lab', () => {
  it('collapses two affine layers into one', () => {
    for (const p of POINTS.slice(0, 20)) {
      const a = apply(LAYER2, apply(LAYER1, p)), b = apply(MERGED, p);
      expect(a[0]).toBeCloseTo(b[0], 10);
      expect(a[1]).toBeCloseTo(b[1], 10);
    }
  });
  it('reduces a network without activation to a linear model', () => {
    const net = NETS.none8, c = collapse(net);
    for (const p of POINTS.slice(0, 20)) expect(logit(net, p)).toBeCloseTo(c.w[0] * p[0] + c.w[1] * p[1] + c.b, 6);
  });
  it('solves XOR with two ReLU units', () => {
    for (const { x, y } of XOR) expect(xorOut(xorHidden(x))).toBe(y);
  });
  it('matches the training accuracy stored by the Python script', () => {
    for (const key of ['tanh3', 'relu8', 'relu2', 'none8'] as const) expect(accuracy(p => logit(NETS[key], p))).toBeCloseTo(NETS[key].trainAccuracy, 2);
    expect(NETS.relu8.testAccuracy).toBeGreaterThan(0.99);
    expect(NETS.none8.testAccuracy).toBeLessThan(0.9);
  });
});

describe('PCA lab', () => {
  it('recomputes the scikit-learn eigen-decomposition', () => {
    expect(EIG.values[0]).toBeCloseTo(MPG.eigenvalues[0], 3);
    expect(EIG.values[1]).toBeCloseTo(MPG.eigenvalues[1], 3);
    expect(Math.abs(EIG.vectors[0][0])).toBeCloseTo(Math.abs(MPG.eigenvectors[0][0]), 3);
    expect(COV[0][1]).toBeCloseTo(MPG.cov[0][1], 3);
  });
  it('finds the maximum projected variance at the first eigenvector', () => {
    let best = 0;
    for (let a = -90; a <= 90; a += 0.5) best = Math.max(best, varianceAlong([Math.cos(a * Math.PI / 180), Math.sin(a * Math.PI / 180)]));
    expect(best).toBeCloseTo(EIG.values[0], 4);
  });
  it('reconstructs a digit exactly with all 64 components', () => {
    const s = DIGITS.samples[0], full = reconstruct(s.coef, 64);
    full.forEach((v, i) => expect(Math.abs(v - s.pixels[i])).toBeLessThan(0.05));
    expect(cumulative[DIGITS.k95 - 1]).toBeGreaterThanOrEqual(0.95);
    expect(cumulative[DIGITS.k95 - 2]).toBeLessThan(0.95);
  });
});
