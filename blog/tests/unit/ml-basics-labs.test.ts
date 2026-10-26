import { describe, expect, it } from 'vitest';
import * as nb from '../../../tools/kien-thuc/naive-bayes-explainer/model';
import * as svm from '../../../tools/kien-thuc/svm-explainer/model';
import * as bv from '../../../tools/kien-thuc/bias-variance-explainer/model';
import * as reg from '../../../tools/kien-thuc/regularization-explainer/model';

describe('naive bayes lab', () => {
  it('counts priors and words as the article tables do', () => {
    expect(nb.classCounts()).toEqual([6, 10]);
    expect([nb.totalWords(0), nb.totalWords(1)]).toEqual([17, 23]);
    expect(nb.likelihood('trúng', 0, 1)).toBeCloseTo(4 / 27, 12);
    expect(nb.likelihood('trúng', 1, 1)).toBeCloseTo(1 / 33, 12);
    expect(nb.likelihood('dự án', 1, 1)).toBeCloseTo(7 / 33, 12);
  });
  it('gives smoothed likelihoods that sum to one per class', () => {
    for (const label of [0, 1] as const) for (const alpha of [0, 0.5, 1, 2]) expect(nb.VOCAB.reduce((sum, word) => sum + nb.likelihood(word, label, alpha), 0)).toBeCloseTo(1, 12);
  });
  it('reproduces the worked example and the double counting comparison', () => {
    const spam = nb.EXAMPLES[0].counts, mixed = nb.EXAMPLES[3].counts;
    expect(nb.classify(spam, 1).posterior[0]).toBeCloseTo(0.9723, 4);
    expect(nb.classify(spam, 1).logScore[0]).toBeCloseTo(-6.709, 3);
    expect(nb.classifyMerged(spam, 1).posterior[0]).toBeCloseTo(0.903, 3);
    expect(nb.classify(mixed, 1).posterior[0]).toBeCloseTo(0.778, 3);
    expect(nb.classifyMerged(mixed, 1).posterior[0]).toBeLessThan(0.5);
  });
  it('is undefined (0/0) without smoothing when each class has an unseen word', () => {
    expect(Number.isNaN(nb.classify(nb.EXAMPLES[3].counts, 0).posterior[0])).toBe(true);
    expect(nb.classify(nb.EXAMPLES[0].counts, 0).posterior[0]).toBe(1);
  });
  it('accumulates evidence word by word to the final posterior', () => {
    const steps = nb.evidence(nb.EXAMPLES[0].counts, 1);
    expect(steps[0].pSpam).toBeCloseTo(0.375, 12);
    expect(steps.at(-1)!.pSpam).toBeCloseTo(nb.classify(nb.EXAMPLES[0].counts, 1).posterior[0], 12);
  });
});

describe('svm lab', () => {
  const hard = svm.train(svm.SEPARABLE, svm.HARD_C);
  it('finds the maximum margin that the angle sweep finds', () => {
    let best = 0;
    for (let degree = -90; degree <= 90; degree += 0.25) best = Math.max(best, svm.corridor(svm.SEPARABLE, degree * Math.PI / 180).width);
    expect(svm.marginWidth(hard)).toBeCloseTo(2.99, 2);
    expect(Math.abs(svm.marginWidth(hard) - best)).toBeLessThan(0.01);
    expect(svm.supportCount(hard)).toBe(3);
    expect(svm.errors(hard)).toBe(0);
  });
  it('satisfies the KKT conditions: Σαy = 0, support vectors on the margin, others outside', () => {
    expect(hard.alpha.reduce((sum, a, i) => sum + a * hard.data[i].y, 0)).toBeCloseTo(0, 8);
    const roles = svm.roles(hard);
    hard.data.forEach((p, i) => {
      const value = p.y * svm.decision(hard, p.x);
      if (roles[i] === 'free') expect(value).toBeGreaterThan(1 - 1e-3); else expect(value).toBeCloseTo(1, 3);
    });
  });
  it('keeps the solution when a non-support point is removed and changes it when a support vector is', () => {
    const roles = svm.roles(hard);
    const free = roles.indexOf('free'), support = roles.findIndex(role => role !== 'free');
    const without = (index: number) => svm.train(svm.SEPARABLE.filter((_, i) => i !== index), svm.HARD_C);
    expect(svm.marginWidth(without(free))).toBeCloseTo(svm.marginWidth(hard), 4);
    expect(svm.marginWidth(without(support))).toBeGreaterThan(svm.marginWidth(hard) + 0.05);
  });
  it('widens the margin and adds support vectors as C shrinks', () => {
    const small = svm.train(svm.OVERLAP, 0.01), large = svm.train(svm.OVERLAP, 100);
    expect(svm.marginWidth(small)).toBeGreaterThan(svm.marginWidth(large));
    expect(svm.supportCount(small)).toBeGreaterThan(svm.supportCount(large));
    expect(small.alpha.every(a => a <= 0.01 + 1e-12)).toBe(true);
  });
  it('needs a kernel for the rings and overfits with a very large γ', () => {
    expect(svm.errors(svm.train(svm.RINGS, 1))).toBe(16);
    const good = svm.train(svm.RINGS, 1, { type: 'rbf', gamma: 0.1 }), sharp = svm.train(svm.RINGS, 1, { type: 'rbf', gamma: 10 });
    expect(svm.errors(good)).toBe(0);
    expect(svm.errors(good, svm.RINGS_TEST)).toBe(2);
    expect(svm.errors(sharp)).toBe(0);
    expect(svm.errors(sharp, svm.RINGS_TEST)).toBe(29);
  });
});

describe('bias-variance lab', () => {
  it('generates reproducible datasets', () => {
    expect(bv.dataset(12, 3)).toEqual(bv.dataset(12, 3));
    expect(bv.dataset(12, 3).y).not.toEqual(bv.dataset(12, 4).y);
  });
  it('fits exactly a polynomial that generated the data', () => {
    const x = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95], y = x.map(v => 1 - 2 * v + 3 * v ** 2);
    const coef = bv.fit({ x, y }, 2);
    expect(bv.predict(coef, 0.42)).toBeCloseTo(1 - 0.84 + 3 * 0.42 ** 2, 8);
  });
  it('reproduces the single-dataset numbers of the article', () => {
    const data = bv.dataset(12, 0);
    expect(bv.trainError(bv.fit(data, 1), data)).toBeCloseTo(0.198, 3);
    expect(bv.testError(bv.fit(data, 3))).toBeCloseTo(0.102, 3);
    expect(bv.testError(bv.fit(data, 9))).toBeCloseTo(2.852, 2);
  });
  it('decomposes the expected error into bias² + variance + σ², matching fresh test points', () => {
    for (const degree of [1, 3, 7]) {
      const row = bv.decompose(degree, 12, 150);
      expect(row.expected).toBeCloseTo(row.bias2 + row.variance + bv.NOISE ** 2, 12);
      expect(Math.abs(row.test - row.expected) / row.expected).toBeLessThan(0.03);
    }
  });
  it('trades bias for variance with degree and lowers variance with more data', () => {
    const low = bv.decompose(1, 12, 150), mid = bv.decompose(3, 12, 150), high = bv.decompose(9, 12, 150);
    expect(low.bias2).toBeGreaterThan(mid.bias2);
    expect(high.variance).toBeGreaterThan(mid.variance);
    expect(mid.expected).toBeLessThan(Math.min(low.expected, high.expected));
    expect(bv.decompose(9, 100, 150).variance).toBeLessThan(high.variance / 50);
  });
});

describe('regularization lab', () => {
  it('matches closed forms in the 2D problem and hits the axis at λ ≈ 1.01', () => {
    expect(reg.ridge2D(0)[0]).toBeCloseTo(2.2, 10);
    expect(reg.lasso2D(1.5)[1]).toBe(0);
    expect(reg.lasso2D(1.5)[0]).toBeCloseTo(1.765, 3);
    expect(reg.ridge2D(1.5)[1]).toBeCloseTo(0.474, 3);
    expect(Math.abs(reg.lasso2D(1.0)[1])).toBeGreaterThan(0);
    expect(reg.lasso2D(1.02)[1]).toBe(0);
  });
  it('implements shrinkage and soft-thresholding', () => {
    expect(reg.ridgeShrink(2.5, 0.5)).toBeCloseTo(1.667, 3);
    expect(reg.lassoShrink(2.5, 0.5)).toBe(2);
    expect(reg.lassoShrink(-0.3, 0.5)).toBe(0);
  });
  it('solves the Lasso optimality conditions', () => {
    const lambda = 0.1, beta = reg.lasso(lambda);
    const n = reg.TRAIN.y.length;
    const residual = reg.TRAIN.y.map((y, i) => y - reg.TRAIN.X[i].reduce((sum, value, j) => sum + value * beta[j], 0));
    beta.forEach((b, j) => {
      const correlation = reg.TRAIN.X.reduce((sum, row, i) => sum + row[j] * residual[i], 0) / n;
      if (b === 0) expect(Math.abs(correlation)).toBeLessThanOrEqual(lambda + 1e-6); else expect(correlation).toBeCloseTo(lambda * Math.sign(b), 6);
    });
  });
  it('reproduces the article numbers on the 8-feature regression', () => {
    const { ridgePath, lassoPath, ols } = reg.paths();
    expect(reg.correlation12).toBeCloseTo(0.976, 2);
    expect(reg.mse(ols, reg.TEST)).toBeCloseTo(3.374, 2);
    const at = (value: number) => reg.LAMBDAS.findIndex(l => Math.abs(l - value) / value < 0.02);
    expect(ridgePath[at(0.1)][0]).toBeCloseTo(1.21, 2);
    expect(lassoPath[at(0.1)][0]).toBe(0);
    expect(lassoPath[at(0.1)][1]).toBeCloseTo(2.88, 2);
    expect(lassoPath.every(beta => beta[0] === 0)).toBe(true);
    expect(Math.min(...lassoPath.map(beta => reg.mse(beta, reg.TEST)))).toBeLessThan(Math.min(...ridgePath.map(beta => reg.mse(beta, reg.TEST))));
    expect(ridgePath.every(beta => reg.nonZero(beta) === 8)).toBe(true);
    expect(reg.nonZero(lassoPath.at(-1)!)).toBe(0);
  });
});
