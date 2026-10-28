import { describe, expect, it } from 'vitest';
import autoMpg from '../../../tools/_shared/glass-lab/data/auto-mpg.json';
import { LinearProblem } from '../../../tools/_shared/glass-lab/regression';
import { march, sample, vn } from '../../../tools/_shared/glass-lab/gfx';
import * as gd from '../../../tools/kien-thuc/gradient-descent-explainer/model';
import * as sgd from '../../../tools/kien-thuc/mini-batch-sgd-explainer/model';
import housing from '../../../tools/kien-thuc/mini-batch-sgd-explainer/data/california-housing.json';
import * as nb from '../../../tools/kien-thuc/naive-bayes-explainer/model';
import * as svm from '../../../tools/kien-thuc/svm-explainer/model';
import * as bv from '../../../tools/kien-thuc/bias-variance-explainer/model';
import * as reg from '../../../tools/kien-thuc/regularization-explainer/model';

describe('glass-lab drawing helpers', () => {
  it('formats numbers the Vietnamese way', () => {
    expect(vn(1234.5, 1)).toBe('1.234,5');
    expect(vn(-0.25)).toBe('−0,25');
  });
  it('finds the level set of a function with marching squares', () => {
    const segments = march(sample((x, y) => x * x + y * y, [-2, 2], [-2, 2], 40, 40), 1);
    for (const [a, b] of segments) for (const [x, y] of [a, b]) expect(Math.hypot(x, y)).toBeCloseTo(1, 1);
  });
});

describe('two-parameter least squares', () => {
  const p = new LinearProblem([1, 2, 3, 4], [2, 4, 5, 8]);
  it('matches finite differences and the closed-form optimum', () => {
    const h = 1e-6, at = [0.3, -0.2] as const, [gw, gb] = p.gradient(at);
    expect(gw).toBeCloseTo((p.loss([at[0] + h, at[1]]) - p.loss([at[0] - h, at[1]])) / (2 * h), 5);
    expect(gb).toBeCloseTo((p.loss([at[0], at[1] + h]) - p.loss([at[0], at[1] - h])) / (2 * h), 5);
    expect(p.gradient(p.optimum).every(v => Math.abs(v) < 1e-9)).toBe(true);
    expect(p.batchGradient(at, [0, 1, 2, 3])).toEqual(p.gradient(at).map(v => expect.closeTo(v, 10)));
  });
});

describe('gradient descent lab (Auto MPG)', () => {
  it('reproduces the scikit-learn fit of mpg on weight', () => {
    const ref = autoMpg.reference.weightFit;
    expect(gd.raw.optimum[0]).toBeCloseTo(ref.slope, 3);
    expect(gd.raw.optimum[1]).toBeCloseTo(ref.intercept, 3);
    expect(gd.raw.minLoss).toBeCloseTo(ref.mse, 3);
    expect(gd.CARS).toHaveLength(392);
  });
  it('has a narrow valley in raw units and a round bowl after standardizing', () => {
    expect(gd.raw.condition).toBeCloseTo(153.7, 0);
    expect(gd.std.condition).toBeCloseTo(1, 6);
    expect(gd.raw.maxStableLr).toBeCloseTo(0.0951, 4);
    expect(gd.raw.stepsTo(gd.START_RAW, 0.09)).toBe(229);
    expect(gd.std.stepsTo(gd.START_STD, 0.3)).toBe(4);
    expect(gd.raw.stepsTo(gd.START_RAW, 0.1)).toBe(Infinity);
  });
  it('maps lines between the two parametrisations', () => {
    const back = gd.stdToRaw(gd.START_STD);
    expect(back[0]).toBeCloseTo(gd.START_RAW[0], 10);
    expect(gd.std.loss(gd.START_STD)).toBeCloseTo(gd.raw.loss(gd.START_RAW), 8);
  });
});

describe('mini-batch SGD lab (California Housing)', () => {
  it('uses all 20 640 block groups and matches the scikit-learn fit', () => {
    expect(sgd.N).toBe(20640);
    expect(sgd.problem.optimum[0]).toBeCloseTo(housing.reference.slope, 3);
    expect(sgd.problem.minLoss).toBeCloseTo(housing.reference.mse, 3);
  });
  it('makes 1, ⌈N/B⌉ and N updates per epoch over a permutation', () => {
    const order = sgd.permutation(sgd.N, 3);
    expect(new Set(order).size).toBe(sgd.N);
    expect(sgd.train(sgd.N, sgd.LR.batch, 2).points).toHaveLength(3);
    expect(sgd.train(64, sgd.LR.mini, 1).points).toHaveLength(Math.ceil(sgd.N / 64) + 1);
  });
  it('shrinks gradient noise like 1/√B and removes it at B = N', () => {
    expect(sgd.gradientNoise(sgd.START, 1)).toBeCloseTo(17.34, 1);
    expect(sgd.gradientNoise(sgd.START, sgd.N)).toBe(0);
    const ratio = sgd.gradientNoise(sgd.START, 64) / sgd.gradientNoise(sgd.START, 4096);
    expect(ratio).toBeGreaterThan(7.5); expect(ratio).toBeLessThan(9.5);
  });
  it('reproduces the race quoted in the article', () => {
    const loss = (run: ReturnType<typeof sgd.train>) => sgd.problem.loss(run.points.at(-1)!);
    expect(loss(sgd.train(sgd.N, sgd.LR.batch, 2))).toBeCloseTo(2.268, 2);
    expect(loss(sgd.train(64, sgd.LR.mini, 2))).toBeCloseTo(0.704, 2);
    expect(sgd.problem.stepsTo(sgd.START, sgd.LR.batch)).toBe(147);
  });
});

describe('naive bayes lab (SMS Spam Collection)', () => {
  it('keeps the split, vocabulary and smoothed likelihoods consistent', () => {
    expect(nb.TRAIN).toEqual({ spam: 598, ham: 3859 });
    expect(nb.V).toBe(3672);
    for (const alpha of [0.01, 1, 10]) {
      let spam = 0, ham = 0;
      for (const [word] of nb.VOCAB) { const [s, h] = nb.likelihood(word, alpha); spam += s; ham += h; }
      expect(spam).toBeCloseTo(1, 9); expect(ham).toBeCloseTo(1, 9);
    }
  });
  it('turns words into evidence as in the article', () => {
    expect(nb.logRatio('claim', 1)).toBeCloseTo(5.71, 2);
    expect(nb.tokenize('WINNER! Claim £1000 now')).toEqual(['winner', 'claim', '£1000', 'now']);
    const r = nb.evidence(nb.EXAMPLES[3].text, 1);
    expect(r.steps.at(-1)!.total).toBeCloseTo(r.logOdds, 12);
    expect(r.pSpam).toBeLessThan(0.01);
    expect(nb.evidence(nb.EXAMPLES[0].text, 1).pSpam).toBeGreaterThan(0.999);
  });
  it('stores the scikit-learn test metrics', () => {
    const m = nb.METRICS.find(x => x.alpha === 1)!;
    expect(m.tp + m.fn).toBe(149); expect(m.tn + m.fp).toBe(966);
    expect(m.accuracy).toBeCloseTo(0.9857, 4);
    expect(nb.EXTREME).toBeGreaterThan(0.8);
    expect(nb.CALIBRATION.reduce((s, b) => s + b.spam + b.ham, 0)).toBe(1115);
  });
});

describe('svm lab (Iris)', () => {
  const hard = svm.train(svm.SEPARABLE.data, svm.HARD_C);
  it('matches scikit-learn on the three Iris problems', () => {
    expect(svm.marginWidth(hard)).toBeCloseTo(svm.REFERENCE.separable.margin, 3);
    expect(svm.supportCount(hard)).toBe(svm.REFERENCE.separable.supportVectors);
    const soft = svm.train(svm.OVERLAP.data, 1);
    expect(svm.supportCount(soft)).toBe(svm.REFERENCE.overlapC1.supportVectors);
    expect(1 - svm.errors(soft) / soft.data.length).toBeCloseTo(svm.REFERENCE.overlapC1.accuracy, 4);
    expect(1 - svm.errors(svm.train(svm.MIDDLE.data, 1)) / 150).toBeCloseTo(svm.REFERENCE.ringLinear.accuracy, 4);
    const rbf = svm.train(svm.MIDDLE.data, 1, { type: 'rbf', gamma: 0.5 });
    expect(1 - svm.errors(rbf) / 150).toBeCloseTo(svm.REFERENCE.ringRbf.accuracy, 4);
    expect(svm.supportCount(rbf)).toBe(svm.REFERENCE.ringRbf.supportVectors);
  });
  it('satisfies the KKT conditions and ignores non-support points', () => {
    expect(hard.alpha.reduce((s, a, i) => s + a * hard.data[i].y, 0)).toBeCloseTo(0, 8);
    const r = svm.roles(hard), free = r.indexOf('free'), support = r.findIndex(x => x !== 'free');
    hard.data.forEach((p, i) => { const v = p.y * svm.decision(hard, p.x); if (r[i] === 'free') expect(v).toBeGreaterThan(1 - 1e-3); else expect(v).toBeCloseTo(1, 3); });
    const without = (i: number) => svm.train(svm.SEPARABLE.data.filter((_, k) => k !== i), svm.HARD_C);
    expect(svm.marginWidth(without(free))).toBeCloseTo(svm.marginWidth(hard), 4);
    expect(svm.marginWidth(without(support))).toBeGreaterThan(svm.marginWidth(hard));
  });
  it('overfits with a very large γ: training accuracy up, cross-validation down', () => {
    const big = svm.train(svm.MIDDLE.data, 1, { type: 'rbf', gamma: 100 });
    expect(1 - svm.errors(big) / 150).toBeGreaterThan(0.99);
    expect(svm.crossValidate(svm.MIDDLE.data, 1, { type: 'rbf', gamma: 100 })).toBeLessThan(svm.crossValidate(svm.MIDDLE.data, 1, { type: 'rbf', gamma: 0.3 }));
  });
});

describe('bias-variance lab (Auto MPG)', () => {
  it('draws reproducible disjoint subsets', () => {
    const a = bv.subset(30, 4), b = bv.subset(30, 4);
    expect(a).toEqual(b);
    expect(new Set([...a.train, ...a.test]).size).toBe(bv.N);
  });
  it('estimates noise and decomposes the error', () => {
    expect(bv.NOISE).toBeCloseTo(18.24, 1);
    const rows = bv.DEGREES.slice(0, 4).map(d => bv.decompose(d, 30));
    for (const r of rows) expect(r.expected).toBeCloseTo(r.bias2 + r.variance + bv.NOISE, 9);
    expect(rows[1].test).toBeLessThan(rows[0].test);
    expect(rows[1].test).toBeLessThan(rows[2].test);
    expect(rows[0].bias2).toBeGreaterThan(4);
    expect(Math.abs(rows[1].test - rows[1].expected) / rows[1].test).toBeLessThan(0.06);
  });
  it('lowers variance with more data', () => {
    expect(bv.decompose(5, 300).variance).toBeLessThan(bv.decompose(5, 30).variance / 100);
  });
});

describe('regularization lab (Diabetes)', () => {
  it('matches the scikit-learn OLS fit and Lasso path', () => {
    reg.ols(reg.FULL).forEach((v, j) => expect(v).toBeCloseTo(reg.REFERENCE.ols[j], 2));
    const path = reg.lassoPath(reg.FULL), alphas = reg.REFERENCE.lassoAlphas;
    reg.REFERENCE.lassoPath.forEach((coef, k) => { const i = reg.LAMBDAS.indexOf(alphas[k]); coef.forEach((v, j) => expect(Math.abs(v - path[i][j])).toBeLessThan(0.5)); });
  });
  it('lets BMI, s5 and blood pressure enter the Lasso path first', () => {
    const path = reg.lassoPath(reg.FULL), order: number[] = [];
    for (let i = reg.LAMBDAS.length - 1; i >= 0; i--) path[i].forEach((v, j) => { if (Math.abs(v) > 1e-9 && !order.includes(j)) order.push(j); });
    expect(order.slice(0, 3)).toEqual([2, 8, 3]);
  });
  it('solves the Lasso optimality conditions', () => {
    const lambda = 1, beta = reg.lasso(reg.FULL, lambda);
    beta.forEach((b, j) => {
      const c = reg.FULL.xty[j] - reg.FULL.gram[j].reduce((s, g, k) => s + g * beta[k], 0);
      if (b === 0) expect(Math.abs(c)).toBeLessThanOrEqual(lambda + 1e-6); else expect(c).toBeCloseTo(lambda * Math.sign(b), 6);
    });
  });
  it('shows s1/s2 instability under OLS and stability under Ridge', () => {
    expect(reg.REFERENCE.corrS1S2).toBeCloseTo(0.897, 3);
    const boot = reg.bootstrap(40, { ridge: 0.1, lasso: 1 });
    const sd = (key: 'ols' | 'ridge', j: number) => { const v = boot.map(b => b[key][j]); const m = v.reduce((a, b) => a + b, 0) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); };
    expect(sd('ols', 4)).toBeGreaterThan(5 * sd('ridge', 4));
  });
});
