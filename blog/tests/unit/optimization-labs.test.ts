import { describe, expect, it } from 'vitest';
import { ALL, CURVATURES, DATA, MAX_STABLE_LR, MIN_LOSS, N, OPTIMUM, epochBatches, gradient, gradientNoise, levelSet, loss, lossPerEpoch, sampleGradients, train, type Point } from '../../../tools/_shared/optimization-lab/model';
import { GLOBAL_MIN, HUMP, LOCAL_MIN, SLICE_CURVATURE, bumpy, contraction, descend1D, descend2D, regime, slice } from '../../../tools/kien-thuc/gradient-descent-explainer/model';
import { START, angle, compare, updatesPerEpoch } from '../../../tools/kien-thuc/mini-batch-sgd-explainer/model';

const numeric = (theta: Point, batch: readonly number[] = ALL): Point => {
  const h = 1e-6;
  return [(loss([theta[0] + h, theta[1]], batch) - loss([theta[0] - h, theta[1]], batch)) / (2 * h), (loss([theta[0], theta[1] + h], batch) - loss([theta[0], theta[1] - h], batch)) / (2 * h)];
};

describe('shared regression problem', () => {
  it('has 64 deterministic samples', () => {
    expect(DATA).toHaveLength(N);
    expect(N).toBe(64);
  });
  it('matches finite differences for full and mini-batch gradients', () => {
    for (const batch of [ALL, [3], [1, 7, 20, 41]]) {
      const [gw, gb] = gradient([0.3, 1.7], batch);
      const [nw, nb] = numeric([0.3, 1.7], batch);
      expect(gw).toBeCloseTo(nw, 5);
      expect(gb).toBeCloseTo(nb, 5);
    }
  });
  it('has zero gradient at the least-squares optimum', () => {
    const [gw, gb] = gradient(OPTIMUM);
    expect(Math.abs(gw) + Math.abs(gb)).toBeLessThan(1e-9);
    expect(MIN_LOSS).toBeGreaterThan(0.2);
  });
  it('draws level sets where the loss equals L* + c', () => {
    for (const point of levelSet(0.5, 12)) expect(loss(point)).toBeCloseTo(MIN_LOSS + 0.5, 6);
  });
  it('quotes the article curvature, threshold and condition number', () => {
    expect(CURVATURES[0]).toBeCloseTo(5.3, 1);
    expect(CURVATURES[1]).toBeCloseTo(0.25, 2);
    expect(MAX_STABLE_LR).toBeCloseTo(0.378, 3);
  });
});

describe('gradient descent lab', () => {
  it('reproduces the first worked step of the article', () => {
    const path = descend1D(slice, -0.5, 0.1, 3);
    expect(slice.df(-0.5)).toBeCloseTo(-6.853, 3);
    expect(path[1]).toBeCloseTo(0.185, 3);
    expect(slice.f(path[1])).toBeCloseTo(2.981, 3);
  });
  it('classifies learning rates by the factor 1 − ηL″', () => {
    expect(SLICE_CURVATURE).toBeCloseTo(3.55, 2);
    expect(contraction(0.45)).toBeCloseTo(-0.6, 2);
    expect([0.05, 0.2, 0.45, 0.6].map(lr => regime(lr))).toEqual(['slow', 'smooth', 'oscillating', 'diverging']);
    expect(slice.f(descend1D(slice, -0.5, 0.6, 10)[10])).toBeGreaterThan(70);
  });
  it('converges in 2D below 2/λmax and diverges above it', () => {
    expect(loss(descend2D([-0.5, 2.6], 0.3, 60)[60])).toBeCloseTo(MIN_LOSS, 3);
    expect(loss(descend2D([-0.5, 2.6], 0.4, 60).at(-1)!)).toBeGreaterThan(1e3);
  });
  it('lands in different valleys from starts on either side of the hump', () => {
    expect(HUMP).toBeCloseTo(0.152, 3);
    expect(bumpy.f(GLOBAL_MIN)).toBeLessThan(bumpy.f(LOCAL_MIN));
    expect(descend1D(bumpy, 0.1, 0.1, 40).at(-1)).toBeCloseTo(GLOBAL_MIN, 2);
    expect(descend1D(bumpy, 0.2, 0.1, 40).at(-1)).toBeCloseTo(LOCAL_MIN, 2);
  });
});

describe('batch, mini-batch and SGD lab', () => {
  it('splits an epoch into ⌈N/B⌉ batches that cover every sample once', () => {
    for (const size of [1, 5, 16, 64]) {
      const batches = epochBatches(size, 3);
      expect(batches).toHaveLength(updatesPerEpoch(size));
      expect(batches.flat().sort((a, b) => a - b)).toEqual(ALL);
    }
  });
  it('makes 1, N/B and N updates per epoch', () => {
    const [batch, mini, sgd] = compare({ lr: 0.1, epochs: 5, mini: 16, seed: 1 });
    expect([batch.run.updates.length, mini.run.updates.length, sgd.run.updates.length]).toEqual([5, 20, 320]);
    expect(batch.run.updates.every(update => angle(update.step, update.full) < 1e-6)).toBe(true);
  });
  it('matches the per-epoch losses quoted in the article', () => {
    const [batch, mini, sgd] = compare({ lr: 0.1, epochs: 5, mini: 16, seed: 1 });
    expect(batch.perEpoch[5]).toBeCloseTo(0.85, 2);
    expect(mini.perEpoch[5]).toBeCloseTo(0.515, 2);
    expect(sgd.perEpoch[1]).toBeCloseTo(0.295, 2);
    expect(lossPerEpoch(train({ batchSize: 64, lr: 0.1, epochs: 10, start: START }), 10)[10]).toBeCloseTo(0.71, 2);
  });
  it('shrinks gradient noise like 1/√B and removes it at B = N', () => {
    expect(gradientNoise(START, 1)).toBeCloseTo(4.36, 2);
    expect(gradientNoise(START, 16)).toBeCloseTo(0.95, 2);
    expect(gradientNoise(START, N)).toBe(0);
    const ratio = gradientNoise(START, 4) / gradientNoise(START, 16);
    expect(ratio).toBeGreaterThan(2);
    expect(ratio).toBeLessThan(2.5);
  });
  it('agrees with the empirical spread of sampled mini-batch gradients', () => {
    const full = gradient(START);
    const samples = sampleGradients(START, 8, 4000, 11);
    const spread = Math.sqrt(samples.reduce((sum, g) => sum + (g[0] - full[0]) ** 2 + (g[1] - full[1]) ** 2, 0) / samples.length);
    expect(spread / gradientNoise(START, 8)).toBeGreaterThan(0.93);
    expect(spread / gradientNoise(START, 8)).toBeLessThan(1.07);
  });
});
