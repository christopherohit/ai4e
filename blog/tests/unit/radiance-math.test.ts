import { describe, expect, it } from 'vitest';
import { composite, volumeAlpha, projectedCovariance, gaussianAlpha } from '../../../tools/_shared/radiance/math';

describe('radiance lab numerical foundations', () => {
  it('conserves transmission and weights', () => {
    const rendered = composite([{ alpha: 0.5, color: [1, 0, 0] }, { alpha: 0.5, color: [0, 1, 0] }]);
    expect(rendered.color).toEqual([0.5, 0.25, 0]);
    expect(rendered.weights.reduce((sum, value) => sum + value, rendered.transmission)).toBeCloseTo(1);
  });
  it('is invariant to subdivisions of constant density', () => {
    expect((1 - volumeAlpha(2, 0.1)) ** 10).toBeCloseTo(1 - volumeAlpha(2, 1));
    expect(volumeAlpha(0, 1)).toBe(0);
  });
  it('projects covariance in pixel squared units', () => {
    expect(projectedCovariance([0, 0, 2], [[1, 0, 0], [0, 2, 0], [0, 0, 3]], 100)).toEqual([[2500, 0], [0, 5000]]);
    expect(() => projectedCovariance([0, 0, 0], [[1]], 1)).toThrow();
  });
  it('evaluates a Gaussian kernel, not a normalized probability density', () => {
    expect(gaussianAlpha([0, 0], [[4, 0], [0, 1]], 0.6)).toBe(0.6);
    expect(gaussianAlpha([2, 0], [[4, 0], [0, 1]], 0.6)).toBeCloseTo(0.6 * Math.exp(-0.5));
    expect(gaussianAlpha([0, 0], [[0, 0], [0, 0]], 0.6)).toBe(0);
  });
});
