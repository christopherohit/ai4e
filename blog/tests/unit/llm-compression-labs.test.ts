import { describe, expect, it } from 'vitest';
import { MODELS, MOE, balanceLoss, touched } from '../../../tools/kien-thuc/moe-explainer/model';
import { BASE, Q, decodeBound, quantizeVector, rmse } from '../../../tools/kien-thuc/quantization-explainer/model';

describe('mixture-of-experts lab data', () => {
  it('routes every sample to top-k experts with normalised gates', () => {
    for (const v of [MOE.balanced, MOE.unbalanced]) for (const s of v.samples) {
      expect(s.experts).toHaveLength(MOE.topK);
      expect(s.gates.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 3);
      expect(s.routerProbs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 3);
    }
    expect(MOE.balanced.activeParams).toBeLessThan(MOE.balanced.totalParams);
  });
  it('balances usage only with the auxiliary loss', () => {
    const usage = (v: typeof MOE.balanced) => v.history.at(-1)!.usage;
    expect(Math.min(...usage(MOE.balanced))).toBeGreaterThan(0.05);
    expect(Math.min(...usage(MOE.unbalanced))).toBeLessThan(0.01);
  });
  it('keeps the routing formulas and published configurations sane', () => {
    expect(balanceLoss([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25])).toBeCloseTo(1, 10);
    expect(balanceLoss([1, 0, 0, 0], [1, 0, 0, 0])).toBe(4);
    expect(touched(8, 128, 1)).toBeCloseTo(1 / 16, 10);
    expect(touched(8, 128, 256)).toBeGreaterThan(0.99);
    for (const m of MODELS) expect(m.active).toBeLessThan(m.total);
  });
});

describe('quantization lab data', () => {
  it('reproduces the baked RTN quantization of the sample row', () => {
    for (const [bits, group] of [[8, 128], [4, 128], [4, 32], [2, 32]] as const) {
      const ref = (Q.sampleQuantized as Record<string, number[]>)[`${bits}-${group}`];
      const { out } = quantizeVector(Q.sampleRow, bits, group);
      // The baked row is rounded to 5 decimals, so a few values sitting on a rounding boundary may land one step away.
      const off = out.filter((v, i) => Math.abs(v - ref[i]) > 1e-4).length;
      expect(off / out.length).toBeLessThan(0.03);
      expect(rmse(out, Q.sampleRow)).toBeCloseTo(rmse(ref, Q.sampleRow), 3);
    }
  });
  it('orders quality by bits and granularity as measured', () => {
    const ppl = Object.fromEntries(Q.results.map(r => [r.name, r.perplexity]));
    expect(BASE.name).toBe('fp32');
    expect(ppl['int8 per-channel']).toBeLessThan(BASE.perplexity * 1.02);
    expect(ppl['int4 group-32']).toBeLessThan(ppl['int4 group-128']);
    expect(ppl['int4 group-128']).toBeLessThan(ppl['int4 per-channel']);
    expect(ppl['int4 per-channel']).toBeLessThan(ppl['int4 per-tensor']);
  });
  it('bounds decode speed by bandwidth over weight bytes', () => {
    expect(decodeBound(7e9, 16)).toBeCloseTo(3.35e12 / 14e9, 6);
    expect(decodeBound(7e9, 4.125) / decodeBound(7e9, 16)).toBeCloseTo(16 / 4.125, 10);
  });
});
