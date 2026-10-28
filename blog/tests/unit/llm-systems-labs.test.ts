import { describe, expect, it } from 'vitest';
import { FIELDS, JEV, kvBytes, parallelJson } from '../../../tools/kien-thuc/jev-explainer/model';
import { BLOCK, BLOCKS, INFER, sharing, simulate } from '../../../tools/kien-thuc/paged-attention-explainer/model';
import { GPUS, M, OUTPUTS, busy, continuousBatching, decodeIntensity, ridge, staticBatching } from '../../../tools/kien-thuc/inference-engineering-explainer/model';

describe('Jev lab data', () => {
  it('keeps normalised probabilities over the valid values of every field', () => {
    for (const d of JEV.docs) for (const f of FIELDS) {
      const c = d.parallel.fields[f].candidates;
      expect(c.map(x => x.value)).toEqual(JEV.schema[f]);
      expect(c.reduce((s, x) => s + x.p, 0)).toBeCloseTo(1, 3);
      expect(d.parallel.fields[f].hidden).toHaveLength(JEV.hidden);
      expect(JSON.parse(parallelJson(d))[f]).toBeDefined();
    }
  });
  it('records the measured autoregressive runs and the KV cache formula', () => {
    expect(JEV.docs.every(d => d.autoregressive.forwardPasses > 10 && d.parallel.forwardPasses === 2)).toBe(true);
    expect(kvBytes(196)).toBe(2 * 28 * 2 * 128 * 196 * 2);
  });
});

describe('PagedAttention simulator', () => {
  it('matches the measured KV bytes per token with the formula', () => {
    expect(INFER.kvBytesPerTokenFp32).toBe(2 * 24 * 2 * 64 * 4);
  });
  it('never over-allocates and pages beat contiguous reservation', () => {
    const runs = (['reserve', 'exact', 'paged'] as const).map(simulate);
    for (const r of runs) for (const s of r.steps) { expect(s.allocated).toBeLessThanOrEqual(BLOCKS); expect(s.stored).toBeLessThanOrEqual(s.allocated * BLOCK); }
    const generated = runs.map(r => r.steps.at(-1)!.generated);
    expect(generated[2]).toBeGreaterThan(generated[1]);
    expect(generated[1]).toBeGreaterThan(generated[0]);
  });
  it('counts shared prefix blocks', () => {
    expect(sharing(100, 4, 60)).toMatchObject({ separate: 40, shared: 22 });
  });
});

describe('inference engineering model', () => {
  it('places decode far below the H100 ridge point', () => {
    expect(ridge(GPUS[1])).toBeCloseTo(295, 0);
    expect(decodeIntensity(7.6e9, 1)).toBeCloseTo(1, 6);
    expect(decodeIntensity(7.6e9, 64)).toBeCloseTo(64, 6);
  });
  it('finishes the same work sooner with continuous batching', () => {
    const s = staticBatching(OUTPUTS, 6), c = continuousBatching(OUTPUTS, 6);
    expect(c.makespan).toBeLessThan(s.makespan);
    expect(busy(c, 6)).toBeGreaterThan(busy(s, 6));
    expect(M.batch.at(-1)!.tokensPerSecond).toBeGreaterThan(M.batch[0].tokensPerSecond * 5);
  });
});
