import { describe, expect, it } from 'vitest';
import { outer, readOut, softmax } from '../../../tools/_shared/sequence-lab/math';
import { ANSWERS, TOKENS as GDN_TOKENS, read, simulate } from '../../../tools/kien-thuc/gated-delta-net-explainer/model';
import { BLOCK, LENGTH, allowed, completeBlocks, coverage, indexerScores, tailStart, topBlocks, weights } from '../../../tools/kien-thuc/sparse-attention-explainer/model';
import { CONFIG, coreReads, memory, mixerOf, residual, route } from '../../../tools/kien-thuc/qwen3-8-flash-next-explainer/model';
import { scenarios as bswmdScenarios } from '../../../tools/du-an/bswmd/model';

const best = (values: number[]) => ANSWERS[values.indexOf(Math.max(...values))];

describe('sequence lab math', () => {
  it('reads Sᵀq with S indexed [key][value]', () => {
    expect(readOut(outer([1, 0], [0.2, 0.8]), [1, 0])).toEqual([0.2, 0.8]);
  });
  it('gives masked scores exactly zero weight', () => {
    const weights = softmax([1, -Infinity, 1]);
    expect(weights).toEqual([0.5, 0, 0.5]);
  });
});

describe('gated delta rule', () => {
  it('matches the article one-dimensional example', () => {
    const [alpha, beta, s, k, v] = [0.9, 0.5, 0.8, 1, 0.2];
    const decayed = alpha * s;
    expect(decayed + beta * k * (v - decayed * k)).toBeCloseTo(0.46);
  });
  it('overwrites the outdated capital, while linear attention mixes both', () => {
    const steps = simulate(0.95, 0.9);
    const last = steps.at(-1)!;
    expect(best(read(last.state, 'capital'))).toBe('Hà Nội');
    const linear = read(last.linear, 'capital');
    expect(Math.abs(linear[0] - linear[1])).toBeLessThan(0.15);
    expect(last.gdnError).toBeLessThan(last.linearError);
  });
  it('produces a negative error on the old answer when a fact changes', () => {
    const update = simulate(0.95, 0.9)[GDN_TOKENS.findIndex(token => token.text.includes('Hà Nội'))];
    expect(update.error[0]).toBeLessThan(-0.3);
    expect(update.error[1]).toBeGreaterThan(0.5);
  });
  it('writes nothing when β = 0 and keeps the state when α = 1, β = 0', () => {
    expect(simulate(1, 0).every(step => step.state.flat().every(value => value === 0))).toBe(true);
  });
});

describe('qwen sparse attention toy', () => {
  it('only scores complete blocks before the query', () => {
    expect(indexerScores(14).filter(score => score !== null)).toHaveLength(3);
    expect(indexerScores(15).filter(score => score !== null)).toHaveLength(4);
    expect(completeBlocks(14)).toBe(3);
    expect(tailStart(14)).toBe(12);
  });
  it('keeps the tail and at most budget·r complete-block tokens', () => {
    for (let i = BLOCK - 1; i < LENGTH; i++) {
      for (const budget of [1, 2, 3]) {
        const keep = allowed('qsa', i, budget);
        expect(keep.filter(j => j < tailStart(i)).length).toBeLessThanOrEqual(budget * BLOCK);
        for (let j = tailStart(i); j <= i; j++) expect(keep).toContain(j);
        expect(keep.every(j => j <= i)).toBe(true);
        expect(topBlocks(i, budget).length).toBeLessThanOrEqual(budget);
      }
    }
  });
  it('normalises core weights over the kept keys only', () => {
    const keep = allowed('qsa', 28, 2);
    const row = weights(28, keep);
    expect(row.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1);
    row.forEach((value, j) => { if (!keep.includes(j)) expect(value).toBe(0); });
  });
  it('lets content-based selection reach a distant definition that a local window misses', () => {
    expect(coverage('qsa', 28, 2)).toBeGreaterThan(coverage('local', 28, 2) + 0.4);
    expect(coverage('dense', 28, 2)).toBeCloseTo(1);
  });
});

describe('qwen architecture lab', () => {
  it('follows the 3 GDN + 1 QSA schedule', () => {
    const mixers = Array.from({ length: CONFIG.layers }, (_, index) => mixerOf(index + 1));
    expect(mixers.filter(mixer => mixer === 'QSA')).toHaveLength(12);
    expect(mixers.slice(0, 4)).toEqual(['GDN', 'GDN', 'GDN', 'QSA']);
  });
  it('keeps GDN memory constant and QSA reads bounded by budget + tail', () => {
    expect(memory(4096).gdnPart).toBe(memory(1_048_576).gdnPart);
    expect(memory(262_144).hybrid).toBeLessThan(memory(262_144).allAttention / 3);
    expect(coreReads(1000)).toBe(1000);
    expect(coreReads(262_144)).toBeLessThanOrEqual(CONFIG.qsa.budget + CONFIG.qsa.block - 1);
  });
  it('routes each token to 10 distinct experts with weights summing to 1', () => {
    const routed = route('attention', 4);
    expect(new Set(routed.map(item => item.expert)).size).toBe(10);
    expect(routed.reduce((sum, item) => sum + item.weight, 0)).toBeCloseTo(1);
    expect(route('attention', 4)).toEqual(routed);
  });
  it('writes the same sublayer output into every branch with its own scalar', () => {
    const result = residual([1, 1, 1, 1], [1.5, 0.5, 1, 0.2]);
    result.gates.flat().forEach(gate => { expect(gate).toBeGreaterThan(0); expect(gate).toBeLessThan(1); });
    const delta = (branch: number) => result.updated[branch].map((value, channel) => value - residual([1, 1, 1, 1], [0, 0, 0, 0]).updated[branch][channel]);
    delta(0).forEach((value, channel) => expect(value).toBeCloseTo(1.5 * result.y[channel]));
  });
});

describe('sdvx modeler lab', async () => {
  const { DATA, PROPOSALS, STAGES, run } = await import('../../../tools/du-an/sdvx-modeler/model');
  it('lists the 13 validation stages in pipeline order', () => {
    expect(STAGES.map(stage => stage.id)).toEqual(['WORKSPACE_ADMISSIBILITY', 'STRICT_ROUNDTRIP', 'XSD_SCHEMA', 'REFERENCE_INTEGRITY', 'ECUC_CONSTRAINT', 'ECUC_CONFORMANCE', 'DOMAIN_INVARIANT', 'SCOPED_BASELINE', 'DUPLICATE_IDENTIFIER', 'EFFECT_DIFF', 'CLOSURE_ACCOUNTING', 'TEXTUAL_CHURN', 'EXTERNAL_TOOLCHAIN']);
  });
  it('stops a refused proposal at exactly one stage and runs nothing after it', () => {
    for (const proposal of PROPOSALS.filter(item => item.stopsAt)) {
      const outcome = run(proposal);
      const stop = STAGES.findIndex(stage => stage.id === proposal.stopsAt);
      STAGES.forEach((stage, index) => expect(outcome[stage.id]).toBe(index < stop ? 'PASS' : index === stop ? 'FAIL' : 'NOT_RUN'));
    }
  });
  it('commits the valid proposal and reports the missing toolchain as skipped, not passed', () => {
    const outcome = run(PROPOSALS.find(item => !item.stopsAt)!);
    expect(Object.values(outcome).filter(value => value === 'FAIL')).toHaveLength(0);
    expect(outcome.EXTERNAL_TOOLCHAIN).toBe('SKIPPED');
  });
  it('keeps measured figures consistent with the sources', () => {
    expect(DATA.fileRoles.reduce((sum, [, count]) => sum + count, 0)).toBe(101);
    expect(DATA.dialects.reduce((sum, count) => sum + count, 0)).toBe(107);
    expect(DATA.eval40.dsh.pass).toBeGreaterThan(DATA.eval40.legacy.pass);
    expect(DATA.tokens.prompt / (DATA.tokens.prompt + DATA.tokens.completion)).toBeCloseTo(0.946, 3);
  });
});

describe('bswmd lab', () => {
  it('keeps authority visible in each validation outcome', () => {
    expect(bswmdScenarios.valid.status).toBe('PASS');
    expect(bswmdScenarios.invalid.status).toBe('FAIL');
    expect(bswmdScenarios.missing.status).toBe('DEGRADED');
    expect(bswmdScenarios.missing.authority).toBe('OBSERVED_CONGRUENCE');
  });
});
