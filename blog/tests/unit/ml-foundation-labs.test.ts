import { describe, expect, it } from 'vitest';
import { DIFF, S, chi2Sf, criticalT, criticalZ, mean, normSf, pairedT, power, signFlipMeans, simulateP, tSf, variance } from '../../../tools/kien-thuc/hypothesis-testing-explainer/model';
import { D, NEG, POS, metrics, pr, precisionAt, roc } from '../../../tools/kien-thuc/classifier-metrics-explainer/model';

describe('hypothesis testing lab', () => {
  it('reproduces the SciPy paired t-test on the sleep data', () => {
    const { t, df } = pairedT(DIFF);
    expect(t).toBeCloseTo(S.sleep.t, 4);
    expect(2 * tSf(Math.abs(t), df)).toBeCloseTo(S.sleep.p, 6);
  });
  it('reproduces the Welch test on Auto MPG', () => {
    const a = S.mpg.japan, b = S.mpg.europe, va = variance(a) / a.length, vb = variance(b) / b.length;
    const t = (mean(a) - mean(b)) / Math.sqrt(va + vb);
    const df = (va + vb) ** 2 / (va ** 2 / (a.length - 1) + vb ** 2 / (b.length - 1));
    expect(t).toBeCloseTo(S.mpg.welch.t, 4);
    expect(2 * tSf(t, df)).toBeCloseTo(S.mpg.welch.p, 5);
  });
  it('reproduces the chi-square test and the normal tail', () => {
    expect(chi2Sf(S.titanic.chi2, S.titanic.dof) / S.titanic.pChi2).toBeCloseTo(1, 3);
    expect(normSf(1.959964)).toBeCloseTo(0.025, 6);
    expect(criticalZ(0.025)).toBeCloseTo(1.96, 2);
    expect(criticalT(0.025, 9)).toBeCloseTo(2.262, 3);
  });
  it('enumerates the exact sign-flip distribution', () => {
    const means = signFlipMeans();
    expect(means).toHaveLength(1024);
    const obs = mean(DIFF), extreme = means.filter(m => Math.abs(m) >= Math.abs(obs) - 1e-9).length;
    expect(extreme).toBe(4);
  });
  it('keeps the false-positive rate near alpha under H0 and power rising with n', () => {
    const p = simulateP(30, 0, 2000, 7), fp = p.filter(v => v < 0.05).length / p.length;
    expect(fp).toBeGreaterThan(0.03);
    expect(fp).toBeLessThan(0.07);
    expect(power(100, 0.5)).toBeGreaterThan(power(20, 0.5));
    expect(power(64, 0.5)).toBeCloseTo(0.8, 1);
  });
});

describe('classifier metrics lab', () => {
  it('matches scikit-learn ROC-AUC and average precision', () => {
    expect(POS + NEG).toBe(D.testSize);
    expect(roc().auc).toBeCloseTo(D.rocAuc, 4);
    expect(pr().ap).toBeCloseTo(D.prAuc, 4);
  });
  it('keeps the confusion matrix consistent', () => {
    const m = metrics(0.5);
    expect(m.tp + m.fn).toBe(POS);
    expect(m.fp + m.tn).toBe(NEG);
    expect(m.f1).toBeCloseTo(2 * m.precision * m.recall / (m.precision + m.recall), 10);
    expect(precisionAt(m.recall, m.specificity, POS / (POS + NEG))).toBeCloseTo(m.precision, 10);
  });
});
