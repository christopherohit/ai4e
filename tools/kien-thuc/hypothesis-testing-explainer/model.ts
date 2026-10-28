// Hypothesis-testing lab: real datasets (scripts/ml-datasets/prepare_stats.py), the t / normal / chi-square
// distributions computed from scratch, an exact sign-flip randomisation test and seeded simulations of power and p-hacking.
import data from './data/stats.json';

export const S = data as {
  sleep: { source: string; drug1: number[]; drug2: number[]; t: number; p: number; df: number };
  mpg: { source: string; japan: number[]; europe: number[]; welch: { t: number; p: number }; student: { t: number; p: number } };
  titanic: { source: string; female: [number, number]; male: [number, number]; z: number; pz: number; classTable: number[][]; chi2: number; pChi2: number; dof: number; expected: number[][] };
};

export const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
export const variance = (a: number[]) => { const m = mean(a); return a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1); };

// ---- special functions -------------------------------------------------------------------------------------------
export function lgamma(x: number): number {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, tmp = x + 5.5, ser = 1.000000000190015;
  tmp -= (x + 0.5) * Math.log(tmp);
  for (const v of c) ser += v / ++y;
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}
function betacf(a: number, b: number, x: number) {
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap;
  d = 1 / (Math.abs(d) < 1e-300 ? 1e-300 : d);
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; d = 1 / (Math.abs(d) < 1e-300 ? 1e-300 : d); c = 1 + aa / c; c = Math.abs(c) < 1e-300 ? 1e-300 : c; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; d = 1 / (Math.abs(d) < 1e-300 ? 1e-300 : d); c = 1 + aa / c; c = Math.abs(c) < 1e-300 ? 1e-300 : c;
    const del = d * c; h *= del;
    if (Math.abs(del - 1) < 1e-14) break;
  }
  return h;
}
/** Regularised incomplete beta I_x(a, b). */
export function betai(a: number, b: number, x: number) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
export const tPdf = (t: number, df: number) => Math.exp(lgamma((df + 1) / 2) - lgamma(df / 2)) / Math.sqrt(df * Math.PI) * (1 + t * t / df) ** (-(df + 1) / 2);
/** P(T ≥ t) for Student's t with df degrees of freedom. */
export const tSf = (t: number, df: number) => { const x = df / (df + t * t), tail = 0.5 * betai(df / 2, 0.5, x); return t >= 0 ? tail : 1 - tail; };
export const normPdf = (z: number) => Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI);
export function normSf(z: number) {
  // Abramowitz–Stegun 7.1.26 via erfc, accurate to ~1e-7; fine for drawing and for moderate z.
  const x = Math.abs(z) / Math.SQRT2, t = 1 / (1 + 0.3275911 * x);
  const erfc = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-x * x);
  return z >= 0 ? erfc / 2 : 1 - erfc / 2;
}
/** Regularised incomplete gamma: lower P(a, x) by series, upper Q(a, x) by continued fraction (no cancellation). */
function gammaSeries(a: number, x: number) {
  let sum = 1 / a, del = sum, ap = a;
  for (let n = 0; n < 500; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break; }
  return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
}
function gammaCf(a: number, x: number) {
  let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 500; i++) { const an = -i * (i - a); b += 2; d = an * d + b; d = Math.abs(d) < 1e-300 ? 1e-300 : d; c = b + an / c; c = Math.abs(c) < 1e-300 ? 1e-300 : c; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-15) break; }
  return Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
}
export const gammaP = (a: number, x: number) => (x <= 0 ? 0 : x < a + 1 ? gammaSeries(a, x) : 1 - gammaCf(a, x));
export const gammaQ = (a: number, x: number) => (x <= 0 ? 1 : x < a + 1 ? 1 - gammaSeries(a, x) : gammaCf(a, x));
export const chi2Pdf = (x: number, k: number) => (x <= 0 ? 0 : Math.exp((k / 2 - 1) * Math.log(x) - x / 2 - (k / 2) * Math.LN2 - lgamma(k / 2)));
export const chi2Sf = (x: number, k: number) => gammaQ(k / 2, x / 2);

// ---- the sleep data: paired t-test and an exact sign-flip test --------------------------------------------------
export const DIFF = S.sleep.drug2.map((v, i) => +(v - S.sleep.drug1[i]).toFixed(2));
export const OBS = mean(DIFF);
export function pairedT(d: number[]) { const m = mean(d), se = Math.sqrt(variance(d) / d.length); return { t: m / se, se, df: d.length - 1 }; }
/** Means of all 2^n sign flips of the differences: the exact null distribution if the drug labels are exchangeable. */
export function signFlipMeans(d = DIFF) {
  const out: number[] = [];
  for (let mask = 0; mask < 2 ** d.length; mask++) out.push(d.reduce((s, v, i) => s + (mask >> i & 1 ? -v : v), 0) / d.length);
  return out;
}

// ---- seeded simulation ---------------------------------------------------------------------------------------------
export function rng(seed: number) {
  let a = seed >>> 0;
  const uniform = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const normal = () => Math.sqrt(-2 * Math.log(uniform() + 1e-12)) * Math.cos(2 * Math.PI * uniform());
  return { uniform, normal };
}
/** p-values of `runs` two-sample t-tests (n per group, true standardised effect d), equal-variance t. */
export function simulateP(n: number, d: number, runs: number, seed = 1) {
  const r = rng(seed), out: number[] = [];
  for (let k = 0; k < runs; k++) {
    const a = Array.from({ length: n }, () => r.normal()), b = Array.from({ length: n }, () => r.normal() + d);
    const sp = Math.sqrt((variance(a) + variance(b)) / 2), t = (mean(b) - mean(a)) / (sp * Math.sqrt(2 / n));
    out.push(2 * tSf(Math.abs(t), 2 * n - 2));
  }
  return out;
}
/** Power of the two-sided two-sample t-test (normal approximation of the noncentral t). */
export function power(n: number, d: number, alpha = 0.05) {
  const zc = criticalZ(alpha / 2), shift = d * Math.sqrt(n / 2);
  return normSf(zc - shift) + normSf(zc + shift);
}
export function criticalZ(tail: number) { let lo = 0, hi = 10; for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (normSf(m) > tail) lo = m; else hi = m; } return (lo + hi) / 2; }
export function criticalT(tail: number, df: number) { let lo = 0, hi = 60; for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (tSf(m, df) > tail) lo = m; else hi = m; } return (lo + hi) / 2; }
