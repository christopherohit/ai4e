// InsTaG mechanics lab. Real data: per-Gaussian displacement statistics of the trained InsTaG model (instag/meta.json,
// 25 baked test frames) and the OpenFace/RMS signals of the same 10 s Obama excerpt; held-out metrics of the two
// trained models. Splatting, the identity-free prior and the face–mouth hook are small simulations computed here.
import meta from '../../_shared/radiance/data/trained/instag/meta.json';
import signals from '../../_shared/radiance/data/trained/clip/signals.json';

export const META = meta as { faceCount: number; mouthCount: number; frames: { index: number; meanDisplacement: { face: number; mouth: number } }[] };
export const SIG = signals as { fps: number; mouth: number[]; eye: number[]; rms: number[] };

export function corr(a: readonly number[], b: readonly number[]) {
  const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return sab / Math.sqrt(saa * sbb);
}
function rng(seed: number) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());

// ---- 1. one pixel of Gaussian splatting ----
export interface Splat { x: number; y: number; sx: number; sy: number; rot: number; depth: number; opacity: number; color: string }
export const SPLATS: Splat[] = [
  { x: 0.42, y: 0.52, sx: 0.16, sy: 0.07, rot: 0.5, depth: 1.2, opacity: 0.85, color: '#ffb648' },
  { x: 0.55, y: 0.45, sx: 0.09, sy: 0.09, rot: 0, depth: 1.5, opacity: 0.7, color: '#4fd6ff' },
  { x: 0.5, y: 0.6, sx: 0.2, sy: 0.05, rot: -0.3, depth: 1.9, opacity: 0.9, color: '#ff5fa8' },
  { x: 0.62, y: 0.58, sx: 0.08, sy: 0.14, rot: 0.2, depth: 1.05, opacity: 0.6, color: '#7af0a0' },
  { x: 0.36, y: 0.4, sx: 0.12, sy: 0.12, rot: 0, depth: 2.3, opacity: 0.95, color: '#a98bff' },
  { x: 0.48, y: 0.48, sx: 0.25, sy: 0.18, rot: 0.9, depth: 2.8, opacity: 0.5, color: '#5b8cff' },
];
/** Front-to-back alpha compositing of the splats at pixel (px, py): αᵢ = oᵢ·exp(−½ dᵀΣ⁻¹d), wᵢ = Tᵢ·αᵢ. */
export function compositeAt(px: number, py: number) {
  let T = 1;
  return [...SPLATS].map((s, k) => ({ s, k })).sort((a, b) => a.s.depth - b.s.depth).map(({ s, k }) => {
    const c = Math.cos(s.rot), sn = Math.sin(s.rot), dx = px - s.x, dy = py - s.y;
    const u = c * dx + sn * dy, v = -sn * dx + c * dy;
    const a = s.opacity * Math.exp(-0.5 * ((u / s.sx) ** 2 + (v / s.sy) ** 2));
    const out = { k, color: s.color, depth: s.depth, alpha: a, T, w: T * a };
    T *= 1 - a;
    return out;
  });
}

// ---- 2. identity-free prior and few-shot adaptation (simulation) ----
// Mouth opening y as a function of an audio feature a ∈ [0, 1]: a shared shape g, warped and scaled per identity.
const g = (a: number) => 1 / (1 + Math.exp(-9 * (a - 0.5)));
const person = (scale: number, shift: number, wiggle: number) => (a: number) => scale * g(a - shift) + wiggle * Math.sin(6 * a);
export const IDENTITIES = [[0.8, 0.05, 0.02], [1.1, -0.06, -0.015], [0.95, 0.0, 0.01], [1.2, 0.08, -0.02], [0.9, -0.03, 0.015]].map(([s, h, w]) => person(s, h, w));
/** Universal motion field learned from the pretraining identities (their average; personal residuals cancel). */
export const PRIOR = (a: number) => IDENTITIES.reduce((s, f) => s + f(a), 0) / IDENTITIES.length;
export const TARGET = person(1.35, 0.1, 0.0);
export const NOISE = 0.05;
/** n frames of a short target clip: they only cover part of the audio range (a ∈ [0.15, 0.65]). */
export function samples(n: number, seed = 11) { const r = rng(seed); return Array.from({ length: n }, () => { const a = 0.15 + 0.5 * r(); return [a, TARGET(a) + NOISE * gauss(r)] as [number, number]; }); }
export const GRID = Array.from({ length: 101 }, (_, i) => i / 100);

function solve(A: number[][], b: number[]) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((row, i) => row[n] / row[i]);
}
/** From scratch: degree-5 polynomial by ridge least squares on the few target frames. */
export function fitScratch(pts: readonly [number, number][], degree = 5, ridge = 1e-4) {
  const d = degree + 1, A = Array.from({ length: d }, () => new Array(d).fill(0)), b = new Array(d).fill(0);
  for (const [a, y] of pts) { const p = Array.from({ length: d }, (_, k) => a ** k); for (let i = 0; i < d; i++) { b[i] += p[i] * y; for (let j = 0; j < d; j++) A[i][j] += p[i] * p[j]; } }
  for (let i = 0; i < d; i++) A[i][i] += ridge;
  const w = solve(A, b);
  return (a: number) => w.reduce((s, c, k) => s + c * a ** k, 0);
}
/** Prior + alignment: y ≈ τ · U(a − δ), only an offset δ (where to query the prior) and a scale τ are learned. */
export function fitAligned(pts: readonly [number, number][]) {
  let best = { err: Infinity, tau: 1, delta: 0 };
  for (let delta = -0.25; delta <= 0.25; delta += 0.005) {
    const u = pts.map(([a]) => PRIOR(a - delta)), num = pts.reduce((s, [, y], i) => s + y * u[i], 0), den = u.reduce((s, v) => s + v * v, 0);
    // τ has a closed form; the alignment starts at identity (τ = 1, δ = 0) and is kept near it, like weight decay.
    const k = (NOISE / 0.3) ** 2, tau = (num + k) / (den + k);
    const err = pts.reduce((s, [, y], i) => s + (y - tau * u[i]) ** 2, 0) + k * (tau - 1) ** 2 + (NOISE / 0.1) ** 2 * delta ** 2;
    if (err < best.err) best = { err, tau, delta };
  }
  const { tau, delta } = best;
  return Object.assign((a: number) => tau * PRIOR(a - delta), { tau, delta });
}
/** Error over the whole audio range, including sounds the short clip never showed. */
export const testError = (fn: (a: number) => number) => Math.sqrt(GRID.reduce((s, a) => s + (fn(a) - TARGET(a)) ** 2, 0) / GRID.length);
/** Median test error over 101 random short clips of n frames (one lucky clip proves nothing). */
export function medianError(n: number, fit: (pts: [number, number][]) => (a: number) => number) {
  const e = Array.from({ length: 101 }, (_, k) => testError(fit(samples(n, 100 + k)))).sort((a, b) => a - b);
  return e[50];
}
export const COUNTS = [3, 5, 10, 20, 40, 80];
export const CURVE = COUNTS.map(n => ({ n, scratch: medianError(n, fitScratch), aligned: medianError(n, fitAligned) }));

// ---- 3. real motion field statistics ----
export const MOTION = META.frames.map(f => ({ frame: f.index, face: f.meanDisplacement.face, mouth: f.meanDisplacement.mouth, au25: SIG.mouth[f.index], rms: SIG.rms[f.index] }));
export const MOTION_CORR = { mouthAu: corr(MOTION.map(m => m.mouth), MOTION.map(m => m.au25)), faceAu: corr(MOTION.map(m => m.face), MOTION.map(m => m.au25)) };

// ---- 4. face–mouth hook (simulation driven by the real AU25 lip opening) ----
export const N = SIG.mouth.length;
const LIP = SIG.mouth.map(v => v / Math.max(...SIG.mouth));
const r4 = rng(3);
/** Without the hook the inside-mouth branch guesses its opening from audio alone, learned from a few seconds. */
export const MOUTH_FREE = LIP.map((_, t) => Math.max(0, Math.min(1, 0.55 * (SIG.rms[Math.min(N - 1, t + 6)] / 0.1) + 0.12 * gauss(r4))));
/** With the hook it reads the face branch's deformation range (μ_dist) and scales its own motion by it. */
export const MOUTH_HOOK = LIP.map(v => Math.max(0, Math.min(1, 0.97 * v + 0.02)));
export const LIP_OPEN = LIP;
export const mismatch = (m: readonly number[]) => Math.sqrt(m.reduce((s, v, i) => s + (v - LIP[i]) ** 2, 0) / m.length);

// ---- 5. held-out results of the two trained models (docs/development/radiance-labs.md) ----
export const RESULTS = [
  { name: 'SyncTalk', frames: 7272, psnr: 35.49, lpips: 0.0178, lmd: 2.73, test: 728, note: 'học từ đầu trên video của người cần dựng' },
  { name: 'InsTaG', frames: 250, psnr: 33.63, lpips: 0.0266, lmd: 2.93, test: 727, note: 'prior từ 5 người khác + 10 giây của người cần dựng' },
];
