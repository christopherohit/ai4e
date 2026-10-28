// SyncTalk mechanics lab. Real signals: OpenFace AU25 (lips apart), AU45 (blink), AU01 (brow) and audio RMS of the
// 10 s Obama test excerpt already used by the SyncTalk/InsTaG labs. Head-pose, tri-plane memory, ray and compositing
// chapters are small simulations of the mechanisms, computed here.
import signals from '../../_shared/radiance/data/trained/clip/signals.json';

export const SIG = signals as { fps: number; mouth: number[]; eye: number[]; brow: number[]; rms: number[] };
export const N = SIG.mouth.length;

export function corr(a: readonly number[], b: readonly number[]) {
  const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return sab / Math.sqrt(saa * sbb);
}
/** Correlation of mouth opening at frame t with audio loudness at frame t + lag (lag > 0: sound comes later). */
export function lagCorr(lag: number, a = SIG.mouth, b = SIG.rms) {
  return lag >= 0 ? corr(a.slice(0, a.length - lag), b.slice(lag)) : corr(a.slice(-lag), b.slice(0, b.length + lag));
}
export const LAGS = Array.from({ length: 21 }, (_, i) => i - 10);
export const LAG_CORR = LAGS.map(k => lagCorr(k));
export const BEST_LAG = LAGS[LAG_CORR.indexOf(Math.max(...LAG_CORR))];
export const PAIRS = [
  { label: 'môi ↔ âm (lệch tốt nhất)', value: Math.max(...LAG_CORR) },
  { label: 'môi ↔ âm (cùng frame)', value: lagCorr(0) },
  { label: 'chớp mắt ↔ âm', value: corr(SIG.eye, SIG.rms) },
  { label: 'lông mày ↔ âm', value: corr(SIG.brow, SIG.rms) },
  { label: 'chớp mắt ↔ môi', value: corr(SIG.eye, SIG.mouth) },
];
export const BLINKS = SIG.eye.reduce((n, v, i) => n + (i > 0 && v > 1 && SIG.eye[i - 1] <= 1 ? 1 : 0), 0);

// ---- head pose: noisy landmarks vs smoothing vs landmark + optical-flow fusion (simulation) ----
function rng(seed: number) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
const r = rng(5);
/** True yaw in degrees: slow sway plus a quick turn around frame 140. */
export const YAW = Array.from({ length: N }, (_, t) => 5 * Math.sin(2 * Math.PI * t / 140) + 9 / (1 + Math.exp(-(t - 142) / 2.2)));
export const LANDMARK_SIGMA = 1.6, FLOW_SIGMA = 0.15;
export const RAW = YAW.map(v => v + LANDMARK_SIGMA * gauss(r));
export const FLOW = YAW.slice(1).map((v, t) => v - YAW[t] + FLOW_SIGMA * gauss(r));
export function movingAverage(x: readonly number[], w: number) {
  const h = Math.floor(w / 2);
  return x.map((_, i) => { let s = 0, n = 0; for (let j = Math.max(0, i - h); j <= Math.min(x.length - 1, i + h); j++) { s += x[j]; n++; } return s / n; });
}
/** argmin Σ (pₜ − rawₜ)² + λ Σ (pₜ₊₁ − pₜ − flowₜ)²: absolute but noisy landmarks + precise relative motion. */
export function fuse(raw: readonly number[], flow: readonly number[], lambda = (LANDMARK_SIGMA / FLOW_SIGMA) ** 2) {
  const n = raw.length, a = new Array(n).fill(0), b = new Array(n).fill(1), c = new Array(n).fill(0), d = raw.slice();
  for (let t = 0; t < n - 1; t++) {
    b[t] += lambda; b[t + 1] += lambda; c[t] = -lambda; a[t + 1] = -lambda;
    d[t] -= lambda * flow[t]; d[t + 1] += lambda * flow[t];
  }
  for (let i = 1; i < n; i++) { const m = a[i] / b[i - 1]; b[i] -= m * c[i - 1]; d[i] -= m * d[i - 1]; }
  const p = new Array(n); p[n - 1] = d[n - 1] / b[n - 1];
  for (let i = n - 2; i >= 0; i--) p[i] = (d[i] - c[i] * p[i + 1]) / b[i];
  return p as number[];
}
export const FUSED = fuse(RAW, FLOW);
export const rmse = (x: readonly number[], y: readonly number[] = YAW) => Math.sqrt(x.reduce((s, v, i) => s + (v - y[i]) ** 2, 0) / x.length);
/** Frame-to-frame jitter: RMS of the error in per-frame motion. */
export const jitter = (x: readonly number[]) => Math.sqrt(x.slice(1).reduce((s, v, i) => s + ((v - x[i]) - (YAW[i + 1] - YAW[i])) ** 2, 0) / (x.length - 1));

// ---- feature-grid memory (float16 features, F = 2 per level as in Instant-NGP) ----
export const F = 2, LEVELS = 16, TABLE = 2 ** 19, BYTES = 2;
export function memory(res: number) {
  const levels = Array.from({ length: LEVELS }, (_, l) => Math.round(16 * (res / 16) ** (l / (LEVELS - 1))));
  return {
    dense3d: levels.reduce((s, n) => s + n ** 3, 0) * F * BYTES,
    triplane: 3 * levels.reduce((s, n) => s + n ** 2, 0) * F * BYTES,
    hash3d: levels.reduce((s, n) => s + Math.min(TABLE, n ** 3), 0) * F * BYTES,
    triHash: 3 * levels.reduce((s, n) => s + Math.min(TABLE, n ** 2), 0) * F * BYTES,
  };
}

// ---- one ray through the face (simulation of the NeRF quadrature) ----
export const RAY = (() => {
  const S = 64, t0 = 0.2, t1 = 1.2, dt = (t1 - t0) / S, out: { t: number; sigma: number; alpha: number; T: number; w: number }[] = [];
  let T = 1;
  for (let i = 0; i < S; i++) {
    const t = t0 + (i + 0.5) * dt;
    const sigma = 0.3 + 160 * Math.exp(-(((t - 0.62) / 0.025) ** 2)) + 35 * Math.exp(-(((t - 0.85) / 0.04) ** 2));
    const alpha = 1 - Math.exp(-sigma * dt);
    out.push({ t, sigma, alpha, T, w: T * alpha });
    T *= 1 - alpha;
  }
  return out;
})();

// ---- compositing the rendered head into the original frame: 1-D cross-section from chin (rendered) to neck (original) ----
export const XS = Array.from({ length: 121 }, (_, i) => i / 120);
export const RENDER = XS.map(x => 0.6 + 0.04 * Math.sin(x * 7));   // rendered skin, a slightly different shade
export const ORIGINAL = XS.map(x => 0.7 + 0.04 * Math.sin(x * 7 + 0.6) + 0.012 * Math.sin(x * 61));  // real neck skin with texture
export function mask(blur: number) {
  return XS.map(x => blur <= 0 ? (x < 0.5 ? 1 : 0) : 1 / (1 + Math.exp((x - 0.5) / (blur / 4))));
}
export const composite = (m: readonly number[]) => XS.map((_, i) => m[i] * RENDER[i] + (1 - m[i]) * ORIGINAL[i]);
/** Largest jump between neighbouring samples: a visible seam. */
export const seam = (c: readonly number[]) => Math.max(...c.slice(1).map((v, i) => Math.abs(v - c[i])));
