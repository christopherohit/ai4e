// Toy associative memory for the Gated DeltaNet lab. Keys live in 4 channels, values are one-hot over 4 answers, so
// Sᵀk reads as "how strongly the memory answers X for this key". Everything is computed here; nothing comes from Qwen.
import { addMatrix, l2normalize, norm, outer, readOut, scaleMatrix, subtract, zeros, type Matrix, type Vector } from '../../_shared/sequence-lab/math';

export const ANSWERS = ['Huế', 'Hà Nội', 'Đồng', 'Tiếng Việt'] as const;
export const CHANNELS = ['kênh 1', 'kênh 2', 'kênh 3', 'kênh 4'] as const;
export type Fact = 'capital' | 'currency' | 'language';
export const FACTS: Record<Fact, { label: string; key: Vector }> = {
  capital: { label: 'thủ đô?', key: l2normalize([1, 0.18, 0, 0.12]) },
  currency: { label: 'tiền tệ?', key: l2normalize([0.12, 1, 0.2, 0]) },
  language: { label: 'ngôn ngữ?', key: l2normalize([0, 0.15, 1, 0.22]) },
};
const answer = (index: number) => ANSWERS.map((_, position) => (position === index ? 1 : 0));

export interface Token { text: string; fact?: Fact; k: Vector; v: Vector }
// Huế was the capital from 1802 (Nguyễn dynasty); Hà Nội is today's capital. The repeat at t=7 shows a near-zero write.
export const TOKENS: Token[] = [
  { text: 'thủ đô (1802) → Huế', fact: 'capital', k: FACTS.capital.key, v: answer(0) },
  { text: 'tiền tệ → Đồng', fact: 'currency', k: FACTS.currency.key, v: answer(2) },
  { text: 'ngôn ngữ → Tiếng Việt', fact: 'language', k: FACTS.language.key, v: answer(3) },
  { text: 'câu đệm', k: l2normalize([0.35, 0.3, 0.3, 1]), v: [0.2, 0.2, 0.2, 0.2] },
  { text: 'thủ đô (nay) → Hà Nội', fact: 'capital', k: FACTS.capital.key, v: answer(1) },
  { text: 'câu đệm', k: l2normalize([0.2, 0.4, 0.1, 1]), v: [0.1, 0.3, 0.1, 0.2] },
  { text: 'tiền tệ → Đồng', fact: 'currency', k: FACTS.currency.key, v: answer(2) },
  { text: 'ngôn ngữ → Tiếng Việt', fact: 'language', k: FACTS.language.key, v: answer(3) },
];

export interface Step {
  token: Token;
  prev: Matrix;
  decayed: Matrix;
  prediction: Vector;
  error: Vector;
  write: Matrix;
  state: Matrix;
  linear: Matrix;
  /** Mean ‖Sᵀk − v_latest‖ over the facts seen so far. */
  gdnError: number;
  linearError: number;
}

/** Latest value written for each fact up to and including step t. */
export function targets(t: number) {
  const latest: Partial<Record<Fact, Vector>> = {};
  for (const token of TOKENS.slice(0, t + 1)) if (token.fact) latest[token.fact] = token.v;
  return latest;
}

export function retrievalError(state: Matrix, t: number) {
  const known = Object.entries(targets(t)) as [Fact, Vector][];
  return known.reduce((sum, [fact, value]) => sum + norm(subtract(readOut(state, FACTS[fact].key), value)), 0) / known.length;
}

/** Gated delta rule: S̃ = αS, e = v − S̃ᵀk, S = S̃ + β k eᵀ. Linear attention for comparison: S = S + k vᵀ. */
export function simulate(alpha: number, beta: number): Step[] {
  let state = zeros(CHANNELS.length, ANSWERS.length);
  let linear = zeros(CHANNELS.length, ANSWERS.length);
  return TOKENS.map((token, t) => {
    const prev = state;
    const decayed = scaleMatrix(prev, alpha);
    const prediction = readOut(decayed, token.k);
    const error = subtract(token.v, prediction);
    const write = outer(token.k, error.map(value => value * beta));
    state = addMatrix(decayed, write);
    linear = addMatrix(linear, outer(token.k, token.v));
    return { token, prev, decayed, prediction, error, write, state, linear, gdnError: retrievalError(state, t), linearError: retrievalError(linear, t) };
  });
}

export const read = (state: Matrix, fact: Fact) => readOut(state, FACTS[fact].key);

// ---- Capacity experiment: random unit keys/values in D dimensions, stored one after another ----
export const D = 32;
function rng(seed: number) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function unit(r: () => number, d = D) { const v = Array.from({ length: d }, () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r())); const n = Math.hypot(...v); return v.map(x => x / n); }
export type Rule = 'linear' | 'delta' | 'gated';
/** Stream `count` (k, v) pairs into a D×D state; return the relative read-back error of every pair at the end. */
export function capacity(rule: Rule, count: number, alpha = 0.97, seed = 1) {
  const r = rng(seed), keys = Array.from({ length: count }, () => unit(r)), values = Array.from({ length: count }, () => unit(r));
  const S = Array.from({ length: D }, () => new Array(D).fill(0));
  const readS = (k: number[]) => Array.from({ length: D }, (_, j) => k.reduce((s, ki, i) => s + ki * S[i][j], 0));
  for (let t = 0; t < count; t++) {
    const k = keys[t], v = values[t];
    if (rule === 'gated') for (const row of S) for (let j = 0; j < D; j++) row[j] *= alpha;
    const e = rule === 'linear' ? v : (p => v.map((x, j) => x - p[j]))(readS(k));
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) S[i][j] += k[i] * e[j];
  }
  const errors = keys.map((k, t) => { const p = readS(k); return Math.hypot(...p.map((x, j) => x - values[t][j])); });
  return { errors, state: S };
}
export const COUNTS = [4, 8, 16, 24, 32, 48, 64, 96, 128];
const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
/** Mean read-back error over all stored pairs and over the 8 most recent ones. */
export const CAPACITY = (['linear', 'delta', 'gated'] as Rule[]).map(rule => ({
  rule,
  all: COUNTS.map(n => mean(capacity(rule, n).errors)),
  recent: COUNTS.map(n => mean(capacity(rule, n).errors.slice(-8))),
}));

// ---- Forget gate: a topic switch; how fast is the old topic overwritten and the new one learned? ----
/** alpha: a constant decay, or 'switch' = 1 everywhere except a near-reset (0.05) at the topic change (data-dependent gate). */
export function topicSwitch(alpha: number | 'switch', steps = 80, seed = 4) {
  const r = rng(seed), A = Array.from({ length: 6 }, () => [unit(r), unit(r)]), B = Array.from({ length: 6 }, () => [unit(r), unit(r)]);
  const S = Array.from({ length: D }, () => new Array(D).fill(0));
  const readS = (k: number[]) => Array.from({ length: D }, (_, j) => k.reduce((s, ki, i) => s + ki * S[i][j], 0));
  const err = (set: number[][][]) => mean(set.map(([k, v]) => Math.hypot(...readS(k).map((x, j) => x - v[j]))));
  const out: { t: number; oldErr: number; newErr: number; norm: number }[] = [];
  for (let t = 0; t < steps; t++) {
    const set = t < steps / 2 ? A : B, [k, v] = set[t % set.length];
    const a = alpha === 'switch' ? (t === steps / 2 ? 0.05 : 1) : alpha;
    for (const row of S) for (let j = 0; j < D; j++) row[j] *= a;
    const p = readS(k), beta = 0.35;
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) S[i][j] += beta * k[i] * (v[j] - p[j]);
    out.push({ t, oldErr: err(A), newErr: err(B), norm: Math.sqrt(S.flat().reduce((s, x) => s + x * x, 0)) });
  }
  return out;
}

// ---- Memory per sequence: the Qwen3.8-Flash-Next shapes are shared with its lab ----
export { CONFIG as QWEN, memory as qwenMemory, gdnStateBytes } from '../qwen3-8-flash-next-explainer/model';
