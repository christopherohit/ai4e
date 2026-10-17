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
