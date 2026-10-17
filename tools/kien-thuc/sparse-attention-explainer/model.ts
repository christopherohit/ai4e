// Toy causal attention over 32 code tokens for the sparse-attention lab. Keys are topic vectors plus fixed noise; each
// query looks for one topic. QSA follows the report's recipe at toy scale: average-pool keys into micro-blocks of r=4,
// score complete blocks with 4 ReLU indexer heads, keep top-k blocks plus the incomplete tail, then softmax over them.
import { dot, mean, seeded, softmax, type Vector } from '../../_shared/sequence-lab/math';

export const BLOCK = 4;
const TOPICS = ['tên hàm area', 'hằng PI', 'tham số r', 'cú pháp', 'chú thích', 'xuống dòng', 'số', 'biến x'] as const;
type Topic = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7;
export interface Token { text: string; topic: Topic; seeks: Topic }
const t = (text: string, topic: Topic, seeks: Topic = topic): Token => ({ text, topic, seeks });

// def area(r): return PI*r*r / # tính diện tích hình tròn / PI = 3.14 / x = area(2)
export const TOKENS: Token[] = [
  t('def', 3, 0), t('area', 0), t('(', 3), t('r', 2), t(')', 3), t(':', 3), t('return', 3, 2), t('PI', 1),
  t('*', 3, 2), t('r', 2), t('*', 3, 2), t('r', 2), t('⏎', 5), t('#', 4), t('tính', 4), t('diện', 4),
  t('tích', 4), t('hình', 4), t('tròn', 4), t('⏎', 5), t('PI', 1), t('=', 3, 1), t('3.14', 6, 1), t('⏎', 5),
  t('x', 7), t('=', 3, 7), t('area', 0), t('(', 3, 0), t('2', 6, 2), t(')', 3, 0), t('⏎', 5), t('print', 3, 7),
];
export const LENGTH = TOKENS.length;
export const BLOCKS = LENGTH / BLOCK;
export const topicName = (topic: Topic) => TOPICS[topic];

const DIM = TOPICS.length;
const random = seeded(7);
const noise = () => (random() - 0.5) * 0.35;
const basis = (topic: number) => Array.from({ length: DIM }, (_, index) => (index === topic ? 1 : 0));
const keys: Vector[] = TOKENS.map(token => basis(token.topic).map(value => value + noise()));
// Every query also leans slightly towards its own topic, as real queries carry local syntax.
const queries: Vector[] = TOKENS.map(token => basis(token.seeks).map((value, index) => 1.6 * value + (index === token.topic ? 0.35 : 0) + noise() * 0.6));
const headNoise: Vector[] = Array.from({ length: 4 }, () => Array.from({ length: DIM }, () => noise() * 0.7));
const TEMPERATURE = 7;

export const blockOf = (position: number) => Math.floor(position / BLOCK);
/** Complete blocks a query at i may score: the block's last token p_b + r − 1 must be ≤ i. */
export const completeBlocks = (i: number) => Math.floor((i + 1) / BLOCK);
export const tailStart = (i: number) => completeBlocks(i) * BLOCK;

/** Mean-pooled keys per block (RoPE omitted; the report pools before applying it). */
export const blockKeys: Vector[] = Array.from({ length: BLOCKS }, (_, block) => mean(keys.slice(block * BLOCK, block * BLOCK + BLOCK)));

export function indexerScores(i: number): (number | null)[] {
  return blockKeys.map((key, block) => (block < completeBlocks(i) ? headNoise.reduce((sum, head) => sum + Math.max(0, dot(queries[i].map((value, index) => value + head[index]), key)), 0) : null));
}

export function topBlocks(i: number, budget: number) {
  return indexerScores(i).map((score, block) => ({ score, block })).filter(item => item.score !== null)
    .sort((a, b) => b.score! - a.score! || b.block - a.block).slice(0, budget).map(item => item.block).sort((a, b) => a - b);
}

export type Pattern = 'dense' | 'local' | 'global' | 'qsa';
/** Key positions a query reads under each pattern; every sparse pattern reads budget·r complete-block tokens + tail. */
export function allowed(pattern: Pattern, i: number, budget: number): number[] {
  const causal = Array.from({ length: i + 1 }, (_, j) => j);
  if (pattern === 'dense') return causal;
  const size = budget * BLOCK;
  const tail = causal.filter(j => j >= tailStart(i));
  if (pattern === 'local') return causal.filter(j => j > i - size - tail.length);
  if (pattern === 'global') return [...new Set([...causal.filter(j => j < BLOCK), ...causal.filter(j => j > i - (size - BLOCK) - tail.length)])].filter(j => j <= i).sort((a, b) => a - b);
  const blocks = topBlocks(i, budget);
  return causal.filter(j => blocks.includes(blockOf(j)) || j >= tailStart(i));
}

/** Softmax over the allowed keys only; masked keys get exactly 0. */
export function weights(i: number, keep: number[]) {
  const scores = TOKENS.map((_, j) => (keep.includes(j) ? TEMPERATURE * dot(queries[i], keys[j]) / Math.sqrt(DIM) : -Infinity));
  return softmax(scores);
}

/** Share of the dense attention mass that falls on the keys a sparse pattern keeps. */
export function coverage(pattern: Pattern, i: number, budget: number) {
  const dense = weights(i, allowed('dense', i, budget));
  return allowed(pattern, i, budget).reduce((sum, j) => sum + dense[j], 0);
}

export const strongest = (row: number[]) => row.indexOf(Math.max(...row));
