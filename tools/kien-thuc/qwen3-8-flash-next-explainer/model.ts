// Qwen3.8-Flash-Next lab data. Architecture numbers are from the model card, config.json and the technical report
// (as summarised in the article); routing scores and residual vectors are seeded toy values, not checkpoint outputs.
import { dot, rmsnormalize, seeded, sigmoid, softmax, type Vector } from '../../_shared/sequence-lab/math';

export const CONFIG = {
  layers: 48, hidden: 2560, branches: 4, grRank: 320, vocab: 248_320, nativeContext: 262_144,
  gdn: { qkHeads: 16, vHeads: 48, headDim: 128, conv: 4 },
  qsa: { indexerHeads: 4, indexerDim: 128, coreHeads: 24, kvHeads: 2, coreDim: 256, ropeDim: 64, budget: 2048, block: 4 },
  moe: { experts: 512, topK: 10, shared: 1, intermediate: 640 },
} as const;

export type Mixer = 'GDN' | 'QSA';
export const mixerOf = (layer: number): Mixer => (layer % 4 === 0 ? 'QSA' : 'GDN');

/** Parameters of one routed expert: gate, up and down projections (2560 ↔ 640). */
export const expertParams = 3 * CONFIG.hidden * CONFIG.moe.intermediate;

const BF16 = 2;
/** Core K/V cached per token in one QSA layer (GQA: 2 KV heads × 256). Indexer keys and conv state are ignored. */
export const kvBytesPerToken = 2 * CONFIG.qsa.kvHeads * CONFIG.qsa.coreDim * BF16;
/** Recurrent state of one GDN layer: one d_k × d_v matrix per value head, independent of sequence length. */
export const gdnStateBytes = CONFIG.gdn.vHeads * CONFIG.gdn.headDim * CONFIG.gdn.headDim * BF16;
const qsaLayers = CONFIG.layers / 4;
export const memory = (tokens: number) => ({
  hybrid: qsaLayers * kvBytesPerToken * tokens + (CONFIG.layers - qsaLayers) * gdnStateBytes,
  allAttention: CONFIG.layers * kvBytesPerToken * tokens,
  gdnPart: (CONFIG.layers - qsaLayers) * gdnStateBytes,
});
/** Tokens that core attention reads per query in a QSA layer: complete-block budget plus the < r tail. */
export const coreReads = (tokens: number) => Math.min(tokens, CONFIG.qsa.budget + ((tokens - 1) % CONFIG.qsa.block));

export const SAMPLE_TOKENS = ['def', 'attention', 'gradient', 'xin', 'chào', '3.14', '<image>'] as const;
const hash = (text: string) => [...text].reduce((value, char) => Math.imul(value ^ char.charCodeAt(0), 16777619) >>> 0, 2166136261);

/** Toy router: deterministic logits per (token, layer); top-10 renormalised with softmax as in the MoE equation. */
export function route(token: string, layer: number) {
  const random = seeded(hash(token) ^ Math.imul(layer, 2654435761));
  const logits = Array.from({ length: CONFIG.moe.experts }, () => random() * 2 + (random() > 0.985 ? 3 : 0));
  const top = logits.map((logit, expert) => ({ logit, expert })).sort((a, b) => b.logit - a.logit).slice(0, CONFIG.moe.topK);
  const weights = softmax(top.map(item => item.logit));
  return top.map((item, index) => ({ expert: item.expert, weight: weights[index] }));
}

/** Toy gated residual with 8 channels standing in for 2560. */
export const CH = 8;
const residualRandom = seeded(2026);
const vector = (spread: number) => Array.from({ length: CH }, () => (residualRandom() - 0.5) * spread);
export const BRANCHES: Vector[] = [vector(2.4), vector(1.2), vector(3), vector(0.8)];
const gateLogits: Vector[] = BRANCHES.map(() => vector(6));
const mixing: Vector[] = Array.from({ length: CH }, () => vector(1.4));

export function residual(readScale: number[], write: number[]) {
  const normalized = BRANCHES.map(rmsnormalize);
  const gates = gateLogits.map((logits, branch) => logits.map(logit => sigmoid(logit) * readScale[branch]));
  const x = Array.from({ length: CH }, (_, channel) => normalized.reduce((sum, branch, index) => sum + gates[index][channel] * branch[channel], 0) / CONFIG.branches);
  const y = mixing.map(row => Math.tanh(dot(row, x) * 1.5));
  const updated = BRANCHES.map((branch, index) => branch.map((value, channel) => value + write[index] * y[channel]));
  return { normalized, gates, x, y, updated };
}
