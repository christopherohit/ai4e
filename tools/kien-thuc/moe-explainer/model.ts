// MoE lab: a small mixture-of-experts trained on the UCI digits (scripts/llm-bake/moe.py) and published sizes of
// real MoE language models.
import data from './data/moe.json';

export interface Sample { label: number; pixels: number[]; routerProbs: number[]; experts: number[]; gates: number[]; pred: number }
export interface Variant { alpha: number; accuracy: number; heat: number[][]; samples: Sample[]; history: { epoch: number; accuracy: number; usage: number[] }[]; totalParams: number; activeParams: number }
export const MOE = data as unknown as { source: string; experts: number; topK: number; hidden: number; balanced: Variant; unbalanced: Variant; dense: { accuracy: number; params: number } };

// Published configurations (billions of parameters). Active = parameters used for one token.
export const MODELS = [
  { name: 'Mixtral 8×7B', total: 46.7, active: 12.9, experts: 8, topK: 2 },
  { name: 'Qwen3-30B-A3B', total: 30.5, active: 3.3, experts: 128, topK: 8 },
  { name: 'gpt-oss-120b', total: 117, active: 5.1, experts: 128, topK: 4 },
  { name: 'Qwen3-235B-A22B', total: 235, active: 22, experts: 128, topK: 8 },
  { name: 'Llama 4 Maverick', total: 400, active: 17, experts: 128, topK: 1 },
  { name: 'DeepSeek-V3', total: 671, active: 37, experts: 256, topK: 8 },
];

/** Expected share of experts touched by a batch of B tokens when each picks k of E uniformly: 1 − (1 − k/E)^B. */
export const touched = (k: number, E: number, B: number) => 1 - (1 - k / E) ** B;
/** Switch-Transformer auxiliary loss for usage fractions f and mean router probabilities P. */
export const balanceLoss = (f: number[], P: number[]) => f.length * f.reduce((s, v, i) => s + v * P[i], 0);
