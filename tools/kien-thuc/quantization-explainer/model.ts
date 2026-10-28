// Quantization lab data: round-to-nearest weight quantization of Qwen2.5-0.5B measured in scripts/llm-bake/quantize.py,
// plus number formats and a bandwidth model for decode speed.
import data from './data/quantization.json';

export interface Result { name: string; bits: number | null; granularity: string | number | null; bitsPerWeight: number; perplexity: number; weightRmse: number; continuation: string }
export const Q = data as unknown as {
  model: string; quantizedParams: number; totalParams: number; evalTokens: number; evalSource: string; prompt: string; results: Result[];
  layerStats: { name: string; absMax: number; std: number; kurtosis: number }[];
  sampleName: string; sampleRowLength: number; sampleRowAbsMax: number; sampleRow: number[]; sampleQuantized: Record<string, number[]>;
};
export const BASE = Q.results[0];

/** Symmetric RTN quantization of a vector with one scale (the lab redraws the grid for any bit width). */
export function quantizeVector(values: number[], bits: number, group = values.length) {
  const qmax = 2 ** (bits - 1) - 1, out: number[] = [], scales: number[] = [];
  for (let i = 0; i < values.length; i += group) {
    const chunk = values.slice(i, i + group), s = Math.max(...chunk.map(Math.abs)) / qmax || 1;
    scales.push(s);
    for (const v of chunk) out.push(Math.max(-qmax - 1, Math.min(qmax, Math.round(v / s))) * s);
  }
  return { out, scales };
}
export const rmse = (a: number[], b: number[]) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length);

export const FORMATS = [
  { name: 'FP32', sign: 1, exponent: 8, mantissa: 23 },
  { name: 'BF16', sign: 1, exponent: 8, mantissa: 7 },
  { name: 'FP16', sign: 1, exponent: 5, mantissa: 10 },
  { name: 'FP8 E4M3', sign: 1, exponent: 4, mantissa: 3 },
  { name: 'INT8', sign: 0, exponent: 0, mantissa: 8 },
  { name: 'INT4', sign: 0, exponent: 0, mantissa: 4 },
];
export const SIZES = [{ name: '7B', params: 7e9 }, { name: '70B', params: 70e9 }];
export const PRECISIONS = [{ name: 'FP16', bits: 16 }, { name: 'INT8', bits: 8 }, { name: 'INT4 g128', bits: 4.125 }];
/** Upper bound for batch-1 decode speed: every token reads all weights once. H100 SXM HBM3 bandwidth. */
export const H100_BW = 3.35e12;
export const decodeBound = (params: number, bitsPerWeight: number, bandwidth = H100_BW) => bandwidth / (params * bitsPerWeight / 8);
