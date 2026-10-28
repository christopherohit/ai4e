// Inference engineering lab: CPU measurements of Qwen2.5-0.5B (scripts/llm-bake/inference.py), a roofline model with
// NVIDIA datasheet peaks, and a static vs continuous batching simulator on real Dolly-15k output lengths.
import data from './data/inference.json';

export const M = data as unknown as {
  model: string; device: string; params: number; layers: number; hidden: number; heads: number; kvHeads: number;
  prefill: { tokens: number; seconds: number }[]; decodeWithCache: number[]; decodeWithoutCache: number[];
  batch: { batch: number; stepSeconds: number; tokensPerSecond: number }[]; kvBytesPerTokenFp32: number;
  workload: { sample: [number, number][]; prompt: Record<string, number>; output: Record<string, number> };
};

export const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
export const TTFT_128 = M.prefill.find(p => p.tokens === 128)!.seconds;
export const TPOT = mean(M.decodeWithCache);
export const PREFILL_RATE = M.prefill.at(-1)!.tokens / M.prefill.at(-1)!.seconds;

// Roofline: attainable FLOP/s = min(peak compute, bandwidth × arithmetic intensity). Dense BF16 tensor-core peaks.
export const GPUS = [
  { name: 'A100 SXM 80GB', tflops: 312, tbps: 2.039 },
  { name: 'H100 SXM', tflops: 989, tbps: 3.35 },
];
export const ridge = (g: typeof GPUS[number]) => g.tflops * 1e12 / (g.tbps * 1e12);
export const attainable = (g: typeof GPUS[number], intensity: number) => Math.min(g.tflops * 1e12, g.tbps * 1e12 * intensity);
/**
 * Decode step of a dense model with P parameters in 16-bit weights, batch B, context n per sequence and KV bytes per
 * token k: FLOPs ≈ 2·P·B (matrix products) and bytes ≈ 2·P (weights read once) + B·n·k (each sequence reads its cache).
 */
export function decodeIntensity(P: number, B: number, n = 0, k = 0) { return (2 * P * B) / (2 * P + B * n * k); }
export function prefillIntensity(P: number, tokens: number) { return (2 * P * tokens) / (2 * P); }

// Batching simulator: each request needs `output` decode steps; a batch has `slots` places.
export interface Lane { id: number; start: number; end: number; slot: number }
export function staticBatching(outputs: number[], slots: number) {
  const lanes: Lane[] = []; let t = 0;
  for (let i = 0; i < outputs.length; i += slots) {
    const group = outputs.slice(i, i + slots), longest = Math.max(...group);
    group.forEach((o, k) => lanes.push({ id: i + k, start: t, end: t + o, slot: k }));
    t += longest;
  }
  return { lanes, makespan: t };
}
export function continuousBatching(outputs: number[], slots: number) {
  const free = Array.from({ length: slots }, () => 0), lanes: Lane[] = [];
  outputs.forEach((o, i) => { const slot = free.indexOf(Math.min(...free)); lanes.push({ id: i, start: free[slot], end: free[slot] + o, slot }); free[slot] += o; });
  return { lanes, makespan: Math.max(...free) };
}
export const OUTPUTS = M.workload.sample.slice(0, 48).map(([, o]) => Math.max(4, Math.min(o, 400)));
export const busy = (r: { lanes: Lane[]; makespan: number }, slots: number) => r.lanes.reduce((s, l) => s + l.end - l.start, 0) / (r.makespan * slots);

export const ROADMAP = [
  { title: 'Nền tảng suy luận', items: ['Pipeline & tokenization', 'Prefill / decode', 'KV cache', 'TTFT, ITL/TPOT', 'Thông lượng & độ trễ'] },
  { title: 'Khái niệm Transformer', items: ['Embedding', 'Self-attention Q, K, V', 'Khối Transformer'] },
  { title: 'GPU & phần cứng', items: ['SM, SRAM, HBM', 'Băng thông, FLOPS', 'Compute vs memory-bound'] },
  { title: 'Kỹ thuật tối ưu', items: ['PagedAttention', 'FlashAttention', 'Continuous batching', 'Chunked prefill', 'Prompt caching', 'Speculative decoding', 'Quantization'] },
  { title: 'Engine phục vụ', items: ['vLLM', 'SGLang', 'TensorRT-LLM', 'llama.cpp'] },
];
