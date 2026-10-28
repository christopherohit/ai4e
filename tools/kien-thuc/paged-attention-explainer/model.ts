// KV-cache memory simulator for the PagedAttention lab. Requests take their prompt/output lengths from a real sample
// of databricks-dolly-15k (counted with the Qwen2.5 tokenizer, scripts/llm-bake/inference.py); the GPU memory and
// arrival rate are illustrative. Three allocators share the same request stream and the same decode loop:
//   reserve  – contiguous region for the maximum sequence length (2 048 tokens), as in pre-vLLM servers;
//   exact    – contiguous region of exactly prompt + output tokens (an oracle that knows the output length);
//   paged    – fixed 16-token blocks allocated on demand anywhere in memory (PagedAttention / vLLM).
import data from './data/inference.json';

export const INFER = data as unknown as {
  model: string; device: string; kvBytesPerTokenFp32: number; layers: number; kvHeads: number; hidden: number; heads: number;
  reference: { name: string; layers: number; kvHeads: number; headDim: number; kvBytesPerTokenFp16: number }[];
  workload: { source: string; count: number; prompt: Record<string, number>; output: Record<string, number>; promptMean: number; outputMean: number; sample: [number, number][] };
};

export const BLOCK = 16;
export const COLS = 48;
export const ROWS = 20;
export const BLOCKS = COLS * ROWS; // 960 blocks = 15 360 tokens
export const MAX_LEN = 2048;
export const STEPS = 480;
export const ARRIVALS_PER_STEP = 1.2;
export type Strategy = 'reserve' | 'exact' | 'paged';
export const STRATEGIES: { key: Strategy; label: string }[] = [
  { key: 'reserve', label: 'Đặt trước 2 048 token liền mạch' },
  { key: 'exact', label: 'Đặt trước đúng độ dài (biết trước)' },
  { key: 'paged', label: 'PagedAttention · khối 16 token' },
];

interface Req { id: number; prompt: number; output: number; generated: number; blocks: number[]; start: number; arrival: number; done: number }
export interface Step { running: number; queued: number; stored: number; allocated: number; finished: number; generated: number; preempted: number }
export interface Run { steps: Step[]; owner: Int16Array[]; fill: Uint8Array[]; reqs: Req[]; tableAt: (step: number) => { id: number; blocks: number[]; tokens: number } | null }

const blocksFor = (tokens: number) => Math.ceil(tokens / BLOCK);
const len = (r: Req) => Math.min(r.prompt + r.generated, MAX_LEN);

export function requests(): Req[] {
  // Deterministic arrivals: request k arrives at step floor(k / rate). Clip very long ones to the context limit.
  return INFER.workload.sample.map(([p, o], k) => ({ id: k, prompt: Math.max(1, Math.min(p, MAX_LEN - 64)), output: Math.max(1, Math.min(o, MAX_LEN - Math.min(p, MAX_LEN - 64))), generated: 0, blocks: [], start: -1, arrival: Math.floor(k / ARRIVALS_PER_STEP), done: -1 }));
}

function firstFit(free: Uint8Array, n: number) {
  let run = 0;
  for (let i = 0; i < free.length; i++) { run = free[i] ? run + 1 : 0; if (run === n) return i - n + 1; }
  return -1;
}

export function simulate(strategy: Strategy): Run {
  const reqs = requests(), free = new Uint8Array(BLOCKS).fill(1);
  const queue: Req[] = [], running: Req[] = [];
  const steps: Step[] = [], owner: Int16Array[] = [], fill: Uint8Array[] = [];
  let next = 0, finished = 0, generated = 0, preempted = 0;
  const release = (r: Req) => { for (const b of r.blocks) free[b] = 1; r.blocks = []; };
  const admit = (r: Req) => {
    if (strategy === 'paged') {
      const need = blocksFor(r.prompt + 1);
      const avail = free.reduce((s, v) => s + v, 0);
      if (avail - need < running.length * 0.05) return false; // keep a little headroom for growth
      for (let i = 0, got = 0; i < BLOCKS && got < need; i++) if (free[i]) { free[i] = 0; r.blocks.push(i); got++; }
      return true;
    }
    const need = blocksFor(strategy === 'reserve' ? MAX_LEN : r.prompt + r.output);
    const at = firstFit(free, need);
    if (at < 0) return false;
    for (let i = at; i < at + need; i++) { free[i] = 0; r.blocks.push(i); }
    return true;
  };
  for (let t = 0; t < STEPS; t++) {
    while (next < reqs.length && reqs[next].arrival <= t) queue.push(reqs[next++]);
    while (queue.length && admit(queue[0])) { const r = queue.shift()!; r.start = t; running.push(r); }
    // One decode iteration: every running request produces one token.
    for (const r of [...running]) {
      r.generated++;
      generated++;
      if (strategy === 'paged' && blocksFor(len(r)) > r.blocks.length) {
        const b = free.indexOf(1);
        if (b < 0) { // out of blocks: preempt the most recent request (vLLM recomputes it later)
          const victim = running.pop()!;
          if (victim === r) { r.generated--; generated--; }
          release(victim); victim.generated = 0; victim.start = -1; queue.unshift(victim); preempted++;
          if (victim === r) continue;
          const again = free.indexOf(1);
          if (again >= 0) { free[again] = 0; r.blocks.push(again); }
        } else { free[b] = 0; r.blocks.push(b); }
      }
      if (r.generated >= r.output) { r.done = t; release(r); running.splice(running.indexOf(r), 1); finished++; }
    }
    const own = new Int16Array(BLOCKS).fill(-1), fl = new Uint8Array(BLOCKS);
    let stored = 0;
    for (const r of running) {
      const tokens = len(r);
      stored += tokens;
      r.blocks.forEach((b, k) => { own[b] = r.id; fl[b] = Math.max(0, Math.min(BLOCK, tokens - k * BLOCK)); });
    }
    steps.push({ running: running.length, queued: queue.length, stored, allocated: BLOCKS - free.reduce((s, v) => s + v, 0), finished, generated, preempted });
    owner.push(own); fill.push(fl);
  }
  const tableAt = (step: number) => {
    const own = owner[step], fl = fill[step];
    const ids = [...new Set(Array.from(own).filter(v => v >= 0))];
    if (!ids.length) return null;
    const id = ids.reduce((a, b) => (reqs[a].start < reqs[b].start || (reqs[a].start === reqs[b].start && a < b) ? a : b));
    const blocks: number[] = [];
    own.forEach((v, i) => { if (v === id) blocks.push(i); });
    if (strategy === 'paged') blocks.sort((a, b) => a - b);
    return { id, blocks, tokens: blocks.reduce((s, b) => s + fl[b], 0) };
  };
  return { steps, owner, fill, reqs, tableAt };
}

/** Prefix sharing for parallel sampling: one prompt, n samples, each generating `out` tokens. */
export function sharing(prompt: number, n: number, out: number) {
  const full = Math.floor(prompt / BLOCK), partial = prompt % BLOCK > 0 ? 1 : 0;
  const separate = n * blocksFor(prompt + out);
  // Shared full prompt blocks; the partial last prompt block is copied on write by each sample.
  const perSample = blocksFor(prompt + out) - full;
  const shared = full + n * perSample;
  return { full, partial, separate, shared, perSample, saving: 1 - shared / separate };
}
