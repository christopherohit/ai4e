// Measurements for the Jev-style decoding lab, baked by scripts/llm-bake/jev.py with Qwen2.5-1.5B-Instruct on CPU.
import data from './data/jev.json';

export interface Candidate { value: string; tokens: string[]; logprob: number; p: number }
export interface Field { candidates: Candidate[]; suffix: string; suffixTokens: string[]; top: { token: string; id: number; p: number; logit: number }[]; firstTokenLogits: { value: string; token: string; id: number; logit: number }[]; vocabLogitStats: { max: number; mean: number; std: number }; hidden: number[]; answer: string }
export interface Doc {
  title: string; text: string;
  autoregressive: { tokens: string[]; text: string; forwardPasses: number; seconds: number; ttft: number; perToken: number; parsed: Record<string, unknown> | null; valid: boolean };
  parallel: { promptTokens: number; prefillSeconds: number; scoreSeconds: number; rows: number; seconds: number; forwardPasses: number; fields: Record<string, Field> };
}
export const JEV = data as unknown as { model: string; device: string; layers: number; hidden: number; vocab: number; kvHeads: number; heads: number; headDim: number; schema: Record<string, string[]>; schemaText: string; rules: string; docs: Doc[] };
export const FIELDS = Object.keys(JEV.schema);
export const FIELD_LABEL: Record<string, string> = { risk_level: 'risk_level · 4 lựa chọn', requires_review: 'requires_review · boolean', action_tier: 'action_tier · 3 lựa chọn' };

/** KV cache bytes for n prompt tokens: K and V × layers × KV heads × head dim × bytes per value. */
export const kvBytes = (tokens: number, bytesPerValue = 2) => 2 * JEV.layers * JEV.kvHeads * JEV.headDim * tokens * bytesPerValue;
export const parallelJson = (d: Doc) => `{"risk_level": "${d.parallel.fields.risk_level.answer}", "requires_review": ${d.parallel.fields.requires_review.answer}, "action_tier": "${d.parallel.fields.action_tier.answer}"}`;
/** Why the autoregressive output failed to parse (the model copied the bare enum style of the schema). */
export function invalidReason(d: Doc) {
  if (d.autoregressive.valid) return '';
  const m = d.autoregressive.text.match(/"risk_level":\s*([A-Z_]+)/);
  return m ? `"risk_level": ${m[1]} — giá trị chuỗi thiếu dấu ngoặc kép` : 'không parse được JSON';
}
