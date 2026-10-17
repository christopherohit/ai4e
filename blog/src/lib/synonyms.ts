import { normalizeVi } from './normalize';

/**
 * EN ↔ VI synonym groups. Every phrase in a group is considered equivalent.
 * Phrases are written naturally; matching happens on normalized (diacritic-free) text.
 */
export const SYNONYM_GROUPS: string[][] = [
  ['attention', 'chú ý', 'cơ chế chú ý', 'self-attention'],
  ['convolution', 'tích chập', 'convolutional'],
  ['cnn', 'mạng tích chập', 'mạng nơ-ron tích chập', 'convolutional neural network'],
  ['neural network', 'mạng nơ-ron', 'mạng neuron', 'mạng thần kinh'],
  ['bias', 'thiên lệch', 'thiên kiến'],
  ['fairness', 'công bằng', 'tính công bằng'],
  ['energy', 'năng lượng'],
  ['carbon', 'khí thải carbon', 'phát thải'],
  ['diffusion', 'khuếch tán', 'mô hình khuếch tán'],
  ['generative', 'sinh', 'tạo sinh', 'mô hình sinh'],
  ['transformer', 'transformers'],
  ['gan', 'mạng đối nghịch', 'mạng đối nghịch tạo sinh', 'generative adversarial network'],
  ['visualization', 'trực quan hóa', 'trực quan'],
  ['deep learning', 'học sâu'],
  ['machine learning', 'học máy'],
  ['image', 'hình ảnh', 'ảnh'],
  ['text', 'văn bản'],
  ['language model', 'mô hình ngôn ngữ', 'llm'],
  ['interpretability', 'khả năng diễn giải', 'diễn giải'],
  ['classification', 'phân loại'],
  ['training', 'huấn luyện'],
  ['activation', 'kích hoạt', 'hàm kích hoạt'],
  ['pooling', 'gộp'],
  ['embedding', 'nhúng', 'vector nhúng'],
  ['token', 'tokens'],
  ['noise', 'nhiễu'],
  ['prompt', 'câu lệnh', 'lời nhắc'],
  ['computer vision', 'thị giác máy tính'],
  ['nlp', 'xử lý ngôn ngữ tự nhiên', 'natural language processing'],
];

/** Canonical phrase key: normalized, punctuation/hyphens → single spaces. */
export function phraseKey(text: string): string {
  return normalizeVi(text)
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim();
}

/** Normalized phrase → set of normalized equivalent phrases (excluding itself). */
const SYNONYM_INDEX: Map<string, string[]> = (() => {
  const index = new Map<string, Set<string>>();
  for (const group of SYNONYM_GROUPS) {
    const norm = group.map(phraseKey);
    for (const phrase of norm) {
      const set = index.get(phrase) ?? new Set<string>();
      for (const other of norm) if (other !== phrase) set.add(other);
      index.set(phrase, set);
    }
  }
  return new Map([...index].map(([k, v]) => [k, [...v]]));
})();

/** Longest phrase length (in words) present in the synonym index. */
const MAX_PHRASE_WORDS = Math.max(...[...SYNONYM_INDEX.keys()].map((k) => k.split(' ').length));

/** Return the normalized synonyms of a phrase (empty array if none). */
export function getSynonyms(phrase: string): string[] {
  return SYNONYM_INDEX.get(phraseKey(phrase)) ?? [];
}

/**
 * Expand a query with synonyms. Returns the canonical original query (see
 * `phraseKey`) plus canonical synonym phrases for every (greedy, longest-first) phrase match
 * found in the query. Duplicates are removed; the original query comes first.
 *
 *   expandQuery('Tích chập')   → ['tich chap', 'convolution', 'convolutional']
 *   expandQuery('attention gan') → ['attention gan', 'chu y', ..., 'mang doi nghich', ...]
 */
export function expandQuery(query: string): string[] {
  const normalized = phraseKey(query);
  if (!normalized) return [];
  const words = normalized.split(' ');
  const result: string[] = [normalized];
  const seen = new Set(result);

  let i = 0;
  while (i < words.length) {
    let matched = 0;
    for (let len = Math.min(MAX_PHRASE_WORDS, words.length - i); len >= 1; len--) {
      const phrase = words.slice(i, i + len).join(' ');
      const syns = SYNONYM_INDEX.get(phrase);
      if (syns) {
        for (const s of syns) {
          if (!seen.has(s)) {
            seen.add(s);
            result.push(s);
          }
        }
        matched = len;
        break;
      }
    }
    i += matched || 1;
  }
  return result;
}
