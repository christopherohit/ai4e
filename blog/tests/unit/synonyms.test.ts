import { describe, expect, it } from 'vitest';
import { expandQuery, getSynonyms, phraseKey } from '../../src/lib/synonyms';

describe('phraseKey', () => {
  it('normalizes diacritics and punctuation', () => {
    expect(phraseKey('  Mạng  nơ-ron ')).toBe('mang no ron');
  });
});

describe('getSynonyms', () => {
  it('maps English to Vietnamese', () => {
    expect(getSynonyms('attention')).toContain('chu y');
    expect(getSynonyms('convolution')).toContain('tich chap');
    expect(getSynonyms('fairness')).toContain('cong bang');
  });

  it('maps Vietnamese (with or without diacritics) to English', () => {
    expect(getSynonyms('tích chập')).toContain('convolution');
    expect(getSynonyms('tich chap')).toContain('convolution');
    expect(getSynonyms('Mạng nơ-ron')).toContain('neural network');
    expect(getSynonyms('khuếch tán')).toContain('diffusion');
  });

  it('returns empty for unknown phrases', () => {
    expect(getSynonyms('xyz không tồn tại')).toEqual([]);
  });
});

describe('expandQuery', () => {
  it('returns empty array for blank queries', () => {
    expect(expandQuery('   ')).toEqual([]);
  });

  it('puts the normalized original query first', () => {
    const out = expandQuery('Tích chập');
    expect(out[0]).toBe('tich chap');
    expect(out).toContain('convolution');
  });

  it('expands multi-word phrases inside a longer query', () => {
    const out = expandQuery('giải thích mạng nơ-ron');
    expect(out[0]).toBe('giai thich mang no ron');
    expect(out).toContain('neural network');
  });

  it('prefers the longest phrase match', () => {
    const out = expandQuery('mạng nơ-ron tích chập');
    expect(out).toContain('cnn');
  });

  it('expands several phrases in one query', () => {
    const out = expandQuery('attention gan');
    expect(out).toContain('chu y');
    expect(out).toContain('mang doi nghich');
  });

  it('expands bias and energy', () => {
    expect(expandQuery('bias')).toContain('thien lech');
    expect(expandQuery('energy')).toContain('nang luong');
  });

  it('returns only the query when nothing matches', () => {
    expect(expandQuery('hello world')).toEqual(['hello world']);
  });

  it('has no duplicates', () => {
    const out = expandQuery('attention chú ý');
    expect(new Set(out).size).toBe(out.length);
  });
});
