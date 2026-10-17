import MiniSearch, { type Query, type SearchResult } from 'minisearch';
import { normalizeVi } from './normalize';
import { expandQuery } from './synonyms';
import type { CategorySlug } from './site';

/** One entry of /search-index.json. */
export interface SearchDoc {
  id: string;
  url: string;
  title: string;
  description: string;
  category: CategorySlug;
  tags: string[];
  headings: string[];
  body: string;
}

export interface SearchHit {
  doc: SearchDoc;
  score: number;
  /** Normalized terms that matched (index terms + query terms), for highlighting. */
  terms: string[];
}

export interface SearchEngine {
  index: MiniSearch<SearchDoc>;
  docs: Map<string, SearchDoc>;
}

const FIELDS = ['title', 'tags', 'headings', 'description', 'body'] as const;
export const SEARCH_BOOST = { title: 4, tags: 3, headings: 2, description: 2, body: 1 };

/** Word tokenizer shared by indexing and querying (letters/digits, Unicode aware). */
export function tokenize(text: string): string[] {
  return text.split(/[^\p{L}\p{N}]+/u).filter(Boolean);
}

export function processTerm(term: string): string | null {
  const normalized = normalizeVi(term);
  return normalized ? normalized : null;
}

export function createSearchEngine(docs: readonly SearchDoc[]): SearchEngine {
  const index = new MiniSearch<SearchDoc>({
    fields: [...FIELDS],
    storeFields: ['category'],
    tokenize,
    processTerm,
    extractField: (doc, field) => {
      const value = (doc as unknown as Record<string, unknown>)[field];
      return Array.isArray(value) ? value.join(' · ') : String(value ?? '');
    },
    searchOptions: { boost: SEARCH_BOOST, prefix: true, fuzzy: 0.2 },
  });
  index.addAll(docs as SearchDoc[]);
  return { index, docs: new Map(docs.map((doc) => [doc.id, doc])) };
}

export interface SearchOptions {
  category?: CategorySlug | '';
  limit?: number;
}

function buildQuery(phrases: string[], combineWith: 'AND' | 'OR'): Query {
  return {
    combineWith: 'OR',
    queries: phrases.map((phrase, index) => ({
      queries: [phrase],
      combineWith,
      boost: index === 0 ? SEARCH_BOOST : { title: 3, tags: 2, headings: 1.5, description: 1.5, body: 0.75 },
    })),
  };
}

/**
 * Search with Vietnamese normalization + EN↔VI synonym expansion.
 * Every phrase (original query + synonyms) must match all of its words; if that finds
 * nothing, falls back to matching any word.
 */
export function searchDocs(engine: SearchEngine, query: string, options: SearchOptions = {}): SearchHit[] {
  const phrases = expandQuery(query);
  if (phrases.length === 0) return [];
  const filter = options.category
    ? (result: SearchResult) => result.category === options.category
    : undefined;
  let results = engine.index.search(buildQuery(phrases, 'AND'), { filter });
  if (results.length === 0) results = engine.index.search(buildQuery(phrases, 'OR'), { filter });
  const queryTerms = phrases.flatMap((phrase) => tokenize(phrase));
  return results.slice(0, options.limit ?? results.length).map((result) => ({
    doc: engine.docs.get(String(result.id))!,
    score: result.score,
    terms: [...new Set([...result.terms, ...queryTerms].map(normalizeVi))],
  }));
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (char) => `&#${char.charCodeAt(0)};`);
}

/** Normalized copy of `text` whose indices map 1:1 to the original string's UTF-16 indices. */
function normalizedParallel(text: string): string {
  let out = '';
  for (let i = 0; i < text.length; i++) {
    const normalized = normalizeVi(text[i]!);
    out += normalized.length === 1 ? normalized : text[i]!.toLowerCase().slice(0, 1) || ' ';
  }
  return out;
}

/** Ranges [start, end) where a word starts with one of the terms. */
function matchRanges(text: string, terms: readonly string[]): Array<[number, number]> {
  const normalized = normalizedParallel(text);
  const wanted = [...new Set(terms.filter((term) => term.length > 0))].sort((a, b) => b.length - a.length);
  const ranges: Array<[number, number]> = [];
  const wordPattern = /[\p{L}\p{N}]+/gu;
  for (const match of normalized.matchAll(wordPattern)) {
    const word = match[0];
    const term = wanted.find((candidate) => word.startsWith(candidate) || (candidate.startsWith(word) && word.length >= 3));
    if (term) ranges.push([match.index!, match.index! + Math.min(word.length, Math.max(term.length, 1))]);
  }
  return ranges;
}

/** HTML-escaped text with matched terms wrapped in <mark>. */
export function highlight(text: string, terms: readonly string[]): string {
  const ranges = matchRanges(text, terms);
  let html = '';
  let cursor = 0;
  for (const [start, end] of ranges) {
    html += escapeHtml(text.slice(cursor, start)) + '<mark>' + escapeHtml(text.slice(start, end)) + '</mark>';
    cursor = end;
  }
  return html + escapeHtml(text.slice(cursor));
}

/** Short excerpt of `text` around the first match (plain text, not escaped). */
export function snippet(text: string, terms: readonly string[], length = 180): string {
  if (text.length <= length) return text;
  const first = matchRanges(text, terms)[0];
  if (!first) return text.slice(0, length).trimEnd() + '…';
  let start = Math.max(0, first[0] - Math.floor(length / 3));
  const space = text.lastIndexOf(' ', start);
  if (start > 0 && space > start - 20) start = space + 1;
  const end = Math.min(text.length, start + length);
  return (start > 0 ? '…' : '') + text.slice(start, end).trim() + (end < text.length ? '…' : '');
}

/** Plain text from an MDX body: drops frontmatter, imports/exports, JSX tags, code fences, math and markdown syntax. */
export function stripMdx(source: string): string {
  return source
    .replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, '')
    .replace(/^\s*(import|export)\s[^\n]*$/gm, '')
    .replace(/```[^\n]*\n([\s\S]*?)```/g, ' $1 ')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/\$[^$\n]+\$/g, ' ')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/\{\/\*[\s\S]*?\*\/\}/g, ' ')
    .replace(/<\/?[A-Za-z][^<>]*?\/?>/g, ' ')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s{0,3}>\s?/gm, '')
    .replace(/^\s*([-*+]|\d+\.)\s+/gm, '')
    .replace(/^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/gm, ' ')
    .replace(/[|*_~`]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
