import { withBase } from '../lib/base';
import { createSearchEngine, searchDocs, type SearchDoc, type SearchEngine, type SearchHit, type SearchOptions } from '../lib/search';

export { highlight, snippet } from '../lib/search';
export type { SearchHit } from '../lib/search';

let enginePromise: Promise<SearchEngine> | undefined;

/** Lazily fetch /search-index.json once and build the MiniSearch engine. */
export function loadSearch(): Promise<SearchEngine> {
  enginePromise ??= fetch(withBase('/search-index.json'))
    .then((response) => {
      if (!response.ok) throw new Error(`search-index.json: HTTP ${response.status}`);
      return response.json() as Promise<SearchDoc[]>;
    })
    .then(createSearchEngine)
    .catch((error) => {
      enginePromise = undefined;
      throw error;
    });
  return enginePromise;
}

export async function search(query: string, options?: SearchOptions): Promise<SearchHit[]> {
  return searchDocs(await loadSearch(), query, options);
}
