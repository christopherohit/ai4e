import { normalizeVi } from './normalize';

/** Diacritics-stripped kebab-case slug: "Thị giác máy tính" → "thi-giac-may-tinh". */
export function tagSlug(tag: string): string {
  return normalizeVi(tag)
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

export interface TagInfo {
  name: string;
  slug: string;
  count: number;
}

/** Minimal shape needed from a post: either a collection entry (`data.tags`) or a plain object. */
type TaggedPost = { data: { tags?: readonly string[] } } | { tags?: readonly string[] };

function tagsOf(post: TaggedPost): readonly string[] {
  if ('data' in post) return post.data.tags ?? [];
  return post.tags ?? [];
}

/**
 * Collect unique tags (deduplicated by slug) with post counts, sorted by
 * count desc, then name (Vietnamese collation) asc. The first-seen spelling
 * of a tag is used as its display name.
 */
export function collectTags(posts: readonly TaggedPost[]): TagInfo[] {
  const map = new Map<string, TagInfo>();
  for (const post of posts) {
    const seenInPost = new Set<string>();
    for (const name of tagsOf(post)) {
      const slug = tagSlug(name);
      if (!slug || seenInPost.has(slug)) continue;
      seenInPost.add(slug);
      const existing = map.get(slug);
      if (existing) existing.count += 1;
      else map.set(slug, { name: name.trim(), slug, count: 1 });
    }
  }
  return [...map.values()].sort(
    (a, b) => b.count - a.count || a.name.localeCompare(b.name, 'vi'),
  );
}

/** Tag page URL. */
export function tagHref(tag: string): string {
  return `/chu-de/${tagSlug(tag)}/`;
}
