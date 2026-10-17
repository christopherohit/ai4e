import type { CategorySlug } from './site';
import { readingTime } from './reading-time';

/** Minimal post shape used by helpers (a `posts` collection entry satisfies it). */
export interface PostLike {
  id: string;
  body?: string;
  data: {
    title: string;
    category: CategorySlug;
    tags: readonly string[];
    order: number;
    date: Date;
  };
}

/** Public URL of a post: /<category>/<id>/ */
export function postHref(post: Pick<PostLike, 'id'> & { data: { category: CategorySlug } }): string {
  return `/${post.data.category}/${post.id}/`;
}

/** Reading time in minutes, computed from the raw MDX body. */
export function postReadingTime(post: Pick<PostLike, 'body'>): number {
  return readingTime(post.body ?? '');
}

/** Sort by `order` asc, then newest date first, then title (Vietnamese collation). */
export function sortPosts<T extends PostLike>(posts: readonly T[]): T[] {
  return [...posts].sort(
    (a, b) =>
      a.data.order - b.data.order ||
      b.data.date.getTime() - a.data.date.getTime() ||
      a.data.title.localeCompare(b.data.title, 'vi'),
  );
}

/**
 * Posts related to `current`, ranked by number of shared tags (desc), then
 * same category first, then `order`. Posts sharing no tag are excluded.
 */
export function relatedPosts<T extends PostLike>(current: T, all: readonly T[], limit = 3): T[] {
  const own = new Set(current.data.tags.map((t) => t.trim().toLowerCase()));
  return all
    .filter((p) => p.id !== current.id)
    .map((p) => ({
      post: p,
      shared: p.data.tags.filter((t) => own.has(t.trim().toLowerCase())).length,
    }))
    .filter((x) => x.shared > 0)
    .sort(
      (a, b) =>
        b.shared - a.shared ||
        Number(b.post.data.category === current.data.category) -
          Number(a.post.data.category === current.data.category) ||
        a.post.data.order - b.post.data.order,
    )
    .slice(0, limit)
    .map((x) => x.post);
}

/** "3 tháng 10, 2026" style Vietnamese date. */
export function formatDateVi(date: Date): string {
  return new Intl.DateTimeFormat('vi-VN', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' }).format(date);
}
