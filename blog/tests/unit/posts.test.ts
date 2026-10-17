import { describe, expect, it } from 'vitest';
import { formatDateVi, postHref, postReadingTime, relatedPosts, sortPosts, type PostLike } from '../../src/lib/posts';

function post(id: string, over: Partial<PostLike['data']> = {}, body = ''): PostLike {
  return {
    id,
    body,
    data: { title: id, category: 'kien-thuc', tags: ['CNN'], order: 1, date: new Date('2026-10-03'), ...over },
  };
}

describe('postHref', () => {
  it('builds /<category>/<id>/', () => {
    expect(postHref(post('cnn-explainer'))).toBe('/kien-thuc/cnn-explainer/');
    expect(postHref(post('dodrio', { category: 'du-an' }))).toBe('/du-an/dodrio/');
  });
});

describe('postReadingTime', () => {
  it('is at least one minute', () => {
    expect(postReadingTime({ body: '' })).toBe(1);
    expect(postReadingTime({})).toBe(1);
  });
  it('counts ~200 words per minute', () => {
    expect(postReadingTime({ body: 'từ '.repeat(401) })).toBe(3);
  });
});

describe('sortPosts', () => {
  it('sorts by order ascending without mutating input', () => {
    const input = [post('b', { order: 2 }), post('a', { order: 1 }), post('c', { order: 3 })];
    expect(sortPosts(input).map((p) => p.id)).toEqual(['a', 'b', 'c']);
    expect(input.map((p) => p.id)).toEqual(['b', 'a', 'c']);
  });
  it('breaks ties by newest date', () => {
    const input = [post('old', { date: new Date('2025-01-01') }), post('new', { date: new Date('2026-01-01') })];
    expect(sortPosts(input).map((p) => p.id)).toEqual(['new', 'old']);
  });
});

describe('relatedPosts', () => {
  const current = post('cnn', { tags: ['CNN', 'Học sâu', 'Thị giác máy tính'] });
  const all = [
    current,
    post('gan', { tags: ['GAN', 'Học sâu'], order: 4 }),
    post('viscuit', { category: 'du-an', tags: ['CNN', 'Thị giác máy tính'], order: 3 }),
    post('energy', { category: 'du-an', tags: ['Năng lượng'] }),
    post('diffusion', { tags: ['Học sâu'], order: 3 }),
  ];
  it('ranks by shared tags, excludes self and unrelated posts', () => {
    expect(relatedPosts(current, all).map((p) => p.id)).toEqual(['viscuit', 'diffusion', 'gan']);
  });
  it('respects the limit', () => {
    expect(relatedPosts(current, all, 1)).toHaveLength(1);
  });
});

describe('formatDateVi', () => {
  it('formats in Vietnamese', () => {
    expect(formatDateVi(new Date('2026-10-03'))).toMatch(/2026/);
    expect(formatDateVi(new Date('2026-10-03'))).toMatch(/tháng 10/);
  });
});
