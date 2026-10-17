import { describe, expect, it } from 'vitest';
import { collectTags, tagHref, tagSlug } from '../../src/lib/tags';

describe('tagSlug', () => {
  it('creates diacritics-free kebab-case slugs', () => {
    expect(tagSlug('Thị giác máy tính')).toBe('thi-giac-may-tinh');
    expect(tagSlug('Mạng đối nghịch')).toBe('mang-doi-nghich');
    expect(tagSlug('CNN')).toBe('cnn');
  });

  it('collapses punctuation and trims dashes', () => {
    expect(tagSlug('  Học sâu / Deep Learning!  ')).toBe('hoc-sau-deep-learning');
    expect(tagSlug('Nơ-ron')).toBe('no-ron');
  });
});

describe('tagHref', () => {
  it('builds the tag URL with trailing slash', () => {
    expect(tagHref('Thị giác máy tính')).toBe('/chu-de/thi-giac-may-tinh/');
  });
});

describe('collectTags', () => {
  it('counts tags across collection entries, sorted by count then name', () => {
    const posts = [
      { data: { tags: ['CNN', 'Thị giác máy tính'] } },
      { data: { tags: ['Transformer', 'Thị giác máy tính'] } },
      { data: { tags: ['Ảnh', 'thị giác máy tính'] } },
    ];
    expect(collectTags(posts)).toEqual([
      { name: 'Thị giác máy tính', slug: 'thi-giac-may-tinh', count: 3 },
      { name: 'Ảnh', slug: 'anh', count: 1 },
      { name: 'CNN', slug: 'cnn', count: 1 },
      { name: 'Transformer', slug: 'transformer', count: 1 },
    ]);
  });

  it('accepts plain objects and ignores duplicate tags within a post', () => {
    const tags = collectTags([{ tags: ['GAN', 'gan'] }, { tags: [] }, {}]);
    expect(tags).toEqual([{ name: 'GAN', slug: 'gan', count: 1 }]);
  });
});
