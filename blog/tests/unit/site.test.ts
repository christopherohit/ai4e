import { describe, expect, it } from 'vitest';
import { CATEGORIES, CATEGORY_LIST, NAV_ITEMS, getCategory, isActivePath, isCategorySlug } from '../../src/lib/site';

describe('categories', () => {
  it('defines kien-thuc and du-an with Vietnamese labels', () => {
    expect(CATEGORY_LIST.map((c) => c.slug)).toEqual(['kien-thuc', 'du-an']);
    expect(CATEGORIES['kien-thuc'].label).toBe('Kiến thức');
    expect(CATEGORIES['du-an'].label).toBe('Dự án');
    for (const c of CATEGORY_LIST) {
      expect(c.href).toBe(`/${c.slug}/`);
      expect(c.description.length).toBeGreaterThan(20);
      expect(c.icon).toMatch(/<\w+/);
    }
  });

  it('validates category slugs', () => {
    expect(isCategorySlug('kien-thuc')).toBe(true);
    expect(isCategorySlug('toString')).toBe(false);
    expect(getCategory('du-an')?.label).toBe('Dự án');
    expect(getCategory('khac')).toBeUndefined();
  });
});

describe('navigation', () => {
  it('has the five main items with trailing slashes', () => {
    expect(NAV_ITEMS.map((n) => n.href)).toEqual(['/', '/kien-thuc/', '/du-an/', '/chu-de/', '/gioi-thieu/']);
  });

  it('computes active state', () => {
    expect(isActivePath('/', '/')).toBe(true);
    expect(isActivePath('/kien-thuc/', '/')).toBe(false);
    expect(isActivePath('/kien-thuc/cnn-explainer/', '/kien-thuc/')).toBe(true);
    expect(isActivePath('/kien-thuc', '/kien-thuc/')).toBe(true);
    expect(isActivePath('/du-an/', '/kien-thuc/')).toBe(false);
  });
});
