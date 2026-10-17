/** Site-wide metadata and category definitions. */

export const SITE = {
  name: 'AI4E',
  tagline: 'Học AI qua trực quan hóa tương tác',
  description:
    'Blog tiếng Việt giải thích các khái niệm trí tuệ nhân tạo bằng những trực quan hóa tương tác: CNN, Transformer, Diffusion, GAN và nhiều hơn nữa.',
  author: 'AI4E',
  year: 2026,
} as const;

export type CategorySlug = 'kien-thuc' | 'du-an';

export interface CategoryMeta {
  slug: CategorySlug;
  label: string;
  description: string;
  /** Inner SVG markup (Lucide-style, 24×24 viewBox, stroke icons). */
  icon: string;
  href: string;
}

export const CATEGORIES: Record<CategorySlug, CategoryMeta> = {
  'kien-thuc': {
    slug: 'kien-thuc',
    label: 'Kiến thức',
    description:
      'Những khái niệm nền tảng của học sâu — mạng tích chập, Transformer, mô hình khuếch tán, GAN — được giải thích qua trực quan hóa bạn có thể tự tay khám phá.',
    icon: '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    href: '/kien-thuc/',
  },
  'du-an': {
    slug: 'du-an',
    label: 'Dự án',
    description:
      'Các công cụ nghiên cứu giúp diễn giải, đánh giá và cải thiện mô hình AI: từ phân tích cơ chế chú ý, kiểm tra tính công bằng đến ước lượng năng lượng tiêu thụ.',
    icon: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/>',
    href: '/du-an/',
  },
};

export const CATEGORY_LIST: CategoryMeta[] = [CATEGORIES['kien-thuc'], CATEGORIES['du-an']];

export function isCategorySlug(value: string): value is CategorySlug {
  return Object.prototype.hasOwnProperty.call(CATEGORIES, value);
}

export function getCategory(slug: string): CategoryMeta | undefined {
  return isCategorySlug(slug) ? CATEGORIES[slug] : undefined;
}

export interface NavItem {
  label: string;
  href: string;
}

export const NAV_ITEMS: NavItem[] = [
  { label: 'Trang chủ', href: '/' },
  { label: 'Kiến thức', href: '/kien-thuc/' },
  { label: 'Dự án', href: '/du-an/' },
  { label: 'Chủ đề', href: '/chu-de/' },
  { label: 'Giới thiệu', href: '/gioi-thieu/' },
];

/** True when `pathname` belongs to the section `href` (home matches only itself). */
export function isActivePath(pathname: string, href: string): boolean {
  const norm = (p: string) => (p.endsWith('/') ? p : `${p}/`);
  const path = norm(pathname);
  const target = norm(href);
  if (target === '/') return path === '/';
  return path === target || path.startsWith(target);
}
