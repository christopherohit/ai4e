import { describe, expect, it } from 'vitest';
import { createSearchEngine, highlight, searchDocs, snippet, stripMdx, type SearchDoc } from '../../src/lib/search';

const doc = (partial: Partial<SearchDoc> & Pick<SearchDoc, 'id' | 'title'>): SearchDoc => ({
  url: `/kien-thuc/${partial.id}/`,
  description: '',
  category: 'kien-thuc',
  tags: [],
  headings: [],
  body: '',
  ...partial,
});

const FIXTURE: SearchDoc[] = [
  doc({
    id: 'cnn-explainer',
    title: 'CNN Explainer: mạng nơ-ron tích chập hoạt động thế nào?',
    description: 'Khám phá từng lớp tích chập, ReLU và max pooling của một mạng nơ-ron phân loại ảnh.',
    tags: ['CNN', 'Thị giác máy tính', 'Học sâu'],
    headings: ['Tích chập là gì?', 'Hàm kích hoạt ReLU'],
    body: 'Mạng nơ-ron tích chập dùng các bộ lọc trượt trên ảnh để phát hiện cạnh và họa tiết.',
  }),
  doc({
    id: 'transformer-explainer',
    title: 'Transformer Explainer: bên trong GPT-2',
    description: 'Cơ chế self-attention, embedding và dự đoán token tiếp theo.',
    tags: ['Transformer', 'Cơ chế chú ý', 'Mô hình ngôn ngữ'],
    headings: ['Self-attention', 'Query, Key, Value'],
    body: 'Mỗi token tạo vector query, key và value; attention đo mức liên quan giữa các token. Mạng nơ-ron MLP xử lý tiếp.',
  }),
  doc({
    id: 'dodrio',
    category: 'du-an',
    url: '/du-an/dodrio/',
    title: 'Dodrio: khám phá các đầu chú ý của Transformer',
    description: 'Công cụ tương tác phân tích attention head theo cú pháp và ngữ nghĩa.',
    tags: ['Transformer', 'Cơ chế chú ý', 'Khả năng diễn giải'],
    headings: ['Attention head'],
    body: 'Dodrio tóm tắt trọng số chú ý của BERT.',
  }),
  doc({
    id: 'fairvis',
    category: 'du-an',
    url: '/du-an/fairvis/',
    title: 'FairVis: kiểm tra tính công bằng của mô hình học máy',
    description: 'Phát hiện thiên lệch giữa các nhóm con.',
    tags: ['Công bằng', 'Thiên lệch'],
    headings: ['Nhóm con'],
    body: 'Mạng nơ-ron hay cây quyết định đều có thể thiên lệch.',
  }),
  doc({
    id: 'diffusion-explainer',
    title: 'Diffusion Explainer: từ nhiễu đến hình ảnh',
    description: 'Mô hình khuếch tán Stable Diffusion sinh ảnh từ câu lệnh văn bản.',
    tags: ['Diffusion', 'Mô hình sinh'],
    headings: ['Khử nhiễu'],
    body: 'Mô hình khuếch tán khử nhiễu dần dần trong không gian ẩn.',
  }),
];

const engine = createSearchEngine(FIXTURE);
const ids = (query: string, category?: 'kien-thuc' | 'du-an') =>
  searchDocs(engine, query, { category }).map((hit) => hit.doc.id);

describe('searchDocs ranking', () => {
  it('finds the CNN post first for "mang no ron" (no diacritics)', () => {
    expect(ids('mang no ron')[0]).toBe('cnn-explainer');
  });

  it('finds the CNN post first for "mạng nơ-ron" (with diacritics)', () => {
    expect(ids('mạng nơ-ron')[0]).toBe('cnn-explainer');
  });

  it('ranks transformer/dodrio first for "attention"', () => {
    expect(ids('attention').slice(0, 2).sort()).toEqual(['dodrio', 'transformer-explainer']);
  });

  it('expands English synonyms to Vietnamese text', () => {
    expect(ids('fairness')[0]).toBe('fairvis');
    expect(ids('convolution')[0]).toBe('cnn-explainer');
    expect(ids('khuech tan')[0]).toBe('diffusion-explainer');
  });

  it('supports prefix and fuzzy matching', () => {
    expect(ids('transfor')).toContain('transformer-explainer');
    expect(ids('difusion')[0]).toBe('diffusion-explainer');
  });

  it('filters by category', () => {
    expect(ids('attention', 'du-an')).toEqual(['dodrio']);
  });

  it('returns nothing for blank queries', () => {
    expect(ids('   ')).toEqual([]);
  });
});

describe('highlight', () => {
  it('marks matches regardless of diacritics and escapes HTML', () => {
    expect(highlight('Mạng nơ-ron <b>', ['mang', 'no'])).toBe('<mark>Mạng</mark> <mark>nơ</mark>-ron &#60;b&#62;');
  });

  it('marks prefix matches only up to the term length', () => {
    expect(highlight('Transformer', ['trans'])).toBe('<mark>Trans</mark>former');
  });
});

describe('snippet', () => {
  it('returns an excerpt around the first match', () => {
    const text = `${'mở đầu '.repeat(60)}tích chập ở đây ${'kết thúc '.repeat(60)}`;
    const result = snippet(text, ['tich'], 100);
    expect(result.startsWith('…')).toBe(true);
    expect(result).toContain('tích chập');
    expect(result.length).toBeLessThanOrEqual(102);
  });
});

describe('stripMdx', () => {
  it('removes frontmatter, imports, JSX, math and markdown syntax', () => {
    const source = `---\ntitle: x\n---\nimport A from './a';\n\n<VizEmbed viz="x" />\n\n## Tiêu đề\n\nCông thức $x^2$ và\n\n$$\n\\sum_i x_i\n$$\n\n<Callout type="try">\nNhấp **vào** [đây](https://a.b).\n</Callout>\n\n- mục một\n`;
    expect(stripMdx(source)).toBe('Tiêu đề Công thức và Nhấp vào đây. mục một');
  });
});
