import { describe, expect, it } from 'vitest';
import { countWords, readingTime, readingTimeLabel } from '../../src/lib/reading-time';

describe('countWords', () => {
  it('counts Vietnamese words separated by spaces', () => {
    expect(countWords('Mạng nơ-ron tích chập là gì')).toBe(7);
  });

  it('ignores frontmatter, imports, JSX, code and math', () => {
    const src = [
      '---',
      'title: Bài viết rất dài',
      '---',
      "import VizEmbed from '../../components/VizEmbed.astro';",
      '',
      '<VizEmbed viz="cnn-explainer" />',
      '',
      'Xin chào thế giới.',
      '',
      '```js',
      'const a = 1 + 2 + 3;',
      '```',
      '',
      '$$ y = \\sigma(Wx + b) $$',
      'Công thức $a^2$ đơn giản.',
    ].join('\n');
    expect(countWords(src)).toBe(8);
  });

  it('keeps link text but drops URLs', () => {
    expect(countWords('Xem [tài liệu gốc](https://example.com/a/b/c) nhé')).toBe(5);
  });
});

describe('readingTime', () => {
  it('is at least 1 minute', () => {
    expect(readingTime('')).toBe(1);
    expect(readingTime('vài từ')).toBe(1);
  });

  it('uses 200 words per minute, rounded up', () => {
    expect(readingTime(Array(200).fill('từ').join(' '))).toBe(1);
    expect(readingTime(Array(201).fill('từ').join(' '))).toBe(2);
    expect(readingTime(Array(1000).fill('từ').join(' '))).toBe(5);
  });

  it('formats a Vietnamese label', () => {
    expect(readingTimeLabel(5)).toBe('5 phút đọc');
  });
});
