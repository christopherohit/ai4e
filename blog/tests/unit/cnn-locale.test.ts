import { describe, expect, it } from 'vitest';
import { translateCnnText } from '../../src/integrations/runtime/cnn-locale';

describe('CNN Vietnamese labels', () => {
  it('translates controls and preserves surrounding whitespace', () => {
    expect(translateCnnText('  Show detail\n')).toBe('  Hiện chi tiết\n');
    expect(translateCnnText('Unit')).toBe('Theo lớp');
    expect(translateCnnText('Output value: 0.123')).toBe('Giá trị đầu ra: 0.123');
  });
  it('preserves technical layer IDs and numerical values', () => {
    for (const term of ['Convolution', 'Pooling', 'Kernel', 'Bias', 'Padding:', 'Stride:', 'conv', 'max_pool']) {
      expect(translateCnnText(term)).toBe(term);
    }
    expect(translateCnnText('conv_1_1')).toBe('conv_1_1');
    expect(translateCnnText('(62, 62, 10)')).toBe('(62, 62, 10)');
  });
});
