import { expect, it } from 'vitest';
import { translateToolLabel } from '../../src/integrations/runtime/tool-locale';

it('translates tool controls without changing model data or whitespace', () => {
  expect(translateToolLabel('  Generate\n')).toBe('  Sinh token\n');
  expect(translateToolLabel('SELECT A DATASET')).toBe('Chọn bộ dữ liệu');
  expect(translateToolLabel('Data Distribution')).toBe('Data Distribution');
  expect(translateToolLabel('0.1234')).toBe('0.1234');
  expect(translateToolLabel('BERT')).toBe('BERT');
  expect(translateToolLabel('Data visualization empowers us to')).toBe('Data visualization empowers us to');
});

it('keeps technical labels in English, including case, punctuation and whitespace', () => {
  for (const term of ['Attention', 'Embedding', 'Query', 'Key', 'Value', 'Dropout', 'Generator', 'Discriminator', 'Learning Rate:', '  Confusion Matrix\n']) {
    expect(translateToolLabel(term)).toBe(term);
  }
});
