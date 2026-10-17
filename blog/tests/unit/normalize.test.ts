import { describe, expect, it } from 'vitest';
import { normalizeVi, tokenizeVi } from '../../src/lib/normalize';

describe('normalizeVi', () => {
  it('strips Vietnamese diacritics and lowercases', () => {
    expect(normalizeVi('Mạng nơ-ron tích chập')).toBe('mang no-ron tich chap');
    expect(normalizeVi('Thị giác máy tính')).toBe('thi giac may tinh');
    expect(normalizeVi('Học sâu Ứng dụng')).toBe('hoc sau ung dung');
  });

  it('maps đ/Đ to d', () => {
    expect(normalizeVi('Đối nghịch')).toBe('doi nghich');
    expect(normalizeVi('đường đi')).toBe('duong di');
  });

  it('handles precomposed and decomposed input identically', () => {
    const precomposed = 'Khuếch tán';
    const decomposed = precomposed.normalize('NFD');
    expect(normalizeVi(precomposed)).toBe(normalizeVi(decomposed));
    expect(normalizeVi(decomposed)).toBe('khuech tan');
  });

  it('leaves ASCII untouched except case', () => {
    expect(normalizeVi('Transformer GAN-Lab 2024')).toBe('transformer gan-lab 2024');
  });

  it('returns empty string for empty input', () => {
    expect(normalizeVi('')).toBe('');
  });
});

describe('tokenizeVi', () => {
  it('splits on non-letter characters', () => {
    expect(tokenizeVi('Mạng nơ-ron, tích chập!')).toEqual(['mang', 'no', 'ron', 'tich', 'chap']);
  });
});
