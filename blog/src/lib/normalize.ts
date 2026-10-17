/**
 * Normalize Vietnamese (or any) text for search:
 * NFD → strip combining marks → đ/Đ → d/D → lowercase.
 * "Mạng nơ-ron Đối nghịch" → "mang no-ron doi nghich"
 */
export function normalizeVi(input: string): string {
  if (!input) return '';
  return input
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/đ/g, 'd')
    .replace(/Đ/g, 'D')
    .toLowerCase();
}

/** Split normalized text into search terms (letters/digits), dropping empties. */
export function tokenizeVi(input: string): string[] {
  return normalizeVi(input)
    .split(/[^\p{L}\p{N}]+/u)
    .filter(Boolean);
}
