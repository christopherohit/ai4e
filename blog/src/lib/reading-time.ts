export const WORDS_PER_MINUTE = 200;

/** Strip MDX/markdown noise so only prose words are counted. */
function toPlainText(source: string): string {
  return source
    .replace(/^---[\s\S]*?\n---\s*/, '') // frontmatter
    .replace(/^\s*(import|export)\s.*$/gm, '') // ESM lines
    .replace(/```[\s\S]*?```/g, ' ') // fenced code
    .replace(/\$\$[\s\S]*?\$\$/g, ' ') // display math
    .replace(/\$[^$\n]+\$/g, ' ') // inline math
    .replace(/<[^>]+>/g, ' ') // JSX / HTML tags
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ') // images
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1') // links → text
    .replace(/[#>*_`~|-]+/g, ' ');
}

export function countWords(source: string): number {
  const words = toPlainText(source).match(/[\p{L}\p{N}]+(?:['’][\p{L}]+)*/gu);
  return words ? words.length : 0;
}

/** Reading time in whole minutes (≈200 words/min), minimum 1. */
export function readingTime(source: string): number {
  return Math.max(1, Math.ceil(countWords(source) / WORDS_PER_MINUTE));
}

/** Vietnamese label: "5 phút đọc". */
export function readingTimeLabel(minutes: number): string {
  return `${minutes} phút đọc`;
}
