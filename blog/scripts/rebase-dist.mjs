// Rewrite root-relative URLs in dist/ for a sub-path deployment (GitHub Pages project site: BASE_PATH=/ai4e).
// Astro already prefixes its own assets when `base` is set; this covers hand-written links in components and articles
// (href="/kien-thuc/…"), the search index and the adapters of external tools. Paths that already carry the base, and
// protocol-relative URLs (//host), are left alone. Without BASE_PATH (or with "/") the script does nothing.
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const base = (process.env.BASE_PATH ?? '').replace(/\/+$/, '');
if (!base) process.exit(0);
const dist = resolve('dist');

const prefix = value => (value.startsWith('/') && !value.startsWith('//') && value !== base && !value.startsWith(`${base}/`) ? base + value : value);

function files(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? files(path) : [path];
  });
}

let changed = 0;
for (const file of files(dist).filter(path => /\.(html|json|js|css)$/.test(path))) {
  const source = readFileSync(file, 'utf8');
  let output = source;
  if (file.endsWith('.html')) {
    output = output
      .replace(/(\s(?:href|src|action|poster|data-src)=)(["'])(\/[^"']*)\2/g, (_, attr, quote, value) => `${attr}${quote}${prefix(value)}${quote}`)
      .replace(/(\ssrcset=)(["'])([^"']*)\2/g, (_, attr, quote, value) => `${attr}${quote}${value.split(',').map(part => part.trim().replace(/^\S+/, url => prefix(url))).join(', ')}${quote}`)
      .replace(/(url\(\s*['"]?)(\/[^'")\s]+)/g, (_, open, value) => open + prefix(value));
  } else if (file.endsWith('search-index.json')) {
    output = output.replace(/("url":")(\/[^"]*)/g, (_, key, value) => key + prefix(value));
  } else if (/[\\/](native|viz)[\\/]/.test(file) && /\.(js|css|json)$/.test(file)) {
    output = output.replace(/(["'`(])\/(viz|native|vendor)\//g, (_, quote, dir) => `${quote}${base}/${dir}/`);
  }
  if (output !== source) { writeFileSync(file, output); changed++; }
}
console.log(`rebase-dist: ${changed} files prefixed with ${base}`);
