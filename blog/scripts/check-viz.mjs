#!/usr/bin/env node
// Smoke-check built viz: serve public/ on a free port, open /viz/<slug>/ in Chrome,
// report failed same-origin requests, console errors and a screenshot.
//   node scripts/check-viz.mjs [slug...] [--wait=8000]
// Screenshots go to test-results/viz/<slug>.png.
import { createServer } from 'node:http';
import { createReadStream, existsSync, mkdirSync, statSync } from 'node:fs';
import { extname, join, normalize, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';
import recipes from './viz-recipes.mjs';

const BLOG = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PUBLIC = join(BLOG, 'public');
const SHOTS = join(BLOG, 'test-results', 'viz');
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.webp': 'image/webp', '.gif': 'image/gif', '.wasm': 'application/wasm', '.woff': 'font/woff', '.woff2': 'font/woff2',
  '.onnx': 'application/octet-stream', '.bin': 'application/octet-stream', '.ico': 'image/x-icon', '.txt': 'text/plain',
};

function serve() {
  const server = createServer((req, res) => {
    let path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    let file = normalize(join(PUBLIC, path));
    if (!file.startsWith(PUBLIC)) return res.writeHead(403).end();
    if (existsSync(file) && statSync(file).isDirectory()) {
      if (!path.endsWith('/')) return res.writeHead(301, { Location: path + '/' }).end();
      file = join(file, 'index.html');
    }
    if (!existsSync(file)) return res.writeHead(404).end('not found');
    res.writeHead(200, { 'Content-Type': TYPES[extname(file).toLowerCase()] ?? 'application/octet-stream' });
    createReadStream(file).pipe(res);
  });
  return new Promise((ok) => server.listen(0, '127.0.0.1', () => ok(server)));
}

const argv = process.argv.slice(2);
const wait = Number(argv.find((a) => a.startsWith('--wait='))?.split('=')[1] ?? 8000);
const slugs = argv.filter((a) => !a.startsWith('--'));
const targets = slugs.length ? slugs : Object.keys(recipes).filter((s) => existsSync(join(PUBLIC, 'viz', s, 'index.html')));

const server = await serve();
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ executablePath: process.env.CHROME ?? '/usr/bin/google-chrome' });
mkdirSync(SHOTS, { recursive: true });
let bad = 0;

for (const slug of targets) {
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const failed = [];
  const errors = [];
  page.on('response', (r) => { if (r.status() >= 400 && r.url().startsWith(origin)) failed.push(`${r.status()} ${r.url().slice(origin.length)}`); });
  page.on('requestfailed', (r) => { if (r.url().startsWith(origin)) failed.push(`FAILED ${r.url().slice(origin.length)} ${r.failure()?.errorText}`); });
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 300)); });
  page.on('pageerror', (e) => errors.push(`pageerror: ${String(e.message).slice(0, 300)}`));
  try {
    await page.goto(`${origin}/viz/${slug}/`, { waitUntil: 'load', timeout: 60000 });
    await page.waitForTimeout(wait);
  } catch (e) {
    errors.push(`goto: ${e.message.split('\n')[0]}`);
  }
  const bodyText = (await page.evaluate(() => document.body?.innerText ?? '').catch(() => '')).replace(/\s+/g, ' ').slice(0, 160);
  await page.screenshot({ path: join(SHOTS, `${slug}.png`) }).catch(() => {});
  const ok = failed.length === 0;
  if (!ok) bad++;
  console.log(`\n${ok ? '✔' : '✘'} ${slug}: ${failed.length} failed same-origin requests, ${errors.length} console errors`);
  console.log(`  text: ${bodyText}`);
  for (const f of failed.slice(0, 20)) console.log(`  [req] ${f}`);
  for (const e of errors.slice(0, 10)) console.log(`  [err] ${e}`);
  await page.close();
}

await browser.close();
server.close();
process.exitCode = bad ? 1 : 0;
