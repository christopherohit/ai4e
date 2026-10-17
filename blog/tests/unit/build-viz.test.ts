import { describe, expect, it } from 'vitest';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { applyReplacements, mergeManifest, parseArgs, selectRecipes } from '../../scripts/viz-lib.mjs';
import recipes from '../../scripts/viz-recipes.mjs';

describe('viz-recipes', () => {
  it('bounds Transformer model downloads while preserving chunk order and cache hits', async () => {
    const replacement = recipes['transformer-explainer'].replacements.find((entry: { file: string }) => entry.file === 'src/utils/fetchChunks.js');
    const urls = Array.from({ length: 20 }, (_, index) => String(index));
    const cachedResponses = urls.map((url, index) => index === 3 ? { arrayBuffer: async () => url } : undefined);
    let active = 0;
    let peak = 0;
    const requested: string[] = [];
    const cache = { put: async () => {} };
    const fetch = async (url: string) => {
      requested.push(url);
      active++;
      peak = Math.max(peak, active);
      await new Promise(resolve => setTimeout(resolve, Number(url) % 3));
      active--;
      return { ok: true, clone: () => ({}), arrayBuffer: async () => url };
    };
    const download = new Function('chunkUrls', 'cachedResponses', 'cache', 'fetch', `return (async () => { let hasCache = false; ${replacement.to}\nreturn { hasCache, modelBuffers }; })();`);
    const result = await download(urls, cachedResponses, cache, fetch);
    expect(peak).toBe(6);
    expect(requested).toHaveLength(19);
    expect(requested).not.toContain('3');
    expect(result).toEqual({ hasCache: true, modelBuffers: urls });
    expect(await download([], [], cache, fetch)).toEqual({ hasCache: false, modelBuffers: [] });
    await expect(download(['missing'], [], cache, async () => ({ ok: false }))).rejects.toThrow('Failed to fetch missing');
  });

  it('self-hosts CNN runtime dependencies and removes obsolete startup scripts', () => {
    const directory = mkdtempSync(join(tmpdir(), 'viz-cnn-'));
    try {
      mkdirSync(join(directory, 'public'));
      const dependencies = [
        ['https://d3js.org/d3.v5.js', '/vendor/cnn/d3.v5.js'],
        ['https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf.min.js', '/vendor/cnn/tf.min.js'],
        ['https://use.fontawesome.com/releases/v5.3.1/js/all.js', '/vendor/cnn/fontawesome.js'],
        ['https://cdn.jsdelivr.net/gh/cferdinandi/smooth-scroll@15/dist/smooth-scroll.polyfills.min.js', '/vendor/cnn/smooth-scroll.min.js'],
        ['https://cdn.jsdelivr.net/npm/bulma@0.8.0/css/bulma.min.css', '/vendor/cnn/bulma.min.css'],
        ['https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js', './vendor/mathjax-tex-svg.js'],
        ['<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>', ''],
        ['    <script\n      type="module"\n      src="https://unpkg.com/recommender-overlay/dist/recommender-overlay.es.js"\n    ></script>', ''],
      ];
      writeFileSync(join(directory, 'public/index.html'), dependencies.map(([url]) => url).join('\n'));
      applyReplacements(directory, recipes['cnn-explainer'].replacements.filter((replacement: { from: string }) => dependencies.some(([url]) => replacement.from === url)));
      expect(readFileSync(join(directory, 'public/index.html'), 'utf8')).toBe(dependencies.map(([, asset]) => asset).join('\n'));
      expect(recipes['cnn-explainer'].build).toContain('mathjax@3.2.2');
      expect(recipes['cnn-explainer'].build).toContain('-o public/vendor/mathjax-tex-svg.js');
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it.each([
    { slug: 'cnn-explainer', quote: '"' },
    { slug: 'dodrio', quote: "'" },
  ])('keeps the optional $slug video off the iframe startup path', ({ slug, quote }) => {
    const directory = mkdtempSync(join(tmpdir(), `viz-${slug}-video-`));
    try {
      mkdirSync(join(directory, 'src/article'), { recursive: true });
      writeFileSync(join(directory, 'src/article/Youtube.svelte'), `<script context=${quote}module${quote}>\nvar tag = document.createElement(${quote}script${quote});\nwindow.dispatchEvent(new Event(${quote}iframeApiReady${quote}));\n</script>\nplayer.seekTo(startSecond);\n<div id={playerId} />`);
      applyReplacements(directory, recipes[slug].replacements.filter((replacement: { file: string }) => replacement.file === 'src/article/Youtube.svelte'));
      const output = readFileSync(join(directory, 'src/article/Youtube.svelte'), 'utf8');
      expect(output).toContain('if (!embedded) {');
      expect(output).toContain('window.parent !== window');
      expect(output).toContain('if (!player) {');
      expect(output).toContain('&t=${startSecond}s');
      expect(output).toContain('Watch the tutorial on YouTube');
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it('self-hosts Dodrio styles and removes the recommender startup script', () => {
    const directory = mkdtempSync(join(tmpdir(), 'viz-dodrio-'));
    try {
      mkdirSync(join(directory, 'public'));
      mkdirSync(join(directory, 'src'));
      writeFileSync(join(directory, 'public/index.html'), '<link href="https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.1/css/bulma.min.css"><script src="https://use.fontawesome.com/releases/v5.3.1/js/all.js"></script>');
      writeFileSync(join(directory, 'src/App.svelte'), '<svelte:head>\n  <script\n    type="module"\n    src="https://unpkg.com/recommender-overlay/dist/recommender-overlay.es.js"\n  ></script>\n</svelte:head>');
      applyReplacements(directory, recipes.dodrio.replacements.filter((replacement: { from: string }) => /https:\/\//.test(replacement.from)));
      expect(readFileSync(join(directory, 'public/index.html'), 'utf8')).toBe('<link href="./vendor/bulma.min.css"><script src="/vendor/cnn/fontawesome.js"></script>');
      expect(readFileSync(join(directory, 'src/App.svelte'), 'utf8')).toBe('<svelte:head>\n\n</svelte:head>');
      expect(recipes.dodrio.build).toContain('-o public/vendor/bulma.min.css');
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it.each([
    { slug: 'viscuit', file: 'index.html', version: 7, asset: 'vendor/d3.v7.min.js' },
    { slug: 'diffusion-explainer', file: 'index.html', version: 7, asset: 'vendor/d3.v7.min.js' },
    { slug: 'energyvis', file: 'public/index.html', version: 4, asset: 'public/vendor/d3.v4.min.js' },
  ])('serves $slug D3 from the copied runtime instead of a CDN', ({ slug, file: path, version, asset }) => {
    const directory = mkdtempSync(join(tmpdir(), `viz-${slug}-`));
    const file = join(directory, path);
    const url = `https://d3js.org/d3.v${version}.min.js`;
    try {
      mkdirSync(dirname(file), { recursive: true });
      writeFileSync(file, `<script src="${url}"></script>`);
      applyReplacements(directory, recipes[slug].replacements.filter((replacement: { from: string }) => replacement.from === url));
      expect(readFileSync(file, 'utf8')).toBe(`<script src="./vendor/d3.v${version}.min.js"></script>`);
      expect(recipes[slug].build).toContain(`-o ${asset}`);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it('bundles GAN Lab Chart.js locally instead of depending on a runtime CDN', () => {
    const directory = mkdtempSync(join(tmpdir(), 'viz-ganlab-'));
    const file = join(directory, 'demo', 'ganlab.html');
    try {
      mkdirSync(join(directory, 'demo'));
      writeFileSync(file, '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.4.0/Chart.bundle.min.js"></script>');
      applyReplacements(directory, recipes.ganlab.replacements.filter((replacement: { file: string }) => replacement.file === 'demo/ganlab.html'));
      expect(readFileSync(file, 'utf8')).toBe('<script src="../node_modules/chart.js/dist/Chart.bundle.min.js"></script>');
      expect(recipes.ganlab.build).toContain('--inline-scripts');
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it('has one recipe per spec viz slug, each with src and out', () => {
    expect(Object.keys(recipes).sort()).toEqual(
      ['cnn-explainer', 'diffusion-explainer', 'dodrio', 'energyvis', 'fairvis', 'ganlab', 'transformer-explainer', 'viscuit'],
    );
    for (const [slug, r] of Object.entries(recipes)) {
      expect(r.slug).toBe(slug);
      expect(typeof r.src).toBe('string');
      expect(typeof r.out).toBe('string');
    }
  });
});

describe('parseArgs', () => {
  it('splits slugs and flags', () => {
    expect(parseArgs(['cnn-explainer', '--skip-install', 'ganlab'])).toEqual({
      slugs: ['cnn-explainer', 'ganlab'],
      skipInstall: true,
      list: false,
    });
    expect(parseArgs([])).toEqual({ slugs: [], skipInstall: false, list: false });
    expect(parseArgs(['--list']).list).toBe(true);
  });
});

describe('selectRecipes', () => {
  const all = { a: { slug: 'a' }, b: { slug: 'b' } };
  it('returns all recipes when no slug given', () => {
    expect(selectRecipes(all, []).map((r) => r.slug)).toEqual(['a', 'b']);
  });
  it('returns the requested recipes in order', () => {
    expect(selectRecipes(all, ['b']).map((r) => r.slug)).toEqual(['b']);
  });
  it('throws on unknown slug', () => {
    expect(() => selectRecipes(all, ['zzz'])).toThrow(/zzz/);
  });
});

describe('applyReplacements', () => {
  it('replaces strings in files relative to root', () => {
    const dir = mkdtempSync(join(tmpdir(), 'viz-'));
    writeFileSync(join(dir, 'f.js'), 'fetch("/data/x.json"); fetch("/data/y.json");');
    applyReplacements(dir, [{ file: 'f.js', from: '"/data/', to: '"./data/', all: true }]);
    expect(readFileSync(join(dir, 'f.js'), 'utf8')).toBe('fetch("./data/x.json"); fetch("./data/y.json");');
  });
  it('throws when the search string is missing (source drifted)', () => {
    const dir = mkdtempSync(join(tmpdir(), 'viz-'));
    writeFileSync(join(dir, 'f.js'), 'nothing here');
    expect(() => applyReplacements(dir, [{ file: 'f.js', from: 'absent', to: 'x' }])).toThrow(/absent/);
  });
});

describe('mergeManifest', () => {
  it('keeps other entries and overwrites the built one', () => {
    const prev = { a: { ok: true, builtAt: 't0' }, b: { ok: false, error: 'boom' } };
    expect(mergeManifest(prev, 'b', { ok: true, builtAt: 't1' })).toEqual({
      a: { ok: true, builtAt: 't0' },
      b: { ok: true, builtAt: 't1' },
    });
  });
});
