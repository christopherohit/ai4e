import { expect, test, type Page, type Request } from '@playwright/test';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const DIST = join(process.cwd(), 'dist');

function htmlFiles(directory: string): string[] {
  return readdirSync(directory).flatMap((name) => {
    const file = join(directory, name);
    if (statSync(file).isDirectory()) return htmlFiles(file);
    return name.endsWith('.html') ? [file] : [];
  });
}

const pages = htmlFiles(DIST);
const routes = pages.map((file) => '/' + relative(DIST, file).replace(/index\.html$/, '').replace(/\\/g, '/'));
const postRoutes = routes.filter((route) => /^\/(kien-thuc|du-an)\/[^/]+\/$/.test(route));
const vizSelectors: Record<string, string> = {
  nerf: 'canvas',
  'gaussian-splatting': 'canvas',
  synctalk: 'canvas',
  instag: 'canvas',
  'cnn-explainer': '#cnn-svg .node-group',
  'transformer-explainer': '.main-section.active .nodes',
  'diffusion-explainer': '#architecture-container img',
  ganlab: 'gan-lab #iteration-count',
  dodrio: '.graph-container svg',
  fairvis: '#root button',
  viscuit: '#selection .subgroup-entry-image-display img',
  energyvis: '#energymap svg',
  'qwen3-8-flash-next': '.layer-map',
  'gated-delta-net': '.heat',
  'sparse-attention': '.attn',
  'gradient-descent': '.glass-screen canvas',
  'batch-mini-batch-sgd': '.glass-screen canvas',
  'naive-bayes': '.glass-screen canvas',
  'svm': '.glass-screen canvas',
  'bias-variance': '.glass-screen canvas',
  'l1-l2-regularization': '.glass-screen canvas',
  'sdvx-modeler': '.pipe',
  bswmd: '.bswmd-pipeline',
};
const vizRoutes = postRoutes.filter((route) => Object.hasOwn(vizSelectors, route.split('/')[2]));

async function noHorizontalScroll(page: Page) {
  const overflow = await page.evaluate(() => ({
    document: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    body: document.body.scrollWidth - document.documentElement.clientWidth,
  }));
  expect(overflow.document, 'document horizontal overflow (px)').toBeLessThanOrEqual(1);
  expect(overflow.body, 'body horizontal overflow (px)').toBeLessThanOrEqual(1);
}

test('all built routes return 200, with a working 404 page', async ({ request }) => {
  expect(postRoutes).toHaveLength(24);
  expect(vizRoutes.map((route) => route.split('/')[2]).sort()).toEqual(Object.keys(vizSelectors).sort());
  for (const route of routes) {
    const response = await request.get(route);
    expect(response.status(), route).toBe(200);
    expect(response.headers()['content-type'], route).toContain('text/html');
  }
  const index = await request.get('/search-index.json');
  expect(index.status()).toBe(200);
  expect(await index.json()).toHaveLength(24);
  expect((await request.get('/missing-smoke-route/')).status()).toBe(404);
});

test('internal links and assets in every dist HTML file resolve', async ({ page, baseURL }) => {
  const documents = pages.map((file) => ({
    route: '/' + relative(DIST, file).replace(/\\/g, '/'),
    html: readFileSync(file, 'utf8'),
  }));
  const links = await page.evaluate(({ documents, origin }) => {
    const parser = new DOMParser();
    const home = parser.parseFromString(documents.find(({ route }) => route === '/index.html')!.html, 'text/html');
    const site = new URL(home.querySelector('link[rel="canonical"]')?.getAttribute('href') ?? origin);
    return documents.flatMap(({ route, html }) => {
      const document = parser.parseFromString(html, 'text/html');
      const documentURL = new URL(route, site);
      const base = new URL(document.querySelector('base[href]')?.getAttribute('href') ?? documentURL.href, documentURL);
      return Array.from(document.querySelectorAll('[href], [src], [srcset]')).flatMap((element) => {
        const references = [element.getAttribute('href'), element.getAttribute('src')];
        const srcset = element.getAttribute('srcset');
        if (srcset) references.push(...srcset.split(',').map((candidate) => candidate.trim().split(/\s+/)[0]));
        return references.flatMap((reference) => {
          if (!reference || /\[\[|\{\{/.test(reference)) return [];
          const url = new URL(reference, base);
          if (url.origin !== site.origin && url.origin !== origin) return [];
          return [{ from: route, reference, pathname: decodeURIComponent(url.pathname) }];
        });
      });
    });
  }, { documents, origin: baseURL! });
  expect(links.length).toBeGreaterThan(100);
  const broken = links.filter(({ pathname }) => {
    const target = join(DIST, pathname);
    return !existsSync(target) || (!statSync(target).isFile() && !existsSync(join(target, 'index.html')));
  });
  expect(broken).toEqual([]);
});

for (const route of vizRoutes) {
  test(`viz renders without failed same-origin requests: ${route}`, async ({ page, baseURL }) => {
    const slug = route.split('/')[2];
    const failed: string[] = [];
    const errors: string[] = [];
    const pending = new Set<Request>();
    const isLocal = (url: string) => new URL(url).origin === baseURL;
    page.on('request', (request) => {
      if (isLocal(request.url())) pending.add(request);
    });
    page.on('response', (response) => {
      if (isLocal(response.url()) && response.status() >= 400) failed.push(`${response.status()} ${response.url()}`);
    });
    page.on('requestfinished', (request) => pending.delete(request));
    page.on('requestfailed', (request) => {
      pending.delete(request);
      const errorText = request.failure()?.errorText;
      const cancelledMediaRange = request.resourceType() === 'media' && errorText === 'net::ERR_ABORTED';
      if (isLocal(request.url()) && !cancelledMediaRange) failed.push(`${errorText} ${request.url()}`);
    });
    page.on('pageerror', (error) => errors.push(error.message));

    if (['cnn-explainer', 'dodrio'].includes(slug)) {
      await page.route(/https:\/\/(d3js\.org|cdn\.jsdelivr\.net|cdnjs\.cloudflare\.com|use\.fontawesome\.com|polyfill\.io|unpkg\.com|www\.youtube\.com)\//, (intercepted) => intercepted.abort());
    }
    if (['diffusion-explainer', 'energyvis', 'viscuit', 'ganlab'].includes(slug)) {
      await page.route(/https:\/\/d3js\.org\/|https:\/\/cdnjs\.cloudflare\.com\/ajax\/libs\/Chart\.js\//, (intercepted) => intercepted.abort());
    }
    await page.goto(route, { waitUntil: 'domcontentloaded' });
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('iframe')).toHaveCount(0);
    const root = page.locator(slug === 'cnn-explainer' ? '#cnn-tool-root' : '#native-tool-root');
    await root.scrollIntoViewIfNeeded();
    await expect(root.locator(vizSelectors[slug]).first()).toBeVisible({ timeout: 60_000 });
    if (slug === 'ganlab') {
      await expect.poll(() => page.evaluate(() => Object.keys(Reflect.get(window, 'Chart')?.instances ?? {}).length)).toBeGreaterThanOrEqual(2);
    }
    // The 3DGS lab streams a 33 MB trained model and draws it with (often software) WebGL before it is ready.
    await expect(root).toHaveAttribute('aria-busy', 'false', { timeout: slug === 'gaussian-splatting' ? 120_000 : 5_000 });
    await expect.poll(() => [...pending].map((request) => request.url()), { timeout: 30_000 }).toEqual([]);
    expect(failed, 'failed same-origin requests').toEqual([]);
    expect(errors, 'uncaught page/iframe errors').toEqual([]);
    if (slug === 'transformer-explainer') {
      await expect(root.getByRole('button', { name: 'Sinh token', exact: true })).toBeEnabled({ timeout: 90_000 });
      await expect.poll(() => [...pending].map((request) => request.url()), { timeout: 30_000 }).toEqual([]);
    }
    expect(failed, 'failed same-origin requests').toEqual([]);
    expect(errors, 'uncaught page/iframe errors').toEqual([]);
  });
}

test('theme toggle persists in both directions across reload', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  const toggle = page.getByRole('button', { name: 'Chuyển giao diện sáng/tối' });
  for (const theme of ['dark', 'light']) {
    await toggle.click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
    expect(await page.evaluate(() => localStorage.getItem('theme'))).toBe(theme);
    await page.reload();
    await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
    await expect(toggle).toHaveAttribute('aria-pressed', String(theme === 'dark'));
  }
});

for (const shortcut of ['Meta+k', 'Control+k']) {
  test(`${shortcut} search "mang no ron" shows the CNN post`, async ({ page }) => {
    await page.goto('/');
    await page.keyboard.press(shortcut);
    const input = page.locator('#search-palette-input');
    await expect(input).toBeFocused();
    await input.fill('mang no ron');
    const cnn = page.locator('[data-search-results] a[href="/kien-thuc/cnn-explainer/"]');
    await expect(cnn).toBeVisible();
    await cnn.click();
    await expect(page).toHaveURL(/\/kien-thuc\/cnn-explainer\/$/);
  });
}

test('search page shows results for ?q=', async ({ page }) => {
  await page.goto('/tim-kiem/?q=mang%20no%20ron');
  await expect(page.locator('[data-search-page-results] a[href="/kien-thuc/cnn-explainer/"]')).toBeVisible();
  await expect(page.locator('[data-search-page-results] mark').first()).toBeVisible();
});

test.describe('375px viewport', () => {
  test.use({ viewport: { width: 375, height: 800 } });
  for (const route of ['/', ...postRoutes, '/tim-kiem/?q=cnn', '/chu-de/', '/gioi-thieu/']) {
    test(`no horizontal scroll: ${route}`, async ({ page }) => {
      await page.goto(route, { waitUntil: 'domcontentloaded' });
      await page.evaluate(() => document.fonts.ready);
      if (route.startsWith('/tim-kiem/')) {
        await expect(page.locator('[data-search-page-results] a').first()).toBeVisible();
      }
      await noHorizontalScroll(page);
      await page.locator('.site-footer').scrollIntoViewIfNeeded();
      await noHorizontalScroll(page);
    });
  }
});
