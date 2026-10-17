import { expect, test } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const screens = join(process.cwd(), 'test-results/screens');
const routes = [
  { name: 'home', url: '/', vizSelector: '' },
  { name: 'post', url: '/kien-thuc/cnn-explainer/', vizSelector: '#cnn-svg .node-group' },
  { name: 'search', url: '/tim-kiem/?q=mang%20no%20ron', vizSelector: '' },
  { name: 'post-transformer', url: '/kien-thuc/transformer-explainer/', vizSelector: '.main-section.active .nodes' },
  { name: 'post-diffusion', url: '/kien-thuc/diffusion-explainer/', vizSelector: '#architecture-container img' },
  { name: 'post-ganlab', url: '/kien-thuc/ganlab/', vizSelector: 'gan-lab #iteration-count' },
  { name: 'post-dodrio', url: '/du-an/dodrio/', vizSelector: '.graph-container svg' },
  { name: 'post-fairvis', url: '/du-an/fairvis/', vizSelector: '#root button' },
  { name: 'post-viscuit', url: '/du-an/viscuit/', vizSelector: '#selection .subgroup-entry-image-display img' },
  { name: 'post-energyvis', url: '/du-an/energyvis/', vizSelector: '#energymap svg' },
];

for (const theme of ['light', 'dark'] as const) {
  for (const width of [1440, 375]) {
    test.describe(`${theme} at ${width}px`, () => {
      test.use({ viewport: { width, height: 900 }, colorScheme: theme });

      for (const route of routes) {
        test(`visual review: ${route.name}`, async ({ page }) => {
          await mkdir(screens, { recursive: true });
          await page.addInitScript((selectedTheme) => localStorage.setItem('theme', selectedTheme), theme);
          const response = await page.goto(route.url, { waitUntil: 'load' });
          expect(response?.status()).toBe(200);
          await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
          await page.evaluate(() => document.fonts.ready);

          if (route.name === 'search') {
            await expect(page.locator('[data-search-page-results] a[href="/kien-thuc/cnn-explainer/"]')).toBeVisible();
            await expect(page.locator('[data-search-page-results] mark').first()).toBeVisible();
          }

          for (const image of await page.locator('main img:visible').all()) {
            await image.scrollIntoViewIfNeeded();
            await expect.poll(() => image.evaluate((element: HTMLImageElement) => element.complete && element.naturalWidth > 0)).toBe(true);
          }

          if (route.vizSelector) {
            await expect(page.locator('iframe')).toHaveCount(0);
            const root = page.locator(route.name === 'post' ? '#cnn-tool-root' : '#native-tool-root');
            await root.scrollIntoViewIfNeeded();
            await expect(root.locator(route.vizSelector).first()).toBeVisible({ timeout: 60_000 });
            await expect(root).toHaveAttribute('aria-busy', 'false');
            if (route.name === 'post-transformer') {
              await expect(root.getByRole('button', { name: 'Sinh token', exact: true })).toBeEnabled({ timeout: 90_000 });
            }
            if (width === 375) {
              const stage = root;
              await stage.focus();
              await page.keyboard.press('ArrowRight');
              await expect.poll(() => stage.evaluate((element) => element.scrollLeft)).toBeGreaterThan(0);
              await stage.evaluate((element: HTMLElement) => { element.scrollLeft = 0; element.blur(); });
            }
          }

          const overflow = await page.evaluate(() => Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - document.documentElement.clientWidth);
          expect(overflow, 'horizontal page overflow').toBeLessThanOrEqual(1);
          await page.evaluate(() => window.scrollTo(0, 0));
          await page.screenshot({ path: join(screens, `${route.name}-${theme}-${width}.png`), fullPage: true, animations: 'disabled' });
          if (route.name === 'post') {
            const stage = page.locator('#cnn-tool-root');
            await stage.scrollIntoViewIfNeeded();
            if (width === 375) {
              await stage.evaluate((element) => { element.scrollLeft = element.scrollWidth; });
              await expect(stage.locator('#cnn-svg .node-output').first()).toBeInViewport();
              await page.locator('.tool-workspace').screenshot({ path: join(screens, `post-${theme}-${width}-panned.png`), animations: 'disabled' });
              await stage.evaluate((element) => { element.scrollLeft = 0; });
            }
          }
        });
      }
    });
  }
}
