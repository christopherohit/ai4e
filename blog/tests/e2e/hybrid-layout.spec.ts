import { expect, test } from '@playwright/test';

const lessons = [
  ['kien-thuc', 'nerf'],
  ['kien-thuc', 'gaussian-splatting'],
  ['du-an', 'synctalk'],
  ['du-an', 'instag'],
  ['kien-thuc', 'cnn-explainer'],
  ['kien-thuc', 'transformer-explainer'],
  ['kien-thuc', 'diffusion-explainer'],
  ['kien-thuc', 'ganlab'],
  ['du-an', 'dodrio'],
  ['du-an', 'fairvis'],
  ['du-an', 'viscuit'],
  ['du-an', 'energyvis'],
];

for (const [category, slug] of lessons) {
  test(`${slug} uses the hybrid layout without the source box`, async ({ page }) => {
    await page.goto(`/${category}/${slug}/`, { waitUntil: 'domcontentloaded' });
    await expect(page.locator('.hybrid-header')).toBeVisible();
    await expect(page.locator('.tool-workspace')).toHaveCount(1);
    await expect(page.locator('#cong-cu')).toHaveCount(1);
    await expect(page.locator('#huong-dan .prose')).toBeVisible();
    await expect(page.locator('.source-box')).toHaveCount(0);
    await expect(page.locator('article').getByText('Nguồn & giấy phép', { exact: true })).toHaveCount(0);
    await page.setViewportSize({ width: 1920, height: 1080 });
    const workspaceWidth = await page.locator('.tool-workspace').evaluate(element => element.clientWidth);
    expect(workspaceWidth).toBeGreaterThan(1440);
    await page.setViewportSize({ width: 375, height: 812 });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(1);
  });
}
