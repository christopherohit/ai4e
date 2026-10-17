import { expect, test } from '@playwright/test';

// WebGL (3DGS) runs on SwiftShader in headless Chrome: ~8 s per frame for 1.25 M Gaussians on CPU, so the 3DGS
// test gets a longer budget and its layout checks wait for the main thread instead of polling.
test.use({ launchOptions: { executablePath: process.env.CHROME ?? '/usr/bin/google-chrome', args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] } });
test.describe.configure({ timeout: 360_000 });

const labs = [
  { category: 'kien-thuc', slug: 'nerf', metric: '22.48 dB', select: false },
  { category: 'kien-thuc', slug: 'gaussian-splatting', metric: '22.48 dB', select: true },
  { category: 'du-an', slug: 'synctalk', metric: '35.49 dB', select: false },
  { category: 'du-an', slug: 'instag', metric: '33.63 dB', select: false },
];

for (const { category, slug, metric, select } of labs) {
  test(`${slug}: trained-model lab loads, inspects a pixel, reacts and resets`, async ({ page }, testInfo) => {
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`/${category}/${slug}/`);
    const lab = page.locator('[data-radiance-mode]');
    await expect(lab).toHaveAttribute('aria-busy', 'false', { timeout: 120_000 });
    await expect(lab.locator('[data-error]')).toBeHidden();
    await expect(page.locator('iframe')).toHaveCount(0);
    await expect(lab.locator('.metrics')).toContainText(metric);
    await expect(lab.locator('[data-loading]')).toBeHidden();
    if (select) await lab.locator('[data-view]').selectOption('1');
    const readout = lab.locator('[data-readout]');
    await expect(lab.locator('[data-chart] path.series').first()).toBeVisible();
    await expect(readout).not.toBeEmpty();
    const before = await lab.locator('[data-selection]').textContent();

    // Click a pixel: the inspector switches to that ray / frame.
    const canvas = lab.locator('[data-scene-canvas]');
    const box = (await canvas.boundingBox())!;
    if (slug === 'synctalk') {
      await canvas.click({ position: { x: box.width * 0.5, y: box.height * 0.45 } });
      await expect(lab.locator('[data-ray-selection]')).toContainText('Ray');
      await expect(lab.locator('[data-ray-readout]')).toContainText('Σ w');
    } else if (slug !== 'instag') {
      await canvas.click({ position: { x: box.width * 0.2, y: box.height * 0.7 } });
      await expect(lab.locator('[data-selection]')).not.toHaveText(before!);
      await expect(readout).toContainText('Σ w');
    }

    // Steps update the explanation.
    for (const stage of [0, 1, 2, 3]) {
      const button = lab.locator(`[data-stage="${stage}"]`);
      await button.click();
      await expect(button).toHaveAttribute('aria-pressed', 'true');
      await expect(lab.locator('[data-step-number]')).toHaveText(`Bước ${stage + 1} / 4`);
    }

    // Slider changes the view; reset restores defaults.
    const slider = lab.locator('input[name="position"]');
    const initial = await slider.inputValue();
    await slider.fill(slug === 'nerf' ? '9' : slug === 'gaussian-splatting' ? '60' : '30');
    await expect(lab.locator('[data-value="position"]')).not.toHaveText(initial);
    await lab.getByRole('button', { name: 'Đặt lại' }).click();
    await expect(slider).toHaveValue(initial);
    await expect(lab.locator('[data-step-number]')).toHaveText('Bước 4 / 4');

    // The table view exposes the raw numbers behind the chart (3DGS: reset returns to free orbit, which has no
    // per-pixel bake, so pick a test view again).
    if (select) { await lab.locator('[data-view]').selectOption('1'); await expect(readout).toContainText('Σ w'); }
    await lab.locator('.table-view summary').click();
    await expect(lab.locator('[data-table] tbody tr').first()).toBeVisible();

    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator('.tool-workspace').screenshot({ path: testInfo.outputPath(`${slug}-desktop.png`) });
    await page.setViewportSize({ width: 375, height: 812 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
    const canvasWidth = await canvas.evaluate(element => element.getBoundingClientRect().width);
    expect(canvasWidth).toBeGreaterThan(340);
    // Small text check: nothing in the lab renders below 12px.
    const smallest = await lab.evaluate(root => Math.min(...[...root.querySelectorAll('p, dt, dd, label, button, figcaption, h3, legend, summary, output')].filter(element => (element as HTMLElement).offsetParent).map(element => parseFloat(getComputedStyle(element).fontSize))));
    expect(smallest).toBeGreaterThanOrEqual(12);
    await page.locator('.tool-workspace').screenshot({ path: testInfo.outputPath(`${slug}-mobile.png`) });
    expect(errors).toEqual([]);
  });
}

for (const slug of ['synctalk', 'instag']) {
  test(`${slug}: reference video, render and audio share one frame clock`, async ({ page }) => {
    await page.goto(`/du-an/${slug}/`);
    const lab = page.locator('[data-radiance-mode]');
    await expect(lab).toHaveAttribute('aria-busy', 'false', { timeout: 60_000 });
    await expect.poll(() => lab.locator('[data-reference]').evaluate(media => (media as HTMLVideoElement).readyState)).toBeGreaterThanOrEqual(1);
    await lab.locator('input[name="position"]').fill('80');
    await expect.poll(() => lab.locator('[data-reference]').evaluate(media => (media as HTMLVideoElement).currentTime)).toBeCloseTo(3.22, 1);
    await expect.poll(() => lab.locator('[data-rendered]').evaluate(media => (media as HTMLVideoElement).currentTime)).toBeCloseTo(3.22, 1);
    await expect(lab.locator('[data-selection]')).toContainText('Frame 80');
    await lab.locator('[data-play]').click();
    await expect(lab.locator('[data-play]')).toHaveAttribute('aria-pressed', 'true');
    await expect.poll(async () => Number(await lab.locator('input[name="position"]').inputValue()), { timeout: 15_000 }).toBeGreaterThan(80);
    await lab.locator('[data-play]').click();
    await expect(lab.locator('[data-play]')).toHaveAttribute('aria-pressed', 'false');
    expect(await lab.locator('[data-audio]').evaluate(media => (media as HTMLAudioElement).muted)).toBe(true);
  });
}

test('NeRF keeps working when the orbit slider moves while ray traces are still downloading', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  // Slow every trace download so the slider always moves mid-fetch (the reported "reading 'columns'" race).
  await page.route(/traces-\d+\.[a-zA-Z0-9_-]+\.bin$/, async route => { await new Promise(resolve => setTimeout(resolve, 400)); await route.continue(); });
  await page.goto('/kien-thuc/nerf/');
  const lab = page.locator('[data-radiance-mode]');
  await expect(lab).toHaveAttribute('aria-busy', 'false', { timeout: 60_000 });
  // Record every error message the lab ever shows, even one that a later refresh hides again.
  await lab.locator('[data-error]').evaluate(element => {
    const seen: string[] = []; (window as unknown as { labErrors: string[] }).labErrors = seen;
    new MutationObserver(() => { if (!(element as HTMLElement).hidden && element.textContent) seen.push(element.textContent); }).observe(element, { attributes: true, childList: true, characterData: true, subtree: true });
  });
  const slider = lab.locator('input[name="position"]');
  for (const value of ['3', '9', '17', '22', '5', '30', '12']) { await slider.fill(value); await page.waitForTimeout(60); }
  await expect(lab.locator('[data-selection]')).toContainText('góc 120°', { timeout: 10_000 });
  await expect(lab.locator('[data-error]')).toBeHidden();
  expect(await page.evaluate(() => (window as unknown as { labErrors: string[] }).labErrors)).toEqual([]);
  await expect(lab.locator('[data-readout]')).toContainText('Σ w');
  expect(errors).toEqual([]);
});

test('3DGS pixel inspector reproduces the CUDA colour for the selected test view', async ({ page }) => {
  await page.goto('/kien-thuc/gaussian-splatting/');
  const lab = page.locator('[data-radiance-mode]');
  await expect(lab).toHaveAttribute('aria-busy', 'false', { timeout: 120_000 });
  await lab.locator('[data-view]').selectOption('2');
  await expect(lab.locator('[data-selection]')).toContainText('0229');
  const values = await lab.locator('[data-readout] dd').allTextContents();
  expect(values[0]).toBe(values[4]); // Màu pixel (recomputed) == Màu CUDA
});
