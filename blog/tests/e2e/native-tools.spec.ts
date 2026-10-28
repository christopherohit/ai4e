import { test, expect } from '@playwright/test';

const tools = [
  ['kien-thuc', 'qwen3-8-flash-next', '.layer-map'],
  ['kien-thuc', 'gated-delta-net', '.heat'],
  ['kien-thuc', 'sparse-attention', '.attn'],
  ['kien-thuc', 'gradient-descent', '.glass-screen canvas'],
  ['kien-thuc', 'batch-mini-batch-sgd', '.glass-screen canvas'],
  ['kien-thuc', 'naive-bayes', '.glass-screen canvas'],
  ['kien-thuc', 'svm', '.glass-screen canvas'],
  ['kien-thuc', 'bias-variance', '.glass-screen canvas'],
  ['kien-thuc', 'l1-l2-regularization', '.glass-screen canvas'],
  ['kien-thuc', 'ham-kich-hoat', '.glass-screen canvas'],
  ['kien-thuc', 'pca', '.glass-screen canvas'],
  ['kien-thuc', 'jev', '.glass-screen canvas'],
  ['kien-thuc', 'paged-attention', '.glass-screen canvas'],
  ['kien-thuc', 'inference-engineering', '.glass-screen canvas'],
  ['kien-thuc', 'mixture-of-experts', '.glass-screen canvas'],
  ['kien-thuc', 'quantization', '.glass-screen canvas'],
  ['kien-thuc', 'p-value', '.glass-screen canvas'],
  ['kien-thuc', 'danh-gia-bo-phan-loai', '.glass-screen canvas'],
  ['du-an', 'synctalk-tung-buoc', '.glass-screen canvas'],
  ['du-an', 'instag-tung-buoc', '.glass-screen canvas'],
  ['du-an', 'sdvx-modeler', '.pipe'],
  ['du-an', 'ban-do-co-the-nguoi', '.ml-stage canvas[data-canvas]'],
  ['du-an', 'dodrio', '.graph-container svg'],
  ['du-an', 'fairvis', '#root button'],
  ['du-an', 'energyvis', '#energymap svg'],
  ['du-an', 'viscuit', '#selection .subgroup-entry-image-display img'],
  ['kien-thuc', 'diffusion-explainer', '#architecture-container img'],
  ['kien-thuc', 'ganlab', 'gan-lab #iteration-count'],
  ['kien-thuc', 'transformer-explainer', '.main-section.active .nodes'],
];
for (const [category, slug, selector] of tools) {
  test(`${slug} mounts directly without iframe`, async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    const failed: string[] = [];
    page.on('response', response => {
      if (response.url().startsWith('http://127.0.0.1:4321/') && response.status() >= 400) failed.push(`${response.status()} ${response.url()}`);
    });
    await page.goto(`/${category}/${slug}/`, { waitUntil: 'domcontentloaded' });
    await expect(page.locator('iframe')).toHaveCount(0);
    try {
      await expect(page.locator(`#native-tool-root ${selector}`).first()).toBeVisible({ timeout: 20000 });
    } finally {
      console.log(slug, JSON.stringify(errors));
      await page.screenshot({ path: `test-results/native-${slug}.png`, timeout: 10000 });
    }
    await expect(page.locator('#native-tool-root')).toHaveAttribute('aria-busy', 'false');
    await expect.poll(() => page.locator('#native-tool-root').evaluate(element => element.scrollHeight - element.clientHeight)).toBeLessThanOrEqual(2);
    if (slug === 'energyvis') {
      await page.locator('#native-tool-root').getByRole('button', { name: 'BERT', exact: true }).click();
      await expect(page.locator('#native-tool-root #energymap svg').first()).toBeVisible();
    }
    if (slug === 'fairvis') {
      await page.locator('#native-tool-root').getByRole('button', { name: 'Chọn', exact: true }).first().click();
      await expect(page.locator('#native-tool-root')).not.toContainText('Chọn bộ dữ liệu');
    }
    if (slug === 'ganlab') {
      const count = page.locator('#native-tool-root #iteration-count');
      const before = await count.textContent();
      await page.locator('#native-tool-root #play-pause-button').click();
      await expect(count).not.toHaveText(before!);
      await page.locator('#native-tool-root #play-pause-button').click();
    }
    if (slug === 'viscuit') {
      await page.locator('#native-tool-root #selection .subgroup-entry').first().dispatchEvent('click');
      await expect(page.locator('#native-tool-root #neurons')).not.toBeEmpty();
    }
    if (slug === 'diffusion-explainer') {
      await page.locator('#native-tool-root #controller-timestep-slider').fill('20');
      await expect(page.locator('#native-tool-root #generated-image')).toHaveAttribute('src', /_20\.jpg$/);
    }
    if (slug === 'transformer-explainer') {
      await page.setViewportSize({ width: 1920, height: 1080 });
      await expect.poll(() => page.locator('#native-tool-root').evaluate(element => element.scrollWidth - element.clientWidth)).toBeLessThanOrEqual(2);
      await expect(page.locator('#native-tool-root .transformer-toolbar')).toBeVisible();
      const mlpPath = page.locator('#native-tool-root g.mlpUp path').first();
      await mlpPath.dispatchEvent('click');
      const popup = page.locator('#native-tool-root .mlp-weight-popover[role="group"]');
      await expect(popup).toBeVisible();
      await expect.poll(() => popup.evaluate(element => getComputedStyle(element).opacity)).toBe('1');
      const canvasBounds = await page.locator('#native-tool-root .transformer-canvas').boundingBox();
      const popupBounds = await popup.boundingBox();
      expect(popupBounds!.y).toBeGreaterThanOrEqual(canvasBounds!.y);
      expect(popupBounds!.y + popupBounds!.height).toBeLessThanOrEqual(canvasBounds!.y + canvasBounds!.height);
      await mlpPath.dispatchEvent('click');
      await expect(popup).toHaveCount(0);
      const generate = page.locator('#native-tool-root').getByRole('button', { name: 'Sinh token', exact: true });
      await expect(generate).toBeEnabled({ timeout: 90000 });
      await generate.click();
      await expect(generate).toBeEnabled({ timeout: 60000 });
      const root = page.locator('#native-tool-root');
      const nextBlock = root.locator('[data-click="transformer-block-next-btn"]');
      for (let block = 2; block <= 12; block++) {
        await expect(nextBlock).toBeEnabled();
        await nextBlock.click();
        await expect(root.locator('.transformer-bounding-title .title-text')).toHaveText(`Transformer Block ${block}`);
        if (block < 12) await expect(nextBlock).toBeEnabled();
        await expect.poll(() => root.evaluate(element => element.scrollWidth - element.clientWidth)).toBeLessThanOrEqual(2);
        expect(await root.evaluate(element => element.scrollLeft)).toBe(0);
      }
      expect(await root.evaluate(element => getComputedStyle(element).overflowY)).toBe('hidden');
      const probabilities = root.locator('.softmax .content');
      const dimensions = await probabilities.evaluate(element => ({ height: element.clientHeight, scrollHeight: element.scrollHeight }));
      expect(dimensions.height).toBeLessThan(700);
      expect(dimensions.scrollHeight).toBeGreaterThan(dimensions.height);
      await probabilities.evaluate(element => { element.scrollTop = 100; });
      await page.waitForTimeout(500);
      expect(await probabilities.evaluate(element => element.scrollTop)).toBe(100);
      expect(await root.evaluate(element => element.scrollTop)).toBe(0);
    }
    await page.setViewportSize({ width: 375, height: 812 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
    expect(failed).toEqual([]);
    expect(errors).toEqual([]);
  });
}
