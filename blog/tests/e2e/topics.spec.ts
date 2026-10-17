import { expect, test } from '@playwright/test';

test('knowledge and project pages group articles by topic without changing URLs', async ({ page }) => {
  await page.goto('/kien-thuc/');
  await expect(page.locator('#thi-giac-may-tinh a[href="/kien-thuc/cnn-explainer/"]')).toBeVisible();
  await expect(page.locator('#mo-hinh-sinh a[href="/kien-thuc/ganlab/"]')).toBeVisible();
  await page.getByRole('button', { name: /^CNN / }).click();
  await expect(page.locator('#mo-hinh-sinh')).toBeHidden();
  await expect(page.locator('#thi-giac-may-tinh')).toBeVisible();
  await page.goto('/du-an/');
  await expect(page.locator('#danh-gia-thien-lech .category-item')).toHaveCount(2);
  await expect(page.locator('#nang-luong-ai a[href="/du-an/energyvis/"]')).toBeVisible();
  await page.goto('/kien-thuc/cnn-explainer/');
  await expect(page.locator('.hybrid-header a[href="/kien-thuc/#thi-giac-may-tinh"]')).toBeVisible();
  await expect(page.locator('.source-box')).toHaveCount(0);
});
