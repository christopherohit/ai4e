import { defineConfig } from '@playwright/test';

// Smoke tests run against the built site (`npm run build` first).
const PORT = 4322;

export default defineConfig({
  testDir: 'tests/e2e',
  outputDir: 'test-results/f2',
  timeout: 120_000,
  fullyParallel: true,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    screenshot: 'only-on-failure',
    launchOptions: { executablePath: process.env.CHROME ?? '/usr/bin/google-chrome' },
  },
  webServer: {
    command: `npx astro preview --host 127.0.0.1 --port ${PORT} --ignore-lock`,
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
