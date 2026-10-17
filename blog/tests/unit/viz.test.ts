import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { isVizAvailable, vizHref } from '../../src/lib/viz';

let pub: string;

beforeAll(() => {
  pub = mkdtempSync(join(tmpdir(), 'ai4e-viz-'));
  for (const slug of ['cnn-explainer', 'ganlab']) {
    mkdirSync(join(pub, 'viz', slug), { recursive: true });
    writeFileSync(join(pub, 'viz', slug, 'index.html'), '<!doctype html>');
  }
  writeFileSync(join(pub, 'viz', 'manifest.json'), JSON.stringify({ ganlab: { ok: false, error: 'build failed' } }));
});

afterAll(() => rmSync(pub, { recursive: true, force: true }));

describe('vizHref', () => {
  it('points to /viz/<slug>/', () => expect(vizHref('dodrio')).toBe('/viz/dodrio/'));
});

describe('isVizAvailable', () => {
  it('is true when index.html exists and manifest does not mark it failed', () => {
    expect(isVizAvailable('cnn-explainer', pub)).toBe(true);
  });
  it('is false when the manifest marks the build as failed', () => {
    expect(isVizAvailable('ganlab', pub)).toBe(false);
  });
  it('is false when the viz was never built', () => {
    expect(isVizAvailable('dodrio', pub)).toBe(false);
  });
  it('rejects non-slug input (no path traversal)', () => {
    expect(isVizAvailable('../secret', pub)).toBe(false);
  });
});
