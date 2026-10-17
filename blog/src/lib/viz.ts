import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

/** Status entry written by scripts/build-viz.mjs into public/viz/manifest.json. */
export interface VizManifestEntry {
  ok: boolean;
  builtAt?: string;
  error?: string;
}

export type VizManifest = Record<string, VizManifestEntry>;

/** Public URL of a built viz. */
export function vizHref(slug: string): string {
  return `/viz/${slug}/`;
}

function readManifest(publicDir: string): VizManifest | undefined {
  const file = join(publicDir, 'viz', 'manifest.json');
  if (!existsSync(file)) return undefined;
  try {
    return JSON.parse(readFileSync(file, 'utf8')) as VizManifest;
  } catch {
    return undefined;
  }
}

/**
 * A viz is available when `public/viz/<slug>/index.html` exists at build time
 * and the manifest (if present) does not mark it as failed.
 * Only call from server-side code (component frontmatter, endpoints).
 */
export function isVizAvailable(slug: string, publicDir = join(process.cwd(), 'public')): boolean {
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) return false;
  if (!existsSync(join(publicDir, 'viz', slug, 'index.html'))) return false;
  const entry = readManifest(publicDir)?.[slug];
  return entry ? entry.ok !== false : true;
}
