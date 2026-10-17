// Pure helpers for scripts/build-viz.mjs (unit-tested in tests/unit/build-viz.test.ts).
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

/** CLI: `build-viz.mjs [slug...] [--skip-install] [--list]`. */
export function parseArgs(argv) {
  const slugs = argv.filter((a) => !a.startsWith('--'));
  return {
    slugs,
    skipInstall: argv.includes('--skip-install'),
    list: argv.includes('--list'),
  };
}

/**
 * Recipes to build: all when `slugs` is empty, else the named ones (unknown slug throws).
 * @template {{ slug: string }} R
 * @param {Record<string, R>} recipes
 * @param {string[]} slugs
 * @returns {R[]}
 */
export function selectRecipes(recipes, slugs) {
  if (slugs.length === 0) return Object.values(recipes);
  return slugs.map((slug) => {
    const recipe = recipes[slug];
    if (!recipe) throw new Error(`Unknown viz slug "${slug}". Known: ${Object.keys(recipes).join(', ')}`);
    return recipe;
  });
}

/**
 * Apply string replacements `{ file, from, to, all? }` to files under `root`.
 * Throws if `from` is not found, so a drifted source fails loudly instead of silently shipping broken paths.
 */
export function applyReplacements(root, replacements = []) {
  for (const { file, from, to, all } of replacements) {
    const path = join(root, file);
    const text = readFileSync(path, 'utf8');
    if (!text.includes(from)) throw new Error(`Replacement target not found in ${file}: ${JSON.stringify(from)}`);
    writeFileSync(path, all ? text.split(from).join(to) : text.replace(from, to));
  }
}

/** Return a new manifest with `slug` set to `entry`, keeping the other entries. */
export function mergeManifest(prev, slug, entry) {
  return { ...prev, [slug]: entry };
}
