#!/usr/bin/env node
// Build the interactive visualizations into public/viz/<slug>/ (spec §6).
//   node scripts/build-viz.mjs                 build all
//   node scripts/build-viz.mjs dodrio ganlab   build some
//   --skip-install                             reuse node_modules in .viz-build/<slug>/
//   --list                                     print recipes and manifest status
// Source repos are never modified: each is copied to .viz-build/<slug>/ first.
import { execSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import recipes from './viz-recipes.mjs';
import { applyReplacements, mergeManifest, parseArgs, selectRecipes } from './viz-lib.mjs';

const BLOG = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const ROOT = resolve(BLOG, '..');
const BUILD_DIR = join(BLOG, '.viz-build');
const PUBLIC_VIZ = join(BLOG, 'public', 'viz');
const MANIFEST = join(PUBLIC_VIZ, 'manifest.json');

const q = (s) => `'${String(s).replaceAll("'", `'\\''`)}'`;

function run(cmd, cwd, env = {}) {
  console.log(`  $ ${cmd}`);
  execSync(cmd, { cwd, stdio: 'inherit', env: { ...process.env, ...env } });
}

function readManifest() {
  try {
    return JSON.parse(readFileSync(MANIFEST, 'utf8'));
  } catch {
    return {};
  }
}

function writeManifest(manifest) {
  mkdirSync(PUBLIC_VIZ, { recursive: true });
  writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2) + '\n');
}

function buildOne(recipe, { skipInstall }) {
  const src = join(ROOT, recipe.src);
  if (!existsSync(src)) throw new Error(`Source dir not found: ${src}`);
  const work = join(BUILD_DIR, recipe.slug);
  const target = join(PUBLIC_VIZ, recipe.slug);
  mkdirSync(work, { recursive: true });

  // 1. Fresh copy of the sources (keeps node_modules so re-builds are fast).
  const excludes = ['.git', 'node_modules', ...(recipe.exclude ?? [])].map((e) => `--exclude ${q(e)}`).join(' ');
  run(`rsync -a --delete ${excludes} ${q(src + '/')} ${q(work + '/')}`, BLOG);

  // 2. Sub-path patches.
  const patchDir = join(BLOG, 'viz-patches', recipe.slug);
  if (recipe.patches && existsSync(patchDir)) {
    for (const p of readdirSync(patchDir).filter((f) => f.endsWith('.patch')).sort()) {
      run(`git apply --whitespace=nowarn ${q(join(patchDir, p))}`, work);
    }
  }
  applyReplacements(work, recipe.replacements);

  // 3. Install + build.
  const env = recipe.env ?? {};
  if (recipe.install && !(skipInstall && existsSync(join(work, 'node_modules')))) run(recipe.install, work, env);
  if (recipe.build) run(recipe.build, work, env);

  // 4. Copy runtime output.
  const out = join(work, recipe.out);
  if (!existsSync(join(out, 'index.html'))) throw new Error(`Build output has no index.html: ${out}`);
  mkdirSync(target, { recursive: true });
  const outExcludes = ['node_modules', ...(recipe.outExclude ?? [])].map((e) => `--exclude ${q(e)}`).join(' ');
  run(`rsync -a --delete --delete-excluded ${outExcludes} ${q(out + '/')} ${q(target + '/')}`, BLOG);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.list) {
    const manifest = readManifest();
    for (const r of Object.values(recipes)) {
      const m = manifest[r.slug];
      const status = m ? (m.ok ? `ok ${m.builtAt}` : `FAILED: ${m.error}`) : 'not built';
      console.log(`${r.slug.padEnd(24)} ${r.src.padEnd(22)} ${status}`);
    }
    return;
  }

  const selected = selectRecipes(recipes, args.slugs);
  const failures = [];
  for (const recipe of selected) {
    console.log(`\n▶ ${recipe.slug} (${recipe.src})`);
    let entry;
    try {
      buildOne(recipe, args);
      entry = { ok: true, builtAt: new Date().toISOString() };
      console.log(`✔ ${recipe.slug} → public/viz/${recipe.slug}/`);
    } catch (err) {
      entry = { ok: false, builtAt: new Date().toISOString(), error: String(err.message ?? err).slice(0, 500) };
      failures.push(recipe.slug);
      console.error(`✘ ${recipe.slug}: ${entry.error}`);
    }
    // Re-read so concurrent runs for different slugs don't clobber each other.
    writeManifest(mergeManifest(readManifest(), recipe.slug, entry));
  }

  console.log(`\nDone: ${selected.length - failures.length}/${selected.length} ok${failures.length ? `; failed: ${failures.join(', ')}` : ''}`);
  process.exitCode = failures.length ? 1 : 0;
}

main();
