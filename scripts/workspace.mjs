import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import recipes from '../blog/scripts/viz-recipes.mjs';
import { nativeTools, validToolName } from '../tools/native-tools.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const command = process.argv[2] ?? 'verify';
if (command === 'verify') {
  for (const recipe of Object.values(recipes)) {
    if (recipe.src.startsWith('tools/kien-thuc/') && !validToolName('kien-thuc', recipe.slug)) throw new Error(`Knowledge tool requires -explainer: ${recipe.slug}`);
    if (!existsSync(join(root, recipe.src))) throw new Error(`Missing tool source: ${recipe.src}`);
    console.log(`${recipe.slug}: ${recipe.src}`);
  }
  for (const [slug, tool] of Object.entries(nativeTools)) {
    if (!existsSync(join(root, tool.source, 'Tool.astro'))) throw new Error(`Missing native tool entry: ${tool.source}`);
    if (tool.category === 'kien-thuc' && !slug.endsWith('-explainer')) throw new Error(`Knowledge tool requires -explainer: ${slug}`);
    console.log(`${slug}: ${tool.source}`);
  }
} else {
  const commands = {
    check: ['run', 'check'], test: ['test'], build: ['run', 'build'],
    'build:tools': ['run', 'build:viz'],
    preview: ['run', 'preview', '--', '--host', '127.0.0.1', '--port', '4321'],
  };
  if (!Object.hasOwn(commands, command)) throw new Error(`Unknown command: ${command}`);
  const result = spawnSync('npm', [...commands[command], ...process.argv.slice(3)], { cwd: join(root, 'blog'), stdio: 'inherit' });
  if (result.error) throw result.error;
  process.exitCode = result.status ?? 1;
}
