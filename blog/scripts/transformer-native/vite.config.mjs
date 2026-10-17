import { defineConfig } from '../../.viz-build/transformer-explainer/node_modules/vite/dist/node/index.js';
import { svelte, vitePreprocess } from '../../.viz-build/transformer-explainer/node_modules/@sveltejs/vite-plugin-svelte/src/index.js';
import { resolve } from 'node:path';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import tailwindConfig from '../../.viz-build/transformer-explainer/tailwind.config.js';
const work = resolve('.viz-build/transformer-explainer');
const require = createRequire(resolve(work, 'package.json'));
const tailwind = require('tailwindcss');
const exports = JSON.parse(readFileSync(resolve(work, 'node_modules/svelte/package.json'))).exports;
const svelteAliases = Object.entries(exports).filter(([key]) => key !== './package.json').map(([key, value]) => ({
  find: key === '.' ? /^svelte$/ : new RegExp(`^svelte/${key.slice(2)}$`),
  replacement: resolve(work, 'node_modules/svelte', value.browser ?? value.default ?? './src/index-client.js'),
}));
export default defineConfig({
  root: work,
  base: '/native/transformer-explainer/standalone/',
  plugins: [{
    name: 'native-sankey-coordinates',
    enforce: 'pre',
    transform(source, id) {
      if (id.endsWith('/components/BlockTransition.svelte')) {
        return source
          .replace('const scrollLeft = window.scrollX;', 'const scrollLeft = 0;')
          .replace("const topbarHeight = document.querySelector('.top-bar')?.offsetHeight;", 'const topbarHeight = 0;')
          .replace('embedding.getBoundingClientRect()', 'nativeBlockRect(embedding)')
          .replace('block.getBoundingClientRect()', 'nativeBlockRect(block)')
          .replace('</script>', `
            function nativeBlockRect(element) {
              const bounds = element.getBoundingClientRect();
              const origin = document.querySelector('#native-tool-root .nodes').getBoundingClientRect();
              return new DOMRect(bounds.left - origin.left, bounds.top - origin.top, bounds.width, bounds.height);
            }
          </script>`);
      }
      if (!id.endsWith('/components/Sankey.svelte')) return;
      const helper = `
      function nativeRect(element) {
        const bounds = element.getBoundingClientRect();
        const origin = svgEl.getBoundingClientRect();
        return new DOMRect(bounds.left - origin.left, bounds.top - origin.top, bounds.width, bounds.height);
      }
      `;
      return source.replace(/(\w+)\?\.getBoundingClientRect\(\)/g, '($1 ? nativeRect($1) : undefined)')
        .replace(/targets\[i\]\?\.getBoundingClientRect\(\)/g, '(targets[i] ? nativeRect(targets[i]) : undefined)')
        .replace(/(start|ends\[i\])\.getBoundingClientRect\(\)/g, 'nativeRect($1)')
        .replaceAll('const scrollTop = window.scrollY;', 'const scrollTop = 0;')
        .replaceAll('const scrollLeft = window.scrollX;', 'const scrollLeft = 0;')
        .replace('</script>', helper + '</script>');
    },
  }, svelte({ configFile: false, preprocess: vitePreprocess() })],
  resolve: { alias: [
    { find: '~', replacement: resolve(work, 'src') },
    { find: '$app/paths', replacement: resolve('scripts/transformer-native/paths.js') },
    { find: '$app/stores', replacement: resolve('scripts/transformer-native/stores.js') },
    ...svelteAliases,
  ] },
  css: {
    postcss: { plugins: [tailwind({ ...tailwindConfig, content: [`${work}/src/**/*.{html,js,svelte,ts}`, `${work}/node_modules/flowbite-svelte/**/*.{html,js,svelte,ts}`] })] },
    preprocessorOptions: { scss: { loadPaths: [work], additionalData: `@import '${work}/src/styles/variables.scss';` } },
  },
  build: {
    outDir: resolve('public/native/transformer-explainer/standalone'),
    emptyOutDir: false,
    rollupOptions: { input: resolve('scripts/transformer-native/main.js'), output: { entryFileNames: 'entry.js', assetFileNames: '[name][extname]' } },
  },
});
