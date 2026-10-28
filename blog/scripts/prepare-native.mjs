import { readdirSync, readFileSync, mkdirSync, writeFileSync, existsSync, copyFileSync } from 'node:fs';
import { join, dirname, relative, resolve } from 'node:path';
import { parse, serialize } from 'parse5';

const slugs = ['dodrio', 'fairvis', 'energyvis', 'viscuit', 'ganlab', 'transformer-explainer', 'diffusion-explainer'];
const publicDir = resolve('public');
// --optional: skip tools whose external build is absent (CI and fresh checkouts); their pages show the tool fallback.
const optional = process.argv.includes('--optional');

function files(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return files(path);
    return /\.(js|css|wasm)$/.test(entry.name) ? [path] : [];
  });
}

export function rewriteAssets(source, slug) {
  return source.replace(/(["'`])(?:\.\/)?(assets\/|icons\/|images\/|figures\/|pretrained_models\/)/g, `$1/viz/${slug}/$2`);
}

function css(source, url) {
  return source.replace(/url\(\s*(['"]?)([^)'"\s]+)\1\s*\)/g, (match, _quote, value) => {
    if (/^(data:|https?:|\/|#)/.test(value)) return match;
    return `url("${new URL(value, url).pathname}")`;
  }).replace(/(^|[},])\s*(html|body|:root)(?=[\s,{.:#])/g, '$1 :scope');
}

for (const slug of slugs) {
  const sourceDir = join(publicDir, 'viz', slug);
  const outputDir = join(publicDir, 'native', slug);
  mkdirSync(outputDir, { recursive: true });
  const built = slug === 'transformer-explainer' ? existsSync(join(outputDir, 'standalone/entry.js')) : existsSync(sourceDir);
  if (!built && optional) { console.warn(`Native adapter skipped (not built): ${slug}`); continue; }
  if (slug === 'transformer-explainer') {
    if (!built) throw new Error('Run npm run build:native-transformer first.');
    const stylesheet = readFileSync(join(outputDir, 'standalone/main.css'), 'utf8');
    writeFileSync(join(outputDir, 'adapter.json'), JSON.stringify({ html: '', scripts: [{ src: '/native/transformer-explainer/standalone/entry.js', type: 'module' }], css: css(stylesheet, new URL('https://local/native/transformer-explainer/standalone/main.css')) }));
    console.log(`Native adapter prepared: ${slug}`);
    continue;
  }
  for (const file of files(sourceDir)) {
    const path = relative(sourceDir, file);
    const destination = join(outputDir, path);
    mkdirSync(dirname(destination), { recursive: true });
    if (file.endsWith('.wasm')) { copyFileSync(file, destination); continue; }
    if (file.endsWith('.css')) {
      writeFileSync(destination, `@scope (#native-tool-root) { ${css(readFileSync(file, 'utf8'), new URL(`https://local/viz/${slug}/${path}`))} }`);
      continue;
    }
    let source = rewriteAssets(readFileSync(file, 'utf8'), slug);
    if (slug === 'dodrio' && path === 'build/bundle.js') {
      source = source.replace(/target:document\.body/g, 'target:document.querySelector("#native-tool-root .native-content")');
      source = source.replace(/window\.parent!==window/g, '(window.parent!==window||Boolean(document.getElementById("native-tool-root")))');
      source = source.replace(/\.select\(document\.body\)/g, '.select(document.getElementById("native-tool-root"))');
    }
    writeFileSync(destination, source);
  }
  const document = parse(readFileSync(join(sourceDir, 'index.html'), 'utf8'));
  const scripts = [];
  const styles = [];
  function visit(node) {
    const children = [];
    for (const child of node.childNodes ?? []) {
      const attributes = Object.fromEntries((child.attrs ?? []).map(attr => [attr.name, attr.value]));
      if (child.tagName === 'script') {
        let source = rewriteAssets(serialize(child), slug);
        if (/youtube|gtag|googletagmanager|recommender-overlay/i.test(attributes.src ?? '') || source.includes('onYouTubeIframeAPIReady')) continue;
        if (slug === 'ganlab' && source.includes('inializePolymerPage')) continue;
        source = source.replace(/import\((["'])(\.\/[^"']+)\1\)/g, (match, quote, path) => `import(${quote}/native/${slug}/${path.slice(2)}${quote})`);
        if (slug === 'transformer-explainer') {
          source = source.replace('new URL(".", location).pathname.slice(0, -1)', '"/viz/transformer-explainer"');
          source = source.replace('document.currentScript.parentElement', 'document.getElementById("native-tool-root")');
          source = source.replace('kit.start(app, element);', 'app.hooks.reroute = () => "/viz/transformer-explainer/"; kit.start(app, element);');
        }
        let src = attributes.src;
        if (src?.includes('kit.fontawesome.com')) src = '/vendor/cnn/fontawesome.js';
        if (slug === 'ganlab' && src?.includes('material.min.js')) src = '/vendor/native/material.min.js';
        if (slug === 'ganlab' && src?.includes('dialog-polyfill')) src = '/vendor/native/dialog-polyfill.min.js';
        if (src) {
          const url = new URL(src, `https://local/viz/${slug}/`);
          if (url.origin === 'https://local') {
            const nativePath = url.pathname.replace(`/viz/${slug}/`, `/native/${slug}/`);
            src = existsSync(join(publicDir, nativePath)) ? nativePath : url.pathname;
          }
        }
        scripts.push({ src, source, type: attributes.type ?? '' });
        continue;
      }
      if (child.tagName === 'link' && attributes.rel === 'stylesheet') {
        if (slug === 'ganlab' && attributes.href.includes('material.min.css')) styles.push(readFileSync(join(publicDir, 'vendor/native/material.min.css'), 'utf8'));
        if (slug === 'ganlab' && attributes.href.includes('dialog-polyfill')) styles.push(readFileSync(join(publicDir, 'vendor/native/dialog-polyfill.min.css'), 'utf8'));
        const url = new URL(attributes.href, `https://local/viz/${slug}/`);
        if (url.origin === 'https://local') styles.push(css(readFileSync(join(publicDir, url.pathname), 'utf8'), url));
        continue;
      }
      if (child.tagName === 'style' && node.tagName !== 'template') {
        styles.push(css(serialize(child), new URL(`https://local/viz/${slug}/index.html`)));
        continue;
      }
      if (child.tagName === 'iframe' || child.tagName === 'noscript') continue;
      for (const attr of child.attrs ?? []) {
        if (['src', 'poster'].includes(attr.name) || (attr.name === 'href' && !attr.value.startsWith('#'))) {
          attr.value = new URL(attr.value, `https://local/viz/${slug}/`).href.replace('https://local', '');
        }
      }
      visit(child);
      if (child.content) visit(child.content);
      children.push(child);
    }
    node.childNodes = children;
  }
  visit(document);
  const html = document.childNodes.find(node => node.tagName === 'html');
  const body = html.childNodes.find(node => node.tagName === 'body');
  if (slug === 'transformer-explainer') {
    for (const file of files(sourceDir).filter(file => file.endsWith('.css'))) {
      styles.push(css(readFileSync(file, 'utf8'), new URL(`https://local/viz/${slug}/${relative(sourceDir, file)}`)));
    }
  }
  if (slug === 'ganlab') scripts.push({ src: '/native/ganlab/bundle.js', type: '' });
  if (slug === 'ganlab') styles.push('@font-face { font-family: "Material Icons"; src: url("/vendor/native/MaterialIcons-Regular.woff2") format("woff2"); } .material-icons { font-family: "Material Icons"; font-weight: normal; font-style: normal; font-size: 24px; line-height: 1; letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap; word-wrap: normal; direction: ltr; font-feature-settings: "liga"; -webkit-font-smoothing: antialiased; }');
  let bodyHtml = serialize(body);
  if (slug === 'ganlab') bodyHtml = bodyHtml.slice(0, bodyHtml.indexOf('</gan-lab>') + '</gan-lab>'.length);
  writeFileSync(join(outputDir, 'adapter.json'), JSON.stringify({ html: bodyHtml, scripts, css: styles.join('\n') }));
  console.log(`Native adapter prepared: ${slug}`);
}
