// Build recipe for /viz/dodrio/ — field docs in ../viz-recipes.mjs.
/** @type {any} */
export default {
  slug: "dodrio",
  src: "tools/du-an/dodrio",
  exclude: [
    "data-generation"
  ],
  replacements: [
    // node-sass 7 does not build on Node 22; svelte-preprocess falls back to `sass` (dart-sass).
    { file: "package.json", from: '    "node-sass": "^7.0.0",\n', to: "" },
    // Asset/data URLs in src/ are written as 'PUBLIC_URL/...' and replaced at build time.
    { file: "rollup.config.js", from: "PUBLIC_URL: production ? '/dodrio' : ''", to: "PUBLIC_URL: production ? '/viz/dodrio' : ''" },
    // index.html uses root-absolute asset paths; make them relative to /viz/dodrio/.
    { file: "public/index.html", from: "href='/", to: "href='./", all: true },
    { file: "public/index.html", from: "src='/build/", to: "src='./build/" },
    { file: "public/index.html", from: "https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.1/css/bulma.min.css", to: "./vendor/bulma.min.css" },
    { file: "public/index.html", from: "https://use.fontawesome.com/releases/v5.3.1/js/all.js", to: "/vendor/cnn/fontawesome.js" },
    { file: "src/App.svelte", from: '  <script\n    type="module"\n    src="https://unpkg.com/recommender-overlay/dist/recommender-overlay.es.js"\n  ></script>', to: "" },
    { file: "src/article/Youtube.svelte", from: "<script context='module'>", to: "<script context='module'>\n  const embedded = window.parent !== window;" },
    { file: "src/article/Youtube.svelte", from: "var tag = document.createElement('script');", to: "if (!embedded) {\n  var tag = document.createElement('script');" },
    { file: "src/article/Youtube.svelte", from: "window.dispatchEvent(new Event('iframeApiReady'));", to: "window.dispatchEvent(new Event('iframeApiReady'));\n  }" },
    { file: "src/article/Youtube.svelte", from: "player.seekTo(startSecond);", to: 'if (!player) {\n      window.open(`https://www.youtube.com/watch?v=${videoId}&t=${startSecond}s`, "_blank", "noopener,noreferrer");\n      return;\n    }\n    player.seekTo(startSecond);' },
    { file: "src/article/Youtube.svelte", from: "<div id={playerId} />", to: '{#if embedded}<a href={`https://www.youtube.com/watch?v=${videoId}`} target="_blank" rel="noopener noreferrer">Watch the tutorial on YouTube</a>{:else}<div id={playerId} />{/if}' },
  ],
  install: "npm install --no-audit --no-fund",
  env: { NODE_EXTRA_CA_CERTS: "/etc/ssl/certs/ca-certificates.crt" },
  build: "mkdir -p public/vendor && curl -fsSL --retry 2 --max-time 60 https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.1/css/bulma.min.css -o public/vendor/bulma.min.css && npm run build",
  out: "public",
  verified: true,
  notes: "Svelte 3 + rollup 2; remove incompatible node-sass in build copy and use existing Dart Sass; patch PUBLIC_URL and shell assets for /viz/dodrio/. Bulma and Font Awesome are served locally. Removed the recommender startup script; embedded lessons link to the optional YouTube video without loading its third-party player, while standalone playback and timestamp links are preserved. Approximately 2.2GB of precomputed model data. Optional Google Fonts and the paper preview remain external."
};
