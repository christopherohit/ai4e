// Build recipe for /viz/cnn-explainer/ — field docs in ../viz-recipes.mjs.
/** @type {any} */
export default {
  slug: "cnn-explainer",
  src: "tools/kien-thuc/cnn-explainer",
  exclude: ["tiny-vgg", "README.md", "deploy-gh-page.sh"],
  replacements: [
    // App code uses the PUBLIC_URL token (rollup replace) for all asset paths; make it relative.
    { file: "rollup.config.js", from: "PUBLIC_URL: production ? '/cnn-explainer' : ''", to: "PUBLIC_URL: '/viz/cnn-explainer'" },
    { file: "src/main.js", from: "target: document.body", to: "target: document.getElementById('cnn-tool-root') || document.body" },
    { file: "src/main.js", from: "props: {}", to: "props: { integrated: Boolean(document.getElementById('cnn-tool-root')) }" },
    { file: "src/App.svelte", from: "<script>", to: "<script>\n  export let integrated = false;" },
    { file: "src/App.svelte", from: "  <recommender-overlay", to: "  {#if !integrated}\n  <recommender-overlay" },
    { file: "src/App.svelte", from: "  <Header />", to: "  <Header />\n  {/if}" },
    { file: "src/App.svelte", from: "<Explainer />", to: "<Explainer {integrated} />" },
    { file: "src/Explainer.svelte", from: "<script>", to: "<script>\n  export let integrated = false;" },
    { file: "src/Explainer.svelte", from: "<Overview />", to: "<Overview {integrated} />" },
    { file: "src/overview/Overview.svelte", from: "<script>", to: "<script>\n  export let integrated = false;" },
    { file: "src/overview/Overview.svelte", from: "<Article/>", to: "{#if !integrated}<Article/>{/if}" },
    { file: "src/article/Youtube.svelte", from: '<script context="module">', to: '<script context="module">\n  const embedded = window.parent !== window || Boolean(document.getElementById("cnn-tool-root"));' },
    { file: "src/article/Youtube.svelte", from: 'var tag = document.createElement("script");', to: 'if (!embedded) {\n  var tag = document.createElement("script");' },
    { file: "src/article/Youtube.svelte", from: 'window.dispatchEvent(new Event("iframeApiReady"));', to: 'window.dispatchEvent(new Event("iframeApiReady"));\n  }' },
    { file: "src/article/Youtube.svelte", from: 'player.seekTo(startSecond);', to: 'if (!player) {\n      window.open(`https://www.youtube.com/watch?v=${videoId}&t=${startSecond}s`, "_blank", "noopener,noreferrer");\n      return;\n    }\n    player.seekTo(startSecond);' },
    { file: "src/article/Youtube.svelte", from: '<div id={playerId} />', to: '{#if embedded}<a href={`https://www.youtube.com/watch?v=${videoId}`} target="_blank" rel="noopener noreferrer">Watch the tutorial on YouTube</a>{:else}<div id={playerId} />{/if}' },
    // Root-absolute links in the HTML shell.
    { file: "public/index.html", from: 'href="/', to: 'href="./', all: true },
    { file: "public/index.html", from: 'src="/bundle.js"', to: 'src="./bundle.js"' },
    { file: "public/index.html", from: "URL('/assets/font/", to: "URL('./assets/font/" },
    { file: "public/index.html", from: 'https://d3js.org/d3.v5.js', to: '/vendor/cnn/d3.v5.js' },
    { file: "public/index.html", from: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf.min.js', to: '/vendor/cnn/tf.min.js' },
    { file: "public/index.html", from: 'https://use.fontawesome.com/releases/v5.3.1/js/all.js', to: '/vendor/cnn/fontawesome.js' },
    { file: "public/index.html", from: 'https://cdn.jsdelivr.net/gh/cferdinandi/smooth-scroll@15/dist/smooth-scroll.polyfills.min.js', to: '/vendor/cnn/smooth-scroll.min.js' },
    { file: "public/index.html", from: 'https://cdn.jsdelivr.net/npm/bulma@0.8.0/css/bulma.min.css', to: '/vendor/cnn/bulma.min.css' },
    { file: "public/index.html", from: '<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>', to: '' },
    { file: "public/index.html", from: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js', to: './vendor/mathjax-tex-svg.js' },
    { file: "public/index.html", from: '    <script\n      type="module"\n      src="https://unpkg.com/recommender-overlay/dist/recommender-overlay.es.js"\n    ></script>', to: '' },
  ],
  install: "npm install --no-audit --no-fund",
  env: { NODE_EXTRA_CA_CERTS: "/etc/ssl/certs/ca-certificates.crt" },
  build: "mkdir -p public/vendor && curl -fsSL --retry 2 --max-time 60 https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js -o public/vendor/mathjax-tex-svg.js && npm run build",
  out: "public",
  verified: true,
  notes: "Verified 2026-10-05: Svelte 3 + rollup 1; assets under /viz/cnn-explainer/. D3, TensorFlow.js, Bulma, Font Awesome and smooth-scroll use the blog's local vendor assets; pinned MathJax is downloaded into the build copy. Removed obsolete polyfill and recommender startup scripts. Embedded/native lessons link to the optional YouTube video without loading a third-party player; standalone video playback is preserved with a timestamp-link fallback. System CA bundle required for npm behind this environment's proxy. Optional Google Fonts remain external."
};
