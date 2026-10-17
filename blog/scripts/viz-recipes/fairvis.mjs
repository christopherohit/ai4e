// Build recipe for /viz/fairvis/ — field docs in ../viz-recipes.mjs.
/** @type {any} */
export default {
  slug: "fairvis",
  src: "tools/du-an/FairVis",
  exclude: [
    "models"
  ],
  replacements: [
    { file: "package.json", from: '"homepage": "https://poloclub.github.io/FairVis"', to: '"homepage": "/viz/fairvis/"' },
    { file: "src/components/App.js", from: "ReactGA.initialize('UA-50459890-3');\nReactGA.pageview(window.location.pathname + window.location.search);", to: "" }
  ],
  install: "npm ci --legacy-peer-deps --no-audit --no-fund",
  build: "npm run build",
  env: {
    NODE_EXTRA_CA_CERTS: "/etc/ssl/certs/ca-certificates.crt",
    NODE_OPTIONS: "--openssl-legacy-provider",
    PUBLIC_URL: "/viz/fairvis",
    CI: "false",
    GENERATE_SOURCEMAP: "false"
  },
  out: "build",
  verified: true,
  notes: "Verified 2026-10-03: CRA 3 + webpack 4; legacy peer resolution, system CA trust and OpenSSL compatibility on Node 22. PUBLIC_URL/homepage target /viz/fairvis/; analytics disabled in build copy. Full install/build passes; welcome and selected UCI Adult audit views render (feature histograms, accuracy/precision/recall, suggested groups). Both smoke and dataset-selection checks report zero failed same-origin requests and zero page/console errors. No backend required; external Google Fonts remain."
};
