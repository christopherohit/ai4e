// Build recipe for /viz/energyvis/ — field docs in ../viz-recipes.mjs.
/** @type {any} */
export default {
  slug: "energyvis",
  src: "tools/du-an/EnergyVis/frontend",
  exclude: ["debug.log"],
  install: "npm install --legacy-peer-deps --no-audit --no-fund",
  build: "mkdir -p public/vendor && curl -fsSL --retry 2 --max-time 60 https://d3js.org/d3.v4.min.js -o public/vendor/d3.v4.min.js && npm run build",
  env: {
    NODE_EXTRA_CA_CERTS: "/etc/ssl/certs/ca-certificates.crt",
    NODE_OPTIONS: "--openssl-legacy-provider",
    PUBLIC_URL: "/viz/energyvis",
    GENERATE_SOURCEMAP: "false"
  },
  replacements: [
    { file: "public/index.html", from: 'https://d3js.org/d3.v4.min.js', to: './vendor/d3.v4.min.js' },
    { file: "public/index.html", from: '<link rel="icon" href="%PUBLIC_URL%/favicon.ico" />', to: '' },
    { file: "public/index.html", from: '<link rel="apple-touch-icon" href="%PUBLIC_URL%/logo192.png" />', to: '' },
    { file: "public/index.html", from: '    <link\n    href="https//cdnjs.cloudflare.com/ajax/libs/KaTeX/0.9.0/katex.min.css"\n    rel="stylesheet"\n    />', to: '' },
    { file: "src/components/DataSourceView/DataSourceView.js", from: '<Button variant="raised"', to: '<Button' },
    // Map GeoJSON is fetched with a relative URL; anchor it to the sub-path.
    { file: "src/components/EnergyMap/USMap.js", from: 'd3.json("us-states.json"', to: 'd3.json(process.env.PUBLIC_URL + "/us-states.json"' }
  ],
  out: "build",
  verified: true,
  notes: "Verified full CRA/react-app-rewired install and build; 2.7MB output. PUBLIC_URL anchors scripts and map GeoJSON under /viz/energyvis/. D3 v4 is downloaded into the build copy and served locally to prevent an empty US map when its CDN is unavailable. Removed references to absent favicon/apple-touch icons, malformed external KaTeX stylesheet and obsolete Material UI raised variant in build copy only. Built-in Transformer/BERT profiles, charts and US map render; BERT selection tested with no failed local requests or page errors. One intermittent external resource 404 on initial smoke run, absent on repeat. Python backend is not deployed: live tracking requires a separately running CarbonTracker backend; demo profiles and counterfactual exploration work without it."
};
