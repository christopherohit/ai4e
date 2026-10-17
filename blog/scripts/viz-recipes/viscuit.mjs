// Build recipe for /viz/viscuit/ — field docs in ../viz-recipes.mjs.
/** @type {any} */
export default {
  slug: "viscuit",
  src: "tools/du-an/VisCUIT",
  replacements: [
    { file: "index.html", from: 'https://d3js.org/d3.v7.min.js', to: './vendor/d3.v7.min.js' }
  ],
  build: "mkdir -p vendor && curl -fsSL --retry 2 --max-time 60 https://d3js.org/d3.v7.min.js -o vendor/d3.v7.min.js",
  out: ".",
  outExclude: [
    "README.md",
    "LICENSE",
    "assets/logo/thumbnail.png"
  ],
  verified: true,
  notes: "Static ES modules, all paths relative (see .handoff/survey.json). D3 v7 is downloaded into the build copy and served locally to prevent d3-is-not-defined runtime failures when the CDN is unavailable. Fonts remain external."
};
