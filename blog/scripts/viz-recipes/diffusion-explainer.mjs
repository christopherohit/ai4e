// Build recipe for /viz/diffusion-explainer/ — field docs in ../viz-recipes.mjs.
/** @type {any} */
export default {
  slug: "diffusion-explainer",
  src: "tools/kien-thuc/diffusion-explainer",
  exclude: [
    "python"
  ],
  out: ".",
  outExclude: [
    "README.md",
    "LICENSE"
  ],
  replacements: [
    { file: "index.html", from: 'https://d3js.org/d3.v7.min.js', to: './vendor/d3.v7.min.js' },
    {
      // Drop Google Analytics: the blog has no analytics (spec non-goal).
      file: "index.html",
      from:
        "    <!-- Google tag (gtag.js) -->\n" +
        "    <script async src=\"https://www.googletagmanager.com/gtag/js?id=G-V5ZFD2995C\"></script>\n" +
        "    <script>\n" +
        "        window.dataLayer = window.dataLayer || [];\n" +
        "        function gtag(){dataLayer.push(arguments);}\n" +
        "        gtag('js', new Date());\n" +
        "        \n" +
        "        gtag('config', 'G-V5ZFD2995C');\n" +
        "    </script>\n",
      to: ""
    }
  ],
  build: "mkdir -p vendor && curl -fsSL --retry 2 --max-time 60 https://d3js.org/d3.v7.min.js -o vendor/d3.v7.min.js",
  verified: true,
  notes:
    "Static HTML + ES modules; python/ (data generation) excluded. D3 v7 is downloaded into the build copy " +
    "and served locally to prevent an empty architecture when its CDN is unavailable. All asset paths are relative " +
    "(./assets, ./icons), so it works under /viz/diffusion-explainer/. The Google " +
    "Analytics gtag snippet is removed from index.html. Remaining external deps: Font " +
    "Awesome kit (kit.fontawesome.com), Google Fonts (Roboto, Roboto Condensed). Output ~640MB (precomputed latent " +
    "images under assets/)."
};
