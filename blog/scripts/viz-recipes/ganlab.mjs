// Build recipe for /viz/ganlab/ — field docs in ../viz-recipes.mjs.
/** @type {any} */
export default {
  slug: "ganlab",
  src: "tools/kien-thuc/ganlab",
  install: "npx -y yarn@1 install --frozen-lockfile --ignore-engines --ignore-scripts --non-interactive --network-timeout 600000",
  env: { NODE_EXTRA_CA_CERTS: "/etc/ssl/certs/ca-certificates.crt" },
  replacements: [
    { file: "demo/ganlab.html", from: 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.4.0/Chart.bundle.min.js', to: '../node_modules/chart.js/dist/Chart.bundle.min.js' },
    { file: "tsconfig.json", from: '"declaration": false,', to: '"declaration": false, "typeRoots": ["./node_modules/@types"], "types": ["polymer", "webcomponents.js"],' },
    { file: "tsconfig-dev.json", from: '"**/*"', to: '"demo/**/*.ts", "lib/**/*.d.ts"' },
    { file: "tsconfig-dev.json", from: '"declaration": false,', to: '"declaration": false, "typeRoots": ["./node_modules/@types"], "types": ["polymer", "webcomponents.js"],' },
  ],
  build: "node_modules/.bin/browserify demo/ganlab.ts -p [ tsify -p tsconfig-dev.json ] -o demo/bundle.js && node_modules/.bin/polymer-bundler --inline-scripts --inline-css --out-html demo/index.bundled.html demo/index.html && mkdir -p output && cp -a demo/. output/ && mv output/index.bundled.html output/index.html",
  out: "output",
  outExclude: ["*.ts", "ganlab.html"],
  verified: true,
  notes: "Verified 2026-10-03: TypeScript 2.7 + Browserify + Polymer 1; bundle demo code and inline Polymer imports/scripts/styles into output/index.html. Chart.js is inlined from the locked npm dependency instead of a runtime CDN request, preventing Chart-is-not-defined initialization failures. Restrict compiler inputs and ambient types to this build copy plus lib/chartjs.d.ts to avoid ancestor Astro types. Full install/build succeeds, main graph/distributions/metrics render, zero failed same-origin requests. External dependencies: Google Fonts, Material Design Lite, dialog-polyfill, YouTube; three non-blocking external HTTP errors during the original smoke check."
};
