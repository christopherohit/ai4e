# bias-variance-explainer

Native educational simulation of the bias–variance tradeoff: polynomial regression of increasing degree on noisy samples
of sin 2πx, repeated over many training sets to measure bias², variance and the expected test error. Article content
stays in blog/src/content/posts/.

- Tool.astro: entry imported by the blog adapter (lab chrome from tools/_shared/sequence-lab/, plots from tools/_shared/optimization-lab/).
- model.ts: data generation, Legendre-basis least squares and the decomposition, covered by blog/tests/unit/ml-basics-labs.test.ts.
- render.ts: this tool's view and the text for its four steps.

All data is synthetic; every fit runs in the browser.
Run node scripts/workspace.mjs check, test or build from the workspace root.
