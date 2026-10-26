# svm-explainer

Native educational simulation of support vector machines: candidate separating lines, the hard-margin solution and its
support vectors, the soft margin controlled by C, and the RBF kernel on non-linear data. Article content stays in
blog/src/content/posts/.

- Tool.astro: entry imported by the blog adapter (lab chrome from tools/_shared/sequence-lab/, plots from tools/_shared/optimization-lab/).
- model.ts: the datasets and an SMO solver for the soft-margin dual with linear and RBF kernels, covered by blog/tests/unit/ml-basics-labs.test.ts.
- render.ts: this tool's view and the text for its four steps.

All data is synthetic and every model is trained in the browser.
Run node scripts/workspace.mjs check, test or build from the workspace root.
