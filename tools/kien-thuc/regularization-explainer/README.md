# regularization-explainer

Native educational simulation comparing L1 (Lasso) and L2 (Ridge) regularization: constraint geometry on a 2D loss,
shrinkage vs soft-thresholding, coefficient paths and behaviour with correlated features. Article content stays in
blog/src/content/posts/.

- Tool.astro: entry imported by the blog adapter (lab chrome from tools/_shared/sequence-lab/, plots from tools/_shared/optimization-lab/).
- model.ts: the 2D problem, the 8-feature regression, closed-form Ridge and coordinate-descent Lasso, covered by blog/tests/unit/ml-basics-labs.test.ts.
- render.ts: this tool's view and the text for its four steps.

All data is synthetic; every fit runs in the browser.
Run node scripts/workspace.mjs check, test or build from the workspace root.
