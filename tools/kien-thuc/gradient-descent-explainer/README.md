# gradient-descent-explainer

Native educational simulation of gradient descent: the update θ ← θ − η∇L on a 1D slice of a linear-regression MSE,
the full MSE over (w, b) and a non-convex 1D curve. Article content stays in blog/src/content/posts/.

- Tool.astro: entry imported by the blog adapter (lab chrome from tools/_shared/sequence-lab/, plots from tools/_shared/optimization-lab/).
- model.ts: the three toy losses and gradient descent on them, pure functions covered by blog/tests/unit/optimization-labs.test.ts.
- render.ts: this tool's view and the text for its four steps.

The dataset is synthetic (y = 1.5x + 0.8 + noise) and every gradient is computed exactly in the browser; no trained model is involved.
Run node scripts/workspace.mjs check, test or build from the workspace root.
