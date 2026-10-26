# mini-batch-sgd-explainer

Native educational simulation of batch gradient descent, mini-batch gradient descent and stochastic gradient descent
on the same linear-regression MSE. Article content stays in blog/src/content/posts/.

- Tool.astro: entry imported by the blog adapter (lab chrome from tools/_shared/sequence-lab/, plots from tools/_shared/optimization-lab/).
- model.ts: the three variants, gradient-direction metrics and run comparison, covered by blog/tests/unit/optimization-labs.test.ts.
- render.ts: this tool's view and the text for its four steps.

The dataset is synthetic and shuffles are seeded; every gradient is computed exactly in the browser. Learning rate is
constant and there is no momentum, so the lab isolates the effect of batch size.
Run node scripts/workspace.mjs check, test or build from the workspace root.
