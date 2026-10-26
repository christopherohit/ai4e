# naive-bayes-explainer

Native educational simulation of a Multinomial Naive Bayes spam filter: priors, Laplace-smoothed word likelihoods,
log-score classification and the double counting caused by the independence assumption. Article content stays in
blog/src/content/posts/.

- Tool.astro: entry imported by the blog adapter (lab chrome from tools/_shared/sequence-lab/, styles from tools/_shared/optimization-lab/).
- model.ts: the 16 training emails, counts, smoothing and classification, covered by blog/tests/unit/ml-basics-labs.test.ts.
- render.ts: this tool's view and the text for its four steps.

All emails are hand-written examples; no real mailbox data is used.
Run node scripts/workspace.mjs check, test or build from the workspace root.
