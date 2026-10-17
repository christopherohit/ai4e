# gated-delta-net-explainer

Native educational simulation of Gated DeltaNet: one 4×4 state matrix stepped through eight key→value tokens, compared with plain linear attention. Article content stays in blog/src/content/posts/.

- Tool.astro: entry imported by the blog adapter (lab chrome from tools/_shared/sequence-lab/).
- model.ts: the toy model and its numbers, pure functions covered by blog/tests/unit/sequence-labs.test.ts.
- render.ts: this tool's view and the text for its four steps.

All computation runs in the browser on small hand-built vectors; nothing is inferred from Qwen weights.
Run node scripts/workspace.mjs check, test or build from the workspace root.
