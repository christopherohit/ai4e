# qwen3-8-flash-next-explainer

Native educational simulation of Qwen3.8-Flash-Next: the 48-layer schedule, long-context memory, MoE routing and gated residual. Article content stays in blog/src/content/posts/.

- Tool.astro: entry imported by the blog adapter (lab chrome from tools/_shared/sequence-lab/).
- model.ts: the toy model and its numbers, pure functions covered by blog/tests/unit/sequence-labs.test.ts.
- render.ts: this tool's view and the text for its four steps.

All computation runs in the browser on small hand-built vectors; nothing is inferred from Qwen weights.
Run node scripts/workspace.mjs check, test or build from the workspace root.
