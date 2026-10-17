# instag

Native visualization of a model trained on the workspace's own GPU; article content stays in blog/src/content/posts/.

- Tool.astro: entry imported by the blog adapter.
- render.ts: this tool's view (how the baked data is drawn and inspected) and the text for its four steps.
- Shared lab chrome, data readers, charts and baked data: tools/_shared/radiance/.

Every number on the page comes from the trained model, precomputed on the GPU (training and baking are documented
in docs/development/radiance-labs.md). Run node scripts/workspace.mjs check, test or build from the workspace root.
