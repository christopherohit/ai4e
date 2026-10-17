# sdvx-modeler

Native educational simulation of SDVX Modeler: one scripted agent turn on a five-file toy ARXML project —
request, MCP tool chain, transaction checks, diff against baseline and undo — plus a loop-guard chart.
Article content stays in blog/src/content/posts/du-an/agent-ky-thuat/sdvx-modeler.mdx.

- Tool.astro: entry imported by the blog adapter (lab chrome from tools/_shared/sequence-lab/).
- model.ts: toy project, scenarios, line diff and dangling-reference check; covered by blog/tests/unit/sequence-labs.test.ts.
- render.ts / sdvx.css: this tool's view.

Scenarios mirror the real system's shape (tool names, transaction status, baseline diff) on invented file contents.
No model, project or customer data is used. Incident figures come from the SDVX Modeler docs and source.
