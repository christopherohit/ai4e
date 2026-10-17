# Content and visualization conventions

- Article sources live only in `blog/src/content/posts/<category>/<topic>/<article-slug>.mdx`.
- Visualization implementation lives in `tools/`, not in blog integration components.
- Every new knowledge visualization uses `tools/kien-thuc/<tool-slug>-explainer/`; its tool ID and article `viz` field end in `-explainer`.
- Project visualizations use `tools/du-an/<project-slug>/`; they do not require an `-explainer` suffix.
- Article URL slugs are independent of tool IDs. Preserve existing article URLs during source reorganizations.
- Blog integration components are thin mounting adapters; tool-specific rendering, controls and source datasets belong to tools. Shared visualization code may live in `tools/_shared/`.
- Register Astro-native tools in `tools/native-tools.mjs` and wire their entries in `blog/src/integrations/LocalTool.astro`. External-source tools use the existing build recipes.
- Existing legacy tool names are not renamed automatically. Apply the naming convention to new tools and explicitly requested migrations.
- Educational simulations must be distinguished from pretrained model inference. Do not publish identifiable dataset media without an explicit decision.
