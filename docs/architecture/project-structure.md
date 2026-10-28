# Project Structure

## Content

`blog/src/content/posts/<category>/<topic>/<slug>.mdx` is the only article source.
The loader derives the public ID from the filename, not its directory. URLs stay
`/<category>/<slug>/`. Each filename must be unique across the collection.
The schema validates that each topic belongs to the selected category.

| Category | Topic | Tools |
| --- | --- | --- |
| kien-thuc | nen-tang | gradient-descent-explainer, mini-batch-sgd-explainer, naive-bayes-explainer, svm-explainer, bias-variance-explainer, regularization-explainer, activation-explainer, pca-explainer |
| kien-thuc | thi-giac-may-tinh | cnn-explainer, nerf-explainer, gaussian-splatting-explainer |
| kien-thuc | xu-ly-ngon-ngu | transformer-explainer |
| kien-thuc | mo-hinh-sinh | diffusion-explainer, ganlab |
| kien-thuc | toi-uu-llm | inference-engineering-explainer, paged-attention-explainer, jev-explainer |
| du-an | giai-thich-mo-hinh | dodrio |
| du-an | danh-gia-thien-lech | FairVis, VisCUIT |
| du-an | nang-luong-ai | EnergyVis |
| du-an | nhan-vat-so | synctalk, instag |
| du-an | medlab | medlab-body-atlas |

Topics organize articles; tags link related concepts across these groups.

New knowledge tools always use the `-explainer` suffix in their directory name,
tool ID and article `viz` field. Project tools do not require this suffix.
Article filenames and URLs are independent of tool IDs. Existing legacy names
are not renamed automatically. These conventions also apply to future articles.

## Code

- `tools/<category>/<tool>/`: editable tool sources and their local build workflows.
- `tools/_shared/`: shared visualization UI, math, lifecycle and source datasets.
- `tools/native-tools.mjs`: registry of local Astro-native visualization entries.
- `blog/src/components/`: general UI, content callouts, cards and navigation.
- `blog/src/integrations/`: thin tool mounting adapters, including local Astro-native entries.
- `blog/src/integrations/runtime/`: mounting, localization and technical terms.
- `blog/scripts/viz-recipes/`: per-tool build and adaptation recipes.
- `blog/scripts/transformer-native/`: standalone Transformer entry and geometry adaptations.

## Generated Files

- `blog/.viz-build/`: patched build copies and cached dependencies.
- `blog/public/viz/`: standalone assets, model files and datasets.
- `blog/public/native/`: generated native adapters.
- `blog/dist/`: deployment output; never edit it directly.
- `blog/test-results/`: browser artifacts.

Do not move or delete these caches during a source-only reorganization. Rebuild
them from recipes when tool sources change. Live preview still uses `blog/dist`.
