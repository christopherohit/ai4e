# MedLab · Bản đồ cơ thể người

Dark 3D anatomy atlas for the article `blog/src/content/posts/du-an/medlab/ban-do-co-the-nguoi.mdx`.
Astro-native, no runtime dependencies (plain WebGL2).

| File | Role |
| --- | --- |
| `Tool.astro` | markup: header, sidebar (search, organ cards, regions/systems/organs tabs), stage, detail panel, attribution dialog |
| `render.ts` | controller: loading, per-part display state, selection, search, views, Nam/Nữ, exploded view, labels, tour, quiz |
| `gl.ts` | renderer: one buffer per data file, per-part state in two textures, ghost (x-ray) pass, id picking, thumbnails |
| `atlas.ts` | data model and pure helpers shared with the unit tests (decoding, female proportion warp, explode, search) |
| `organs.ts` | Vietnamese texts: systems, regions, 30 organs, guided tour, quiz |
| `atlas.css` | styles; narrow layouts via container queries on `.medlab-workspace` |
| `data/` | `atlas.json` (manifest), `core.bin`, `extra.bin` (muscles and vessels, loaded after the first view) |

## Data

Built by `scripts/medlab-atlas/build.mjs` (see its README) from:

- BodyParts3D 4.0, © The Database Center for Life Science, CC BY 4.0, as packaged by
  [ashemag/human-atlas](https://github.com/ashemag/human-atlas) (MIT) at commit `1c38bf3`: 2,229 male reference meshes
  (skin and hair dropped).
- Human Reference Atlas, *3D Reference Organ Set for Female v1.5* (Browne, Schlehlein; HuBMAP), CC BY 4.0, as packaged
  by the same repository at commit `d72b4f6`: female reproductive organs, bladder and breasts (female atlas only) and
  bronchopulmonary segment surfaces (both atlases, because BodyParts3D has no lung surface).

HRA parts are mapped axis-wise onto BodyParts3D landmarks (bony pelvis, bronchial tree, thoracic spine); the maps are
stored under `alignment` in the manifest. The female atlas reuses the male skeleton, muscles, vessels and shared organs
with a smooth proportion warp (`FEMALE_*` in `atlas.ts`, mirrored in the vertex shader): 93% stature, narrower shoulders
and chest, wider pelvis. These factors are teaching approximations, and the article and the attribution dialog say so.
No female urethra is modelled.

Vertex format: int16 × 3 position over the manifest box + int8 × 2 octahedral normal (8 bytes); uint16 indices per part.
Organ texts are Vietnamese with Latin terms; individual structure names stay in the original English with FMA IDs.
TA2 IDs are not shown: the TA2 viewer and Wikidata were not reachable when the data was built.
