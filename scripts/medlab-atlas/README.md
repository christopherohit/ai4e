# MedLab atlas build

Regenerates `tools/du-an/medlab-body-atlas/data/` (atlas.json, core.bin, extra.bin).

```sh
# sources (≈130 MB, keep them outside the repo, e.g. in .cache/)
git clone https://github.com/ashemag/human-atlas.git .cache/medlab-atlas/human-atlas
git -C .cache/medlab-atlas/human-atlas checkout 1c38bf3
mkdir -p .cache/medlab-atlas/female
for f in $(git -C .cache/medlab-atlas/human-atlas ls-tree --name-only d72b4f6 public/models/ | grep female | grep -v '\.gz$'); do
  git -C .cache/medlab-atlas/human-atlas show d72b4f6:$f > .cache/medlab-atlas/female/$(basename $f)
done

cd scripts/medlab-atlas && npm install
node build.mjs ../../.cache/medlab-atlas/human-atlas/public/models ../../.cache/medlab-atlas/female
```

What it does:

1. Keeps every BodyParts3D mesh except the body surface (skin, hair); reclassifies a few source groups by name (brain
   ventricles, liver "hepatovenous" segments, papillary muscles).
2. Welds and simplifies each mesh with meshoptimizer (share kept per system in `KEEP`, error limit 2%).
3. Adds HRA female pelvic organs and breasts (female only) and HRA lung segments (both), each mapped onto a BodyParts3D
   landmark box.
4. Assigns regions from part centres, resolves organs from FMA concepts, quantises and writes the two files.

Output at the last run: 2,312 parts, 564,772 triangles, core 3.97 MB + extra 1.74 MB (≈4.3 MB gzip).
