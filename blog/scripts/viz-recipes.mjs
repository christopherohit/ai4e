// Per-viz build recipes for scripts/build-viz.mjs (spec §6).
//
// Fields:
//   slug          URL slug: served at /viz/<slug>/
//   src           source dir relative to the ai4e root (never edited in place)
//   exclude       extra rsync excludes (beyond .git, node_modules)
//   replacements  [{ file, from, to, all? }] string patches applied to the build copy
//   patches       true → apply blog/viz-patches/<slug>/*.patch with `git apply`
//   install       shell command run in the build copy (skipped with --skip-install)
//   build         shell command run in the build copy
//   env           extra env vars for install/build
//   out           dir (relative to the build copy) whose contents go to public/viz/<slug>/
//   outExclude    rsync excludes when copying `out` to public/viz/<slug>/
//   notes         what works / what doesn't under /viz/<slug>/
//
// One file per viz in viz-recipes/<slug>.mjs (so Phase D tasks can edit them independently).
// Entries marked `verified: false` are unverified guesses.

import cnnExplainer from './viz-recipes/cnn-explainer.mjs';
import transformerExplainer from './viz-recipes/transformer-explainer.mjs';
import diffusionExplainer from './viz-recipes/diffusion-explainer.mjs';
import ganlab from './viz-recipes/ganlab.mjs';
import dodrio from './viz-recipes/dodrio.mjs';
import fairvis from './viz-recipes/fairvis.mjs';
import viscuit from './viz-recipes/viscuit.mjs';
import energyvis from './viz-recipes/energyvis.mjs';

/** @type {Record<string, any>} */
const recipes = Object.fromEntries(
  [cnnExplainer, transformerExplainer, diffusionExplainer, ganlab, dodrio, fairvis, viscuit, energyvis].map((r) => [r.slug, r]),
);

export default recipes;
