# Radiance labs: trained models, baked data, web viewers

Four native labs show models that were trained on the workspace GPU (2 × RTX 5090) and then exported for the browser:
NeRF and 3D Gaussian Splatting on the drone scene (kiến thức), SyncTalk and InsTaG on Obama (dự án).

Article sources remain in blog/src/content/posts. Each tool keeps its view in tools/<category>/<slug>/render.ts
(how its baked data is drawn and inspected, plus the text of its four steps). Shared pieces live in
tools/_shared/radiance: Lab.astro (markup), lab.css, runtime.ts (controls, pointer, clip clock), lab-core.ts (asset
lookup, view contract, ray inspector), trained.ts (binary readers), charts.ts (D3 chart + table view),
splat-view.ts (Spark WebGL viewer, lazy-loaded), talking-view.ts (shared SyncTalk/InsTaG view), math.ts (formulas used in
the article tests). LocalTool.astro remains a thin adapter.

## What is shown, and how honest it is

Nothing on these pages is a simulation. Every pixel, σ, α, T, w and Δμ was produced by a forward pass of the trained
model on the GPU and stored; the browser only displays and indexes it.

| Lab | Model / code | Training | Held-out result | Browser shows |
| --- | --- | --- | --- | --- |
| NeRF | nerfstudio nerfacto 1.1.5 | 30k steps, 175 drone images | 25 views: PSNR 22.48, SSIM 0.682, LPIPS 0.261 | 36-frame orbit (RGB + depth), 24×15 rays/frame with every proposal sample |
| 3DGS | graphdeco-inria/gaussian-splatting | 30k it., densify grad threshold 0.0006 | 25 views: PSNR 22.48 (SH3), 1,251,840 Gaussians | live WebGL (Spark 2.3.1) + per-pixel contributor lists for 4 test views, 16×10 pixels each |
| SyncTalk | ZiqiaoPeng/SyncTalk, AVE + OpenFace AU45 | 60k + 100k lip fine-tune | 728 frames: PSNR 35.49, LPIPS 0.0178, LMD 2.73 | 10 s test clip vs ground truth, 25 head renders + depth, 18×18 rays/frame |
| InsTaG | Fictionarry/InsTaG, AVE | pretrain 5 identities (30k face + 30k mouth), adapt with 250 frames (10 s) | 727 frames: PSNR 33.63, LPIPS 0.0266, LMD 2.93 | 10 s test clip vs ground truth, 25 fused renders, face/mouth alpha, per-Gaussian Δμ projected through the real test camera |

Cross-checks baked into the data and asserted by tests:

- 3DGS: the per-pixel contributor list recomputes the colour with the CUDA forward rules (EWA + 0.3 low-pass, α ≤ 0.99,
  skip α < 1/255, stop at T < 1e-4) and matches the CUDA rasterizer within 6e-4 (meta.json maxAbsDiffVsCuda).
- NeRF: Σw + T_end = 1 on every ray (float16 storage keeps each term to ~1e-3).
- SyncTalk: traces use 96 uniform samples inside aabb_infer; their colour matches the CUDA occupancy-grid marcher within
  mean abs 0.0019.

The full 3DGS model (6.12 M Gaussians, 1.5 GB, PSNR 22.85) does not fit a web page. Pruning it without retraining
collapses quality (2 M → 18.0 dB, 1 M → 15.4 dB) and a hard 1 M cap during densification gave 15.8 dB, so the web model
was retrained with a higher densification threshold instead. The page states the numbers of the model it actually shows.

## Data and publication

Approved by the workspace owner (see the AI4E media publication memory):

- 2026-10-06: EmoTag excerpts (no longer used by these pages; removed).
- 2026-10-08: Instag-data/Obama, LIMITED: a 10 s test-split excerpt (source frames 7272–7521, audio 290.88–300.88 s),
  the matching model renders, 25 baked frames per project. No full video, no training images.
- 2026-10-08: drone 3DGS web asset 33 MB SPZ with SH3, lazy-loaded with progress.

Baked data lives in tools/_shared/radiance/data/trained (~60 MB, 32 MB of it the SPZ) and is bundled by Vite as hashed
assets. Formats are documented in trained.ts and in each meta.json:

- traces-*.bin: int32 rays, int32 fields, int32 offsets[rays+1], float16 samples × (t, σ, α, T, w, r, g, b). Each ray
  keeps the contiguous window carrying 99.9% of its weight.
- gaussian/pixels-*.json: depth-sorted top-64 contributors (depth, opacity, α, T, w, rgb) plus n and the CUDA colour.
- instag/dxyz-*.f16: Δμ per Gaussian (float16 × 3); proj-*.f16: (x0, y0, x1, y1) of μ and μ+Δμ in test-camera pixels.
- clip/signals.json: OpenFace AU25/AU45/AU01 (raw 0–5 scale) and audio RMS (16 kHz, 640 samples per frame).

## Reproducing

Training runs outside the repository in ~/work/ai4e-train; the datasets on /shared stay read-only. The scripts used
are copied in scripts/radiance-bake. Environment: Python 3.12, torch 2.11 + CUDA 12.8 (sm_120). InsTaG and SyncTalk
venvs reuse that torch through a .pth file; install their extra packages with --no-deps, otherwise uv pulls a CUDA 13
torch. Patches needed for torch 2.11 / sm_120 (none change the algorithms):

- gaussian-splatting: np.byte → np.uint8 in the Blender reader; optional AI4E_MAX_GAUSSIANS densify cap (unused in
  the final model).
- InsTaG/SyncTalk CUDA encoders: -std=c++17, torch/torch.h → torch/extension.h, missing `import torch` in setup.py.
- InsTaG: torch.load(..., weights_only=False) for its own checkpoints; guard the TensorBoard call in train_fuse_con.
- SyncTalk: AdamW betas (0, 0.99) → (0.0, 0.99). Face masks regenerated from the dataset's parsing maps by
  make_face_masks.py (face skin + mouth, no hair/neck; verified by overlay).

Order: convert_drone.py → 3DGS train.py (… --densify_grad_threshold 0.0006) → ns-train nerfacto (camera optimizer
off, filename split every 8th frame) → run_synctalk.sh → run_instag_pretrain.sh + run_instag_obama.sh → bake_*.py →
pack_traces.py → Spark transcodeSpz (maxSh 3, version 3).

## Browser behaviour

- 3DGS renders live with WebGL2; on machines without a GPU (software GL) one frame takes seconds, which is why its e2e
  tests have long timeouts. The camera select snaps to the four inspected test poses.
- The other labs draw baked images on a 2D canvas; the selection snaps to the nearest baked ray and the inspector shows
  its real samples. Charts have direct labels and a table view with the raw numbers.
- SyncTalk/InsTaG: the reference video is the clock; the render clip, audio (muted by default) and frame slider follow it.
- Lifecycle: one AbortController per lab, ResizeObserver, Spark and WebGL disposed on astro:before-swap / pagehide.

## Validation

Unit tests (blog/tests/unit/radiance-trained.test.ts, radiance-math.test.ts) check the binary readers, energy
conservation on real NeRF rays, SyncTalk coverage, 3DGS CUDA agreement and InsTaG projection bounds. Browser tests
(blog/tests/e2e/radiance-labs.spec.ts) cover loading, pixel inspection, steps, reset, table view, the shared clip
clock, CUDA colour agreement, mobile width (no horizontal scroll, canvas ≥ 340 px, no text under 12 px). Build before
browser tests against the static preview.

## Article covers

scripts/radiance-bake/make_covers.py builds the four 1600×900 WebP covers in blog/src/assets/covers from real model
outputs only: NeRF orbit frame with its depth (diagonal split); 3DGS held-out view rendered by the CUDA rasterizer at
scaling_modifier 0.2 and 1.0 (render_shrunk.py); SyncTalk head as 90×90 baked rays (stroke = samples along the ray,
brightness = weight); InsTaG Δμ field projected through the test camera. No face texture appears on covers (user
decision 2026-10-09), because covers are also listing thumbnails and og:image.
