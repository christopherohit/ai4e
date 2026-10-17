# AI4E

Blog va cong cu tuong tac, tach thanh hai nhom: Kien thuc va Du an.

## Cau truc

- `blog/`: website Astro, noi dung, UI va adapter native.
- `tools/kien-thuc/`: CNN Explainer, Transformer Explainer, Diffusion Explainer, GAN Lab.
- `tools/du-an/`: Dodrio, FairVis, VisCUIT, EnergyVis.
- `tools/kien-thuc/{nerf-explainer,gaussian-splatting-explainer}/`: visualize kien thuc moi.
- `tools/du-an/{synctalk,instag}/`: visualize du an talking-head.
- `tools/_shared/`: code va dataset dung chung cho visualize.
- `docs/architecture/`: kien truc va ban do chuyen muc.
- `docs/development/`: quy trinh build va kiem tra.
- `docs/deployment/`: preview va tunnel.
- `scripts/`: lenh quan ly workspace.

## Lenh chung

```sh
node scripts/workspace.mjs verify
node scripts/workspace.mjs check
node scripts/workspace.mjs test
node scripts/workspace.mjs build
node scripts/workspace.mjs preview
```

Xem `docs/architecture/project-structure.md` va `blog/README.md`.

## Quy uoc bai moi

Bai viet luon o `blog/src/content/posts/`. Visualize kien thuc o
`tools/kien-thuc/<ten>-explainer/`, visualize du an o `tools/du-an/<ten>/`.
Adapter trong blog chi noi tool vao trang; khong chua implementation visualize.
Ten tool khong lam thay doi URL bai viet. Quy uoc duoc ghi trong `AGENTS.md`
va kiem tra bang content schema, workspace verify va unit tests.
