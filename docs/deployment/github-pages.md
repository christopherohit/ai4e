# GitHub Pages

The site deploys to `https://christopherohit.github.io/ai4e/` from `.github/workflows/deploy.yml` on every push to
`main` (or manually: Actions → Deploy to GitHub Pages → Run workflow).

One-time setup: repository Settings → Pages → Build and deployment → Source: **GitHub Actions**.

How the sub-path works:

- `actions/configure-pages` provides `SITE_URL` and `BASE_PATH` (`/ai4e`); `astro.config.mjs` uses them as `site`/`base`.
- `npm run build:site` = `prepare-native.mjs --optional` (skips external tools that are not built in CI) → `astro build`
  → `scripts/rebase-dist.mjs`, which prefixes hand-written root-relative links (`href="/kien-thuc/…"`), the search index
  and external tool adapters with the base. Client code builds URLs with `withBase()` from `src/lib/base.ts`.
- Without `BASE_PATH` the same commands produce a site for `/` (local preview, custom domain).

External source tools (CNN/Transformer/Diffusion Explainer, GAN Lab, Dodrio, FairVis, EnergyVis, VisCUIT) need their
data and `npm run build:viz` locally; in CI their pages render the tool's fallback. Native and glass labs, radiance labs
and all articles build from the repository alone.

`.github/workflows/ci.yml` runs on other branches and pull requests: tool registry check, `astro check`, unit tests and
`build:site`.
