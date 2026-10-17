# AI4E Blog — Design Spec

Date: 2026-10-03 · Status: approved in brainstorming (sections 1–2 approved by user; section 3 derived from repo survey, user away — proceeding per their instruction to be proactive)

## 1. Intent

A Vietnamese-language blog for sharing AI knowledge, built around the 8 interactive visualization
repos (Polo Club of Data Science) in `/home/nhanht41/Documents/ai4e`. Each post explains a concept/project in
Vietnamese and embeds the real, self-hosted interactive viz. Users land on a blog home page and can
search for topics they want to explore.

**Success criteria**
- `npm run build` in `blog/` produces a static `dist/` that, served by any static server, shows:
  home page, 2 category pages, 8 posts, tag pages, search page; all 8 viz load and work inside posts.
- Consistent UI across all pages; light + dark mode; responsive down to 360px width.
- Search finds posts by Vietnamese text with or without diacritics, and by English synonyms.
- All UI copy and article prose in Vietnamese; viz UIs remain in original English.

**Non-goals (YAGNI)**: no deployment, no CMS, no comments, no auth, no analytics, no i18n switcher,
no modification of viz UIs beyond what is needed to build/serve them under a sub-path.

## 2. Categories

| Category (slug) | Label | Posts (viz slug) |
|---|---|---|
| `kien-thuc` | Kiến thức | `cnn-explainer`, `transformer-explainer`, `diffusion-explainer`, `ganlab` |
| `du-an` | Dự án | `dodrio`, `fairvis`, `viscuit`, `energyvis` |

## 3. Visual design

Reference: https://aioconquer.aivietnam.edu.vn/ (user's choice), plus light/dark mode.

- Font: **Inter** (Google Fonts, `vietnamese` subset), weights 400/500/600/700. Mono: JetBrains Mono for code.
- Brand palette (sky): 50 `#f0f9ff`, 100 `#e0f2fe`, 500 `#0ea5e9`, 600 `#0284c7`, 700 `#0369a1`, 900 `#0c4a6e`.
- Accent gradient for hero emphasis: yellow-300 `#fde047` → orange-300 `#fdba74`.
- Light: page bg `#f9fafb` (gray-50), surfaces white, text `#111827`/`#374151`, borders `#e5e7eb`.
- Dark: page bg `#0b1220`, surfaces `#111a2e`, text `#e5e7eb`/`#9ca3af`, borders `#1f2a44`; hero gradient `#0c4a6e → #082f49`.
- Theme tokens are CSS custom properties on `:root` and `:root[data-theme="dark"]`; initial theme from
  `localStorage` → `prefers-color-scheme`; inline head script sets `data-theme` before paint (no flash).
- Cards: white surface, `rounded-xl`, subtle shadow, hover lift (translateY(-2px) + shadow-lg), cover image 16:9.
- Icons: inline SVG (Lucide-style), no icon font.
- Viz frame in dark mode keeps a white background with rounded border (viz are light-only).

### Page layouts

**Header (all pages, sticky)**: logo + "AI4E" · nav: Trang chủ, Kiến thức, Dự án, Giới thiệu · pill search
input (opens command palette; shortcuts `/` and `Ctrl/⌘+K`) · theme toggle · mobile hamburger.

**Home `/`**: gradient hero ("Học AI bằng cách **nhìn thấy** nó hoạt động", subtitle, 2 CTA buttons, stats row:
8 trực quan hóa · 2 chuyên mục · N chủ đề · 100% mã nguồn mở) → "Kiến thức" card grid (4) → "Dự án" card grid (4)
→ topic chips with counts → footer.

**Category `/kien-thuc/`, `/du-an/`**: title + description, tag filter chips (client-side), card grid.

**Post `/kien-thuc/<slug>/`, `/du-an/<slug>/`**: breadcrumb · tags · reading time · title · description ·
`<VizEmbed>` (iframe, lazy, toolbar: Toàn màn hình / Mở tab mới / Tải lại) · two-column body with sticky TOC
(desktop) · callouts ("Thử ngay", "Lưu ý", "Ghi nhớ") · "Nguồn & giấy phép" box (paper, authors, venue, repo, license) ·
"Bài liên quan" (same tags).

**Tag `/chu-de/<tag-slug>/`**: all posts with that tag. **Tags index `/chu-de/`**.

**Search `/tim-kiem/?q=`**: input, category filter, result list with highlighted matches.

**About `/gioi-thieu/`**: purpose of the blog, credits to Polo Club & authors, licenses.

**404**: Vietnamese not-found page with search box.

## 4. Content model

Astro content collection `posts` (`src/content/posts/*.mdx`), zod-validated frontmatter:

```yaml
title: string            # Vietnamese
description: string      # ≤ 180 chars
category: kien-thuc | du-an
tags: string[]           # Vietnamese display names, e.g. ["CNN", "Thị giác máy tính"]
viz: string              # slug under /viz/ (optional for pure-text posts)
vizHeight: number        # px, default 820
cover: image             # src/assets/covers/<slug>.(png|jpg)
order: number            # display order within category
paper: { title, authors, venue, year, url }
source: { repo, url, license, demo }
date: date
```

Reading time computed from MDX body (≈200 words/min). Tag slug = diacritics-stripped kebab-case.

Components usable in MDX: `<VizEmbed>`, `<Callout type="try|note|key">`, `<Figure>`, `<KeyTerm vi en>`.

Article structure (each post, Vietnamese, ~1500–3000 words): mở đầu / vấn đề → khái niệm cốt lõi
(with formulas where relevant, KaTeX) → hướng dẫn khám phá viz từng bước (Callout "Thử ngay") → góc nhìn sâu hơn →
tóm tắt "Ghi nhớ" → nguồn tham khảo. Content must be technically accurate and grounded in the repo's
README/article text and paper.

## 5. Search

- Build-time endpoint `src/pages/search-index.json.ts` emits `[{id, url, title, description, category, tags, headings, body}]`
  (body = plain text stripped of MDX/JSX, truncated to ~4000 chars).
- Client: MiniSearch, fields title(boost 4) / tags(3) / headings(2) / description(2) / body(1); prefix + fuzzy 0.2.
- Normalization (`normalizeVi`): NFD → strip combining marks → `đ→d`, `Đ→D` → lowercase. Applied to both index
  and query via MiniSearch `processTerm`.
- Synonyms map (EN↔VI), expanded at query time: e.g. `attention↔chú ý`, `convolution↔tích chập`,
  `neural network↔mạng nơ-ron`, `bias↔thiên lệch`, `fairness↔công bằng`, `energy↔năng lượng`, `diffusion↔khuếch tán`,
  `generative↔sinh`, `transformer`, `gan↔mạng đối nghịch`.
- Command palette (header): top 6 results, keyboard navigation (↑↓ Enter Esc); Enter on query → `/tim-kiem/?q=`.
- Index loaded lazily on first focus.
- Unit-tested: normalization, synonym expansion, ranking of known queries.

## 6. Viz build & embedding pipeline

Source repos are never edited in place. `blog/scripts/build-viz.mjs`:
1. Copies each repo to `blog/.viz-build/<slug>/` (rsync, excluding `.git`, `node_modules`).
2. Applies sub-path patches from `blog/viz-patches/<slug>/` (documented, minimal) so the app works under `/viz/<slug>/`.
3. Installs deps & builds with per-repo recipe (see table; filled from survey).
4. Copies the runtime output to `blog/public/viz/<slug>/`.
5. Writes `blog/public/viz/manifest.json` with status per viz; failures don't abort other viz.
`npm run build:viz [slug...]` builds all or some; `npm run build` = astro build (expects viz already built).

Per-repo recipe table: see §6.1 (generated from repo survey).

**Embedding**: `<VizEmbed viz="cnn-explainer" height={820} title="…" />` renders an iframe
`src="/viz/<slug>/"`, `loading="lazy"`, skeleton placeholder until `load`, toolbar (fullscreen via Fullscreen API,
open in new tab, reload). If manifest marks viz missing, renders a Vietnamese fallback card linking to the original
live demo. Each post also offers a "Mở toàn màn hình" link to `/viz/<slug>/` directly.

## 7. Project structure

```
ai4e/
  <8 repos>                  # untouched sources
  blog/
    astro.config.mjs         # site, integrations: mdx, sitemap
    package.json
    scripts/build-viz.mjs
    viz-patches/<slug>/      # patch files + notes
    src/
      assets/covers/
      components/  (Header, Footer, ThemeToggle, SearchPalette, PostCard, TagChip, VizEmbed, Callout, Toc, …)
      content/posts/*.mdx
      content.config.ts
      layouts/ (BaseLayout, PostLayout)
      lib/ (search-normalize.ts, synonyms.ts, tags.ts, reading-time.ts)
      pages/ (index, kien-thuc/, du-an/, chu-de/, tim-kiem, gioi-thieu, 404, search-index.json.ts)
      styles/global.css      # tokens + base
    tests/                   # vitest unit tests; playwright smoke tests
```

## 8. Testing & verification

- Unit (vitest): `normalizeVi`, synonym expansion, tag slugging, reading time, search ranking fixtures.
- Build: `astro check` + `astro build` with zero errors.
- Smoke (Playwright, chromium): every page returns 200; each post's iframe loads its viz `index.html` and the viz's
  main element renders without uncaught console errors; theme toggle persists; search "mang no ron" returns CNN post;
  mobile viewport 375px has no horizontal scroll.
- Link check over `dist/` for internal broken links.

## 9. Risks

- Old toolchains (webpack4 / rollup / Polymer) on Node 22 → use `NODE_OPTIONS=--openssl-legacy-provider`, pin deps, or
  use `npx -p node@16` per recipe.
- Large assets (Dodrio ~2.2GB, Transformer/Diffusion ~640MB) → total dist several GB; acceptable for local use.
- External runtime fetches (CDNs, model files) — documented per viz in §6.1.
