# AI4E — Blog trực quan hóa AI bằng tiếng Việt

Blog tĩnh (Astro 7 + MDX) giới thiệu 8 công cụ trực quan hóa AI của
[Polo Club of Data Science](https://poloclub.github.io/) (Georgia Tech). Mỗi bài viết
bằng tiếng Việt và tích hợp bản viz tương tác, được build từ mã nguồn gốc vào
`public/viz/<slug>/`. Giao diện của các viz giữ nguyên tiếng Anh.

- **Kiến thức** (`/kien-thuc/`): CNN Explainer, Transformer Explainer, Diffusion Explainer, GAN Lab
- **Dự án** (`/du-an/`): Dodrio, FairVis, VisCUIT, EnergyVis

## Yêu cầu

- Node.js ≥ 22.12.0, npm ≥ 9.6.5; `git` và `rsync` để build viz.
- Các repo nguồn nằm trong `../tools/kien-thuc/` và `../tools/du-an/`. Chỉ cần khi build lại viz.
- Google Chrome tại `/usr/bin/google-chrome` cho test e2e và `check-viz`.
  Nếu dùng trình duyệt khác, đặt `CHROME=/đường/dẫn/tới/tệp/thực/thi` khi chạy lệnh.
  `npx playwright install chromium` chỉ tải Chromium; vẫn cần đặt `CHROME` trỏ tới
  tệp thực thi đã tải vì cấu hình hiện tại không tự chọn trình duyệt này.

## Lệnh thường dùng

Chạy các lệnh từ thư mục `blog/` (nếu đang ở thư mục gốc `ai4e/`, chạy `cd blog`).
Lần đầu, cài phụ thuộc → build viz → build blog → preview theo đúng thứ tự:

```bash
npm install            # cài phụ thuộc
npm run build:viz      # build cả 8 viz vào public/viz/ (lâu, cần mạng)
npm run build          # build site tĩnh vào dist/
npm run preview -- --host 127.0.0.1 --port 4323 --ignore-lock
                       # xem dist/ tại http://127.0.0.1:4323/
```

Khi soạn bài hoặc chỉnh giao diện, dùng `npm run dev` thay cho build + preview;
máy chủ phát triển mặc định ở `http://localhost:4321/`.

`npm run build` chỉ build Astro; nó **không** chạy `build:viz`. Nếu `public/viz/`
đã có đủ các bản build, không cần build lại viz khi chỉ sửa bài viết.
Preview dùng cổng 4323 để tách khỏi dev (4321) và Playwright (4322).
Không dừng tiến trình đang dùng cổng 4399/4400; nếu cổng preview bận, chọn cổng khác.

### Công cụ CNN tích hợp

Route `src/pages/[category]/[slug].astro` hiện chọn `InteractivePostLayout.astro`
khi frontmatter có `viz: cnn-explainer`: công cụ CNN chạy trực tiếp trong DOM,
kèm nội dung tiếng Việt và mục lục, không dùng iframe. Bài CNN nằm tại
`/kien-thuc/cnn-explainer/`; `/cong-cu/cnn-explainer/` cũng hiển thị bài học này,
không phải trang chuyển hướng. Các viz khác cũng dùng layout hybrid và mount
trực tiếp vào `#native-tool-root`; không có iframe trên 8 trang bài học.
`InteractivePostLayout.astro` là khung dùng lại cho các bài học tương tác;
`CnnTool.astro` là adapter CNN, `cnn-locale.ts` Việt hóa nhãn động và tooltip.
Nhấp tên lớp trong mạng để cuộn tới phần giải thích tiếng Việt tương ứng.
Recipe CNN mount ứng dụng vào `#cnn-tool-root`, bỏ header và recommender riêng
khi chạy trong blog; bản standalone `/viz/cnn-explainer/` vẫn dùng được.
CSS của công cụ được cô lập trong `@scope`; mô hình và ảnh tải từ `/viz/cnn-explainer/`.
D3, TensorFlow.js, SmoothScroll và Font Awesome được phục vụ từ `public/vendor/cnn/`.
Phần hướng dẫn tiếng Anh gốc không mount trong bản tích hợp; bài tiếng Việt dùng KaTeX của blog.
Sau khi thay đổi recipe, chạy `node scripts/build-viz.mjs cnn-explainer --skip-install`
rồi `npm run build`.

### Build viz

```bash
node scripts/build-viz.mjs                  # build tất cả
npm run build:viz -- dodrio ganlab          # build một vài viz
node scripts/build-viz.mjs --skip-install   # dùng lại node_modules trong .viz-build/<slug>/
node scripts/build-viz.mjs --list           # liệt kê recipe và trạng thái
node scripts/check-viz.mjs dodrio           # mở viz trong Chrome, báo request lỗi + chụp màn hình
```

Repo nguồn **không bao giờ bị sửa**: driver sao chép từng repo sang `.viz-build/<slug>/`,
áp các bản vá (`viz-patches/<slug>/*.patch`) và thay thế chuỗi trong recipe
(`scripts/viz-recipes/<slug>.mjs`; driver báo lỗi ngay nếu mã nguồn đã thay đổi khiến
chuỗi không còn khớp), rồi build và chép kết quả sang `public/viz/<slug>/`.
Trạng thái được ghi vào `public/viz/manifest.json`. Với bài dùng `VizEmbed`, nếu
viz chưa được build (hoặc `ok: false`), bài viết hiển thị thẻ thay thế dẫn tới
bản demo gốc. Trang CNN tích hợp trực tiếp vẫn cần bản build CNN và tài nguyên
trong `public/vendor/cnn/`; không dùng thẻ thay thế của `VizEmbed`.
Driver tiếp tục build các viz còn lại khi một viz lỗi, nhưng kết thúc với mã lỗi;
kiểm tra manifest và chạy lại slug bị lỗi trước khi build blog.
Lần build đầu có thể lâu và cần nhiều GB dung lượng cho bản sao nguồn, phụ thuộc,
mô hình, `public/viz/` và `dist/`. Một số recipe tải phụ thuộc hoặc tài nguyên qua
mạng; `--skip-install` chỉ bỏ cài phụ thuộc nếu bản sao đã có `node_modules`.

## Cấu trúc

```
src/
  content/posts/<category>/<topic>/<slug>.mdx   bài viết (MDX)
  content.config.ts          schema frontmatter (zod)
  assets/covers/             ảnh bìa (≤ 1600px, WebP)
  components/                Callout, KeyTerm, Figure, Toc, PostCard, …
  integrations/              CnnTool, ToolWorkspace, runtime/
  layouts/                   BaseLayout, PostLayout
  lib/                       topics, normalize, synonyms, tags, search (MiniSearch), …
  pages/                     /, /[category]/, /[category]/[slug]/, /chu-de/, /tim-kiem/, …
scripts/                     build-viz, check-viz, prepare-covers
tests/unit, tests/e2e        Vitest, Playwright
```

## Thêm bài viết mới

1. Chọn tên tệp dạng kebab-case chưa tồn tại, ví dụ `cnn-thuc-hanh`.
   Đặt ảnh bìa vào `src/assets/covers/<slug>.webp` (rộng không quá 1600px), hoặc
   dùng lại một ảnh hiện có. Ví dụ dưới đây dùng ảnh và viz CNN đã có sẵn.
2. Tạo `src/content/posts/kien-thuc/thi-giac-may-tinh/cnn-thuc-hanh.mdx` với frontmatter sau. Khi viết bài
   khác, thay tiêu đề, ảnh, chuyên mục và tag bằng dữ liệu thực tế.

   ```yaml
   ---
   title: 'Thực hành quan sát mạng CNN'
   description: 'Theo dõi bộ lọc và các lớp mạng trong CNN Explainer qua một bài thực hành ngắn.'
   category: kien-thuc
   topic: thi-giac-may-tinh
   tags: ['CNN', 'Học sâu', 'Thị giác máy tính']
   viz: cnn-explainer
   vizHeight: 820
   cover: ../../../../assets/covers/cnn-explainer.webp
   order: 5
   date: 2026-10-05
   ---
   ```

   `category` chỉ nhận `kien-thuc` hoặc `du-an`; `description` tối đa 200 ký tự;
   `tags` cần ít nhất một tag. `order` tăng dần quyết định thứ tự trong chuyên mục.
   `viz` là tùy chọn và độc lập với tên tệp bài viết; `vizHeight` mặc định 820,
   nhận số nguyên 200–4000. `topic` phải thuộc `category` theo `src/lib/topics.ts`.
   Ảnh bìa dùng đường dẫn tương đối từ tệp MDX. Tên tệp quyết định URL, thư mục
   chuyên mục không làm thay đổi slug công khai.

3. Ngay sau frontmatter, import các component cần dùng và viết nội dung:

   ```mdx
   import Callout from '../../../../components/Callout.astro';
   import KeyTerm from '../../../../components/KeyTerm.astro';

   ## Mở đầu

   <KeyTerm vi="Mạng nơ-ron tích chập" en="Convolutional Neural Network" /> xử lý ảnh qua nhiều lớp.

   ## Thử với một bức ảnh

   <Callout type="try">

   [Mở trực quan hóa phía trên](#cong-cu), chọn một ảnh và quan sát các lớp tích chập.

   </Callout>

   ## Ghi nhớ

   Mỗi lớp biến đổi biểu diễn đầu vào để hỗ trợ dự đoán cuối cùng.
   ```

   Layout tự hiển thị công cụ phía trên nội dung khi có `viz`: riêng
   `cnn-explainer` dùng `CnnTool`, các viz khác dùng `VizEmbed` toàn chiều rộng.
   **Không import/nhúng thêm `<VizEmbed>` trong thân bài** theo bố cục này.
   Nếu viết bài không có viz, bỏ `viz`, `vizHeight` và liên kết `#cong-cu`.
   `Callout` có ba loại: `try` (Thử ngay), `note` (Lưu ý), `key` (Ghi nhớ).
   Có thể dùng `Figure` từ `../../components/Figure.astro` cho ảnh có `alt`, chú thích
   và nguồn; `KeyTerm` hiển thị thuật ngữ tiếng Việt kèm tiếng Anh.
   Công thức viết bằng KaTeX: `$…$` (inline) và `$$…$$` (khối riêng).
   Lưu ý cú pháp MDX: không dùng `{`, `}`, `<` trần trong văn bản (đặt trong `$…$` hoặc code),
   không dùng comment HTML, và để dòng trống quanh markdown bên trong component.

4. Nếu dùng một viz đã có, chỉ build lại khi cần. Nếu viz mới cần build: thêm recipe
   `scripts/viz-recipes/<slug>.mjs`, đăng ký nó trong
   `scripts/viz-recipes.mjs`, chạy `node scripts/build-viz.mjs <slug>` rồi
   `node scripts/check-viz.mjs <slug>`.
5. Chạy `npm run check && npm test && npm run build`, rồi chạy preview và mở
   `/kien-thuc/cnn-thuc-hanh/` (hoặc `/du-an/<slug>/` theo `category`). Tên tệp quyết
   định slug URL; tiêu đề không đổi URL. Bài tự xuất hiện ở trang chủ, chuyên mục,
   trang tag và chỉ mục tìm kiếm. Mục lục lấy các tiêu đề `##` và `###`.

Tìm kiếm (⌘K / Ctrl+K / `/`) tự động lập chỉ mục bài mới khi build; không cần dấu,
ví dụ "mang no ron" vẫn tìm ra CNN.

## Kiểm tra trước khi hoàn tất

```bash
npm test
npm run check
npm run build
npm run test:e2e
```

`npm test` chạy unit test bằng Vitest; `npm run check` kiểm tra Astro/TypeScript
và schema nội dung. Build phải hoàn tất trước khi chạy test e2e.

Playwright tự mở Astro preview của `dist/` trên cổng 4322; cổng này cần trống.
Nếu Chrome không ở vị trí mặc định, dùng `CHROME=/đường/dẫn/chrome npm run test:e2e`.
Smoke test kiểm tra route, công cụ DOM trực tiếp, liên kết nội bộ, theme, tìm kiếm và tràn ngang
trên mobile. Visual review lưu ảnh tại `test-results/screens/`; một số kiểm tra
kiểm tra toàn bộ 12 bài hiện tại; cần cập nhật kỳ vọng khi chủ động thêm bài mới.
`native-tools.spec.ts` kiểm tra khởi động không iframe, dữ liệu local, thao tác
chọn hồ sơ BERT, bộ dữ liệu FairVis, nhóm VisCUIT, đổi bước Diffusion, chạy GAN
và sinh token GPT-2. Mô hình GPT-2 cần tải khoảng 600 MB ở lần đầu.

## Tích hợp công cụ

Cả 12 bài dùng `InteractivePostLayout`. Adapter nằm trong `src/integrations/`,
runtime và bản địa hóa nằm trong `src/integrations/runtime/`. Thuật ngữ chuyên
ngành giữ tiếng Anh; nút thao tác và hướng dẫn bằng tiếng Việt.

NeRF, 3DGS, SyncTalk và InsTaG có source trong `../tools/`, được nối bằng adapter
mỏng `LocalTool.astro`, không cần recipe bên ngoài. Công cụ kiến thức mới luôn
có hậu tố `-explainer`; công cụ dự án không bắt buộc hậu tố này. Bài viết vẫn
ở `src/content/posts/`, URL độc lập với tên công cụ. Đăng ký công cụ native ở
`../tools/native-tools.mjs` và entry ở `LocalTool.astro`.
Dataset excerpt, cách tái tạo và giới hạn demo được mô tả trong
`../docs/development/radiance-labs.md`; đây không phải inference từ checkpoint.

Trước khi build trên máy mới, chạy `npm ci --prefix ../tools/_shared/radiance`
để cài Three.js, D3 và Open Props của package visualize. Bản mới dùng PLY thật,
scene 3D, pixel inspector và excerpt video/audio đã được phép công khai;
clip render đối chiếu là EmoTag có sẵn, không gán thành output SyncTalk/InsTaG.

### Build adapter native

`npm run build` build entry Svelte độc lập cho Transformer, chuẩn bị 7 adapter
trong `public/native/` rồi build Astro. Không sửa các repo nguồn hoặc bản standalone
trong `public/viz/`. Transformer dùng `scripts/transformer-native/` để bỏ router,
trang giới thiệu và theo dõi bên ngoài; giữ tokenizer và suy luận ONNX local.
Các adapter khác lấy HTML, script và CSS từ output viz, neo đường dẫn ảnh/dữ liệu
về `/viz/<slug>/`, bỏ video iframe và các script theo dõi/đề xuất.

Điều kiện: đã build đủ 8 viz và giữ phụ thuộc Transformer trong
`.viz-build/transformer-explainer/node_modules/` (tạo bằng `npm run build:viz`).
GAN Lab dùng font icon, Material Design Lite và dialog-polyfill local tại
`public/vendor/native/`. Dependency và vendor không thuộc phạm vi chỉnh sửa metadata của công cụ.
