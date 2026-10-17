import { drawChart } from '../../_shared/radiance/charts';
import { contributors, nearestRay, type ContributorSample, type GaussianPixel } from '../../_shared/radiance/trained';
import { asset, pad, rayInspector, type Explanations, type LabView, type Root, type ViewContext } from '../../_shared/radiance/lab-core';
import type { SplatRig, SplatView } from '../../_shared/radiance/splat-view';

export const explanations: Explanations = [
    ['Gaussian 3D', 'Scene là 1.25 triệu Gaussian đã train: vị trí μ, scale s, quaternion q, opacity và màu SH bậc 3.', 'Σ = R(q)·diag(s)²·R(q)ᵀ'],
    ['Footprint', 'Mỗi Gaussian được chiếu thành một ellipse 2D. Thu nhỏ footprint (σ) để thấy từng Gaussian riêng lẻ.', 'Σ₂D = J·W·Σ·Wᵀ·Jᵀ'],
    ['Sort theo depth', 'Các Gaussian phủ lên một pixel được sắp xếp gần → xa trước khi trộn màu.', 'z₁ < z₂ < … < zₙ'],
    ['Alpha compositing', 'Gaussian gần che Gaussian xa. Trọng số w = α·T giống volume rendering, nhưng không cần lấy mẫu dọc ray.', 'C = Σ cᵢ·αᵢ·Π(1 − αⱼ)'],
];

// ---------- 3D Gaussian Splatting: live WebGL ----------
export async function createView(root: Root, context: ViewContext): Promise<LabView> {
  const { SplatView: View } = await import('../../_shared/radiance/splat-view');
  const rig = await context.fetchJson<SplatRig>('gaussian/cameras.json');
  const meta = await context.fetchJson<{ grid: [number, number]; views: { index: number; name: string; width: number; height: number; maxAbsDiffVsCuda: number }[] }>('gaussian/meta.json');
  const view: SplatView = new View(context.canvas, rig);
  await view.load(asset('drone-3dgs.spz'), context.onProgress);
  let viewIndex = -1; const pixels = new Map<number, GaussianPixel[]>();
  const tuning = { stdDev: Math.sqrt(8), stage: 3 };
  const apply = () => view.setFootprint(tuning.stage === 0 ? 0.6 : tuning.stage === 1 ? tuning.stdDev : Math.sqrt(8));
  return {
    // WebGL draws on its own animation frame; the inspector does not wait for it (software GL can take seconds).
    async render() { view.useView(viewIndex >= 0 ? rig.views[viewIndex] : undefined); apply(); view.request(); },
    async inspect(state) {
      if (viewIndex < 0) {
        const $ = root.element.querySelector.bind(root.element);
        $('[data-selection]')!.textContent = 'Chọn một test view trong mục Camera để xem các Gaussian tạo nên từng pixel.';
        $('[data-caption]')!.textContent = 'Orbit tự do render live trên GPU. Danh sách Gaussian theo pixel được tính sẵn cho 4 test view (ảnh không dùng khi train).';
        drawChart($('[data-chart]') as SVGSVGElement, { series: [], xLabels: ['gần camera', 'xa camera'], caption: '' });
        ($('[data-readout]') as HTMLElement).replaceChildren(); ($('[data-table]') as HTMLTableElement).replaceChildren();
        return;
      }
      const info = meta.views[viewIndex];
      if (!pixels.has(info.index)) pixels.set(info.index, await context.fetchJson<GaussianPixel[]>(`gaussian/pixels-${pad(info.index, 2)}.json`));
      const ray = nearestRay({ columns: meta.grid[0], rows: meta.grid[1] }, state.selected);
      state.selected = ray.centre;
      const pixel = pixels.get(info.index)![ray.index];
      const samples: ContributorSample[] = contributors(pixel);
      rayInspector(root, samples.map(sample => ({ ...sample, density: sample.opacity })), {
        densityLabel: 'Opacity', selection: `Pixel (${pixel.x}, ${pixel.y}) · ${info.name}`,
        caption: `${pixel.n} Gaussian phủ lên pixel này; biểu đồ hiển thị ${samples.length} Gaussian có w lớn nhất, xếp theo depth. Opacity và w chuẩn hóa theo đỉnh để thấy hình dạng; T giữ thang 0–1. Màu tính lại khớp CUDA rasterizer (sai lệch ${info.maxAbsDiffVsCuda.toExponential(1)}).`,
        extraRows: [['Màu CUDA', pixel.cudaRgb.map(value => value.toFixed(3)).join(' · ')], ['Tổng Gaussian', String(pixel.n)]],
      });
    },
    onInput(name, value) { if (name === 'position') { view.orbit = value; viewIndex = -1; const select = root.element.querySelector<HTMLSelectElement>('[data-view]'); if (select) select.selectedIndex = 0; } if (name === 'detail') tuning.stdDev = value; },
    onStage(stage) { tuning.stage = stage; },
    onView(index) { viewIndex = index; if (index >= 0) { const grid = meta.grid; root.element.dataset.inspectGrid = `${grid[0]}x${grid[1]}`; } },
    drag(dx, dy) { viewIndex = -1; const select = root.element.querySelector<HTMLSelectElement>('[data-view]'); if (select) select.selectedIndex = 0; view.orbit = ((view.orbit + dx * 0.4 + 540) % 360) - 180; view.elevation = Math.max(-30, Math.min(50, view.elevation - dy * 0.25)); const input = root.element.querySelector<HTMLInputElement>('input[name=position]'); if (input) { input.value = String(Math.round(view.orbit)); input.dispatchEvent(new Event('change')); root.element.querySelector('[data-value="position"]')!.textContent = input.value + ' °'; } },
    zoom(factor) { view.distance = Math.max(0.35, Math.min(2.5, view.distance * factor)); },
    resetCamera() { view.orbit = 0; view.elevation = 0; view.distance = 1; },
    dispose() { view.dispose(); },
  };
}

