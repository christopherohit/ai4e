import { decodeTraces, nearestRay, type TraceGrid } from '../../_shared/radiance/trained';
import { asset, drawImageCover, loadImage, pad, rayInspector, traceSamples, type Explanations, type LabView, type Root, type ViewContext } from '../../_shared/radiance/lab-core';

export const explanations: Explanations = [
    ['Ảnh & camera', 'Model học từ 175 ảnh drone đã biết pose. Mỗi pixel là một ray r(t) = o + t·d xuất phát từ tâm camera.', 'r(t) = o + t·d'],
    ['Ray sampling', 'Proposal network chọn 48 điểm trên ray, dồn vào vùng có vật thể thay vì chia đều.', 't₁ < t₂ < … < t₄₈'],
    ['Density σ', 'MLP đã train trả về mật độ σ và màu c tại từng điểm. σ lớn nghĩa là điểm đó nằm trên bề mặt.', 'F(x, d) → (σ, c)'],
    ['Volume rendering', 'Màu pixel là tổng màu các sample, trọng số w = T·α. T (transmittance) giảm dần khi ray đi qua vật thể.', 'C = Σ Tᵢ·αᵢ·cᵢ, αᵢ = 1 − e^(−σᵢδᵢ)'],
];

// ---------- NeRF: baked orbit + real per-sample traces ----------
export async function createView(root: Root, context: ViewContext): Promise<LabView> {
  const meta = await context.fetchJson<{ grid: [number, number]; frames: { index: number; azimuth: number; depthRange: [number, number] }[] }>('nerf/meta.json');
  const images = new Map<string, HTMLImageElement>(); const traces = new Map<number, TraceGrid>();
  let frame = 0;
  const image = async (kind: string, index: number) => { const key = kind + index; if (!images.has(key)) images.set(key, await loadImage(asset(`nerf/${kind}-${pad(index, 2)}.webp`))); return images.get(key)!; };
  const preload = async () => { for (let index = 0; index < meta.frames.length; index++) { if (root.signal.aborted) return; await image('frame', index).catch(() => {}); context.onProgress((index + 1) / meta.frames.length); } };
  await image('frame', 0); void preload();
  return {
    async render(state) { drawImageCover(context.canvas, await image(state.layer === 'depth' ? 'depth' : 'frame', frame)); },
    overlay(draw, width, height, state) {
      if (state.stage === 1 || state.stage === 2) {
        draw.fillStyle = 'rgb(255 255 255 / 0.55)';
        for (let row = 0; row < meta.grid[1]; row++) for (let column = 0; column < meta.grid[0]; column++) { draw.beginPath(); draw.arc((column + 0.5) / meta.grid[0] * width, (row + 0.5) / meta.grid[1] * height, 1.6, 0, Math.PI * 2); draw.fill(); }
      }
    },
    async inspect(state) {
      // Pin the frame: the slider can move while this frame's traces are downloading.
      const current = frame;
      if (!traces.has(current)) traces.set(current, decodeTraces(await context.fetchBytes(`nerf/traces-${pad(current, 2)}.bin`), meta.grid[0], meta.grid[1]));
      const grid = traces.get(current)!; const ray = nearestRay(grid, state.selected); state.selected = ray.centre;
      const samples = traceSamples(grid.rays[ray.index]);
      rayInspector(root, samples, { densityLabel: 'Density σ', selection: `Ray ${ray.index + 1} / ${grid.rays.length} · góc ${meta.frames[current].azimuth.toFixed(0)}°`, caption: 'Các sample proposal network giữ lại trên ray, xếp từ gần đến xa. σ và w chuẩn hóa theo đỉnh để thấy hình dạng; T giữ thang 0–1. Bảng bên dưới có giá trị gốc.' });
    },
    onInput(name, value) { if (name === 'position') frame = value; },
    drag(dx) { frame = (frame + (dx > 0 ? 1 : -1) + meta.frames.length) % meta.frames.length; const input = root.element.querySelector<HTMLInputElement>('input[name=position]')!; input.value = String(frame); root.element.querySelector('[data-value="position"]')!.textContent = `${frame} × 10°`; },
    resetCamera() { frame = 0; const input = root.element.querySelector<HTMLInputElement>('input[name=position]')!; input.value = '0'; root.element.querySelector('[data-value="position"]')!.textContent = '0 × 10°'; },
    dispose() { images.clear(); traces.clear(); },
  };
}

