import { drawChart, fillTable } from './charts';
import { decodeDisplacements, decodeTraces, nearestRay, type TraceGrid } from './trained';
import { asset, drawImageCover, loadImage, pad, rayInspector, traceSamples, type Clip, type LabView, type Root, type ViewContext } from './lab-core';

// ---------- SyncTalk / InsTaG: real test clip + baked traces / motion ----------
export async function talkingView(root: Root, context: ViewContext): Promise<LabView> {
  const synctalk = root.mode === 'synctalk';
  const folder = synctalk ? 'synctalk' : 'instag';
  const meta = await context.fetchJson<{ grid?: [number, number]; width: number; height: number; frames: { frame?: number; index?: number }[]; faceCount?: number }>(`${folder}/meta.json`);
  const clip = await context.fetchJson<Clip>('clip/signals.json');
  const baked = meta.frames.map(entry => entry.frame ?? entry.index ?? 0);
  const nearestBaked = (frame: number) => baked.reduce((best, value) => Math.abs(value - frame) < Math.abs(best - frame) ? value : best, baked[0]);
  const images = new Map<string, HTMLImageElement>(); const traces = new Map<number, TraceGrid>();
  // InsTaG per baked frame: Δμ (world units, for statistics) and the projected segment μ → μ+Δμ in test-camera pixels.
  const motion = new Map<number, { displacement: Float32Array; projected: Float32Array; faceCount: number }>();
  const image = async (path: string) => { if (!images.has(path)) images.set(path, await loadImage(asset(path))); return images.get(path)!; };
  let frame = 100; let magnify = 3;
  const loadMotion = async (key: number) => {
    const join = (first: Float32Array, second: Float32Array) => { const joined = new Float32Array(first.length + second.length); joined.set(first); joined.set(second, first.length); return joined; };
    const [faceDisplacement, mouthDisplacement, faceProjected, mouthProjected] = await Promise.all(['dxyz-face', 'dxyz-mouth', 'proj-face', 'proj-mouth'].map(async name => decodeDisplacements(await context.fetchBytes(`instag/${name}-${pad(key, 4)}.f16`))));
    motion.set(key, { displacement: join(faceDisplacement, mouthDisplacement), projected: join(faceProjected, mouthProjected), faceCount: faceDisplacement.length / 3 });
  };
  context.onProgress(1);
  const view: LabView = {
    frame,
    setFrame(value) { frame = value; view.frame = value; },
    async render(state) {
      const key = nearestBaked(frame);
      if (synctalk) {
        if (state.layer === 'depth') drawImageCover(context.canvas, await image(`synctalk/depth-${pad(key, 4)}.webp`));
        else drawImageCover(context.canvas, await image(`synctalk/head-${pad(key, 4)}.webp`));
      } else if (state.layer === 'face' || state.layer === 'mouth') drawImageCover(context.canvas, await image(`instag/alpha-${state.layer}-${pad(key, 4)}.webp`));
      else drawImageCover(context.canvas, await image(`instag/frame-${pad(key, 4)}.webp`));
      if (!synctalk && !motion.has(key)) await loadMotion(key);
    },
    overlay(draw, width, height, state) {
      if (synctalk || !root.element.querySelector<HTMLInputElement>('[data-vectors]')?.checked || state.stage === 3) return;
      const entry = motion.get(nearestBaked(frame)); if (!entry) return;
      // Stage 1 shows the motion field on the face; stage 0 and 2 show it sparser so the picture stays readable.
      const sx = width / meta.width; const sy = height / meta.height; const count = entry.projected.length / 4;
      const faceStep = state.stage === 1 ? 4 : 9;
      for (let index = 0; index < count; index += index < entry.faceCount ? faceStep : 1) {
        const x0 = entry.projected[index * 4] * sx; const y0 = entry.projected[index * 4 + 1] * sy;
        const x1 = x0 + (entry.projected[index * 4 + 2] * sx - x0) * magnify; const y1 = y0 + (entry.projected[index * 4 + 3] * sy - y0) * magnify;
        draw.strokeStyle = index < entry.faceCount ? 'rgb(253 224 71 / 0.8)' : 'rgb(251 113 133 / 0.95)';
        draw.lineWidth = 1; draw.beginPath(); draw.moveTo(x0, y0); draw.lineTo(x1, y1); draw.stroke();
      }
    },
    async inspect(state) {
      // Pin the frame: playback or the slider can move it while baked data for this frame is downloading.
      const frame = view.frame ?? 100;
      const key = nearestBaked(frame);
      // Signals chart (shared clip): AU25, audio RMS, AU45 on one 0–1 axis after dividing by each series' own peak.
      const peak = (values: number[]) => Math.max(1e-6, ...values);
      drawChart(root.element.querySelector<SVGSVGElement>('[data-chart]')!, {
        series: [
          { key: 'au25', label: 'AU25 (mở môi)', values: clip.mouth.map(value => value / peak(clip.mouth)) },
          { key: 'rms', label: 'Audio RMS', values: clip.rms.map(value => value / peak(clip.rms)) },
          { key: 'au45', label: 'AU45 (chớp mắt)', values: clip.eye.map(value => value / peak(clip.eye)) },
        ],
        xLabels: ['0 s', `${((clip.mouth.length - 1) / clip.fps).toFixed(1)} s`], cursor: frame, caption: '',
        onPick: index => { view.setFrame?.(index); const input = root.element.querySelector<HTMLInputElement>('input[name=position]')!; input.value = String(index); input.dispatchEvent(new Event('input')); },
      });
      root.element.querySelector('[data-caption]')!.textContent = 'AU25 và AU45 là cường độ OpenFace (thang gốc 0–5), audio RMS tính từ waveform 16 kHz; mỗi đường chia cho đỉnh của chính nó để so hình dạng. Bảng có giá trị gốc. Click vào biểu đồ để nhảy tới frame.';
      root.element.querySelector('[data-selection]')!.textContent = `Frame ${frame} · nguồn ${clip.sourceStart + frame} · ${(frame / clip.fps).toFixed(2)} s`;
      const readout = root.element.querySelector('[data-readout]')!; readout.replaceChildren();
      for (const [label, value] of [['AU25', clip.mouth[frame].toFixed(2)], ['AU45', clip.eye[frame].toFixed(2)], ['Audio RMS', clip.rms[frame].toFixed(4)], ['Frame tính sẵn gần nhất', String(key)]]) { const wrap = document.createElement('div'); const term = document.createElement('dt'); term.textContent = label; const detail = document.createElement('dd'); detail.textContent = value; wrap.append(term, detail); readout.append(wrap); }
      fillTable(root.element.querySelector<HTMLTableElement>('[data-table]')!, ['Frame', 'AU25', 'AU45', 'AU01', 'RMS'], clip.mouth.map((_, index) => [String(index), clip.mouth[index], clip.eye[index], clip.brow[index], clip.rms[index]]));
      if (synctalk) {
        if (!traces.has(key)) traces.set(key, decodeTraces(await context.fetchBytes(`synctalk/traces-${pad(key, 4)}.bin`), meta.grid![0], meta.grid![1]));
        const grid = traces.get(key)!; const ray = nearestRay(grid, state.selected); state.selected = ray.centre;
        rayInspector(root, traceSamples(grid.rays[ray.index]), { prefix: 'ray', densityLabel: 'Density σ', selection: `Ray ${ray.index + 1} · frame ${key}`, caption: '' });
      } else {
        if (!motion.has(key)) await loadMotion(key);
        const { displacement, faceCount } = motion.get(key)!;
        const magnitudes = (from: number, to: number) => { const values: number[] = []; for (let index = from; index < to; index++) values.push(Math.hypot(displacement[index * 3], displacement[index * 3 + 1], displacement[index * 3 + 2])); return values.sort((first, second) => first - second); };
        const face = magnitudes(0, faceCount); const mouth = magnitudes(faceCount, displacement.length / 3);
        const median = (values: number[]) => values[Math.floor(values.length / 2)]; const p95 = (values: number[]) => values[Math.floor(values.length * 0.95)];
        root.element.querySelector('[data-ray-title]')!.textContent = 'Δμ tại frame đã chọn';
        root.element.querySelector('[data-ray-selection]')!.textContent = `Frame ${key} · ${faceCount.toLocaleString('vi')} Gaussian mặt, ${(displacement.length / 3 - faceCount).toLocaleString('vi')} Gaussian miệng`;
        const bins = 24; const top = Math.max(p95(face), p95(mouth)) * 1.2;
        const histogram = (values: number[]) => { const counts = new Array(bins).fill(0); values.forEach(value => { counts[Math.min(bins - 1, Math.floor(value / top * bins))]++; }); const max = Math.max(...counts); return counts.map(count => count / max); };
        drawChart(root.element.querySelector<SVGSVGElement>('[data-ray-chart]')!, { series: [{ key: 'face', label: 'Mặt', values: histogram(face) }, { key: 'mouth', label: 'Miệng', values: histogram(mouth) }], xLabels: ['|Δμ| = 0', `${top.toFixed(3)}`], caption: '' });
        const rayReadout = root.element.querySelector('[data-ray-readout]')!; rayReadout.replaceChildren();
        for (const [label, value] of [['Median |Δμ| mặt', median(face).toFixed(4)], ['P95 |Δμ| mặt', p95(face).toFixed(4)], ['Median |Δμ| miệng', median(mouth).toFixed(4)], ['P95 |Δμ| miệng', p95(mouth).toFixed(4)]]) { const wrap = document.createElement('div'); const term = document.createElement('dt'); term.textContent = label; const detail = document.createElement('dd'); detail.textContent = value; wrap.append(term, detail); rayReadout.append(wrap); }
      }
    },
    onInput(name, value) { if (name === 'position') { frame = value; view.frame = value; } if (name === 'detail') magnify = value; },
    resetCamera() {},
    dispose() { images.clear(); traces.clear(); motion.clear(); },
  };
  return view;
}

