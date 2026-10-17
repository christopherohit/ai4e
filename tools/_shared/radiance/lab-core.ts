import { drawChart, fillTable, type Series } from './charts';
import { summarize, type TraceSample } from './trained';
import type { Triple } from './scene-types';

// Shared pieces of the radiance labs: asset lookup, the view contract each tool implements, and the ray inspector.
export type Mode = 'nerf' | 'gaussian-splatting' | 'synctalk' | 'instag';
export type Explanations = [string, string, string][];
const assets = import.meta.glob('./data/trained/**/*.{json,webp,bin,f16,f32,spz}', { query: '?url', import: 'default', eager: true }) as Record<string, string>;
export const asset = (path: string) => { const url = assets['./data/trained/' + path]; if (!url) throw new Error('Thiếu dữ liệu: ' + path); return url; };
export const pad = (value: number, size: number) => String(value).padStart(size, '0');

export interface Root { element: HTMLElement; mode: Mode; signal: AbortSignal }
export interface Clip { fps: number; sourceStart: number; mouth: number[]; eye: number[]; brow: number[]; rms: number[] }

export interface ViewContext { $: <T extends Element = HTMLElement>(selector: string) => T | null; canvas: HTMLCanvasElement; fetchJson: <T>(path: string) => Promise<T>; fetchBytes: (path: string) => Promise<ArrayBuffer>; onProgress: (fraction: number) => void }
export interface LabState { stage: number; selected: [number, number]; layer: string }
export interface LabView {
  render(state: LabState): Promise<void>;
  inspect(state: LabState): Promise<void>;
  overlay?(context: CanvasRenderingContext2D, width: number, height: number, state: LabState): void;
  onInput(name: string, value: number): void;
  onStage?(stage: number): void;
  onView?(index: number): void;
  drag?(dx: number, dy: number): void;
  zoom?(factor: number): void;
  resetCamera(): void;
  frame?: number;
  setFrame?(frame: number): void;
  dispose(): void;
}

// ---------- shared inspector rendering ----------
export function rayInspector(root: Root, samples: { depth: number; density: number; alpha: number; transmittance: number; weight: number; color: Triple }[], options: { densityLabel: string; caption: string; selection: string; prefix?: string; extraRows?: [string, string][] }) {
  const $ = <T extends Element = HTMLElement>(selector: string) => root.element.querySelector<T>(selector);
  const prefix = options.prefix ?? '';
  const peakDensity = Math.max(1e-6, ...samples.map(sample => sample.density));
  const peakWeight = Math.max(1e-6, ...samples.map(sample => sample.weight));
  const series: Series[] = [
    { key: 'density', label: options.densityLabel, values: samples.map(sample => sample.density / peakDensity) },
    { key: 'weight', label: 'Weight w', values: samples.map(sample => sample.weight / peakWeight) },
    { key: 'T', label: 'Transmittance T', values: samples.map(sample => sample.transmittance), step: true },
  ];
  const svg = $<SVGSVGElement>(prefix ? `[data-${prefix}-chart]` : '[data-chart]')!;
  drawChart(svg, { series, xLabels: ['gần camera', 'xa camera'], caption: options.caption });
  if (!prefix) $('[data-caption]')!.textContent = options.caption;
  $(prefix ? `[data-${prefix}-selection]` : '[data-selection]')!.textContent = options.selection;
  const summary = summarize(samples);
  const rows: [string, string][] = [
    ['Màu pixel', summary.color.map(value => value.toFixed(3)).join(' · ')],
    ['Số mẫu đóng góp', String(samples.length)],
    ['Σ w', summary.weightSum.toFixed(4)],
    ['T cuối', summary.finalTransmittance.toFixed(4)],
    ...(options.extraRows ?? []),
  ];
  const readout = $(prefix ? `[data-${prefix}-readout]` : '[data-readout]')!; readout.replaceChildren();
  rows.forEach(([label, value]) => { const wrap = document.createElement('div'); const term = document.createElement('dt'); term.textContent = label; const detail = document.createElement('dd'); detail.textContent = value; wrap.append(term, detail); readout.append(wrap); });
  if (!prefix) fillTable($<HTMLTableElement>('[data-table]')!, ['#', 'depth', options.densityLabel, 'α', 'T', 'w'], samples.map((sample, index) => [String(index + 1), sample.depth, sample.density, sample.alpha, sample.transmittance, sample.weight]));
}

export function traceSamples(samples: TraceSample[]) { return samples.map(sample => ({ depth: sample.t, density: sample.sigma, alpha: sample.alpha, transmittance: sample.transmittance, weight: sample.weight, color: sample.color })); }

export async function loadImage(url: string) { const image = new Image(); image.decoding = 'async'; image.src = url; await image.decode(); return image; }
export function drawImageCover(canvas: HTMLCanvasElement, image: CanvasImageSource & { width: number; height: number }, pixelated = false) {
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.round(canvas.clientWidth * ratio); const height = Math.round(canvas.clientHeight * ratio);
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const context = canvas.getContext('2d')!; context.imageSmoothingEnabled = !pixelated; context.imageSmoothingQuality = 'high';
  context.fillStyle = '#0e1c27'; context.fillRect(0, 0, width, height);
  context.drawImage(image, 0, 0, width, height);
}

