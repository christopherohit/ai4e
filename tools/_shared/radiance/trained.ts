// Readers for data baked from the trained models (see docs/development/radiance-labs.md).
// Nothing here simulates a model: every number comes from a forward pass on the GPU at bake time.
import type { PixelSample, Triple } from './scene-types';

export interface TraceSample { t: number; sigma: number; alpha: number; transmittance: number; weight: number; color: Triple }
export interface TraceGrid { columns: number; rows: number; rays: TraceSample[][] }
export interface ContributorSample { depth: number; opacity: number; alpha: number; transmittance: number; weight: number; color: Triple }
export interface GaussianPixel { x: number; y: number; n: number; rgb: Triple; cudaRgb: Triple; samples: number[][] }

// IEEE 754 half -> float (DataView has no getFloat16 everywhere yet).
export function halfToFloat(bits: number) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

// Packed trace format: int32 rays, int32 fields, int32 offsets[rays + 1], float16 samples[offsets[rays]][fields].
// Fields: t, sigma, alpha, T, weight, r, g, b. Each ray keeps the contiguous window holding 99.9% of its weight.
export function decodeTraces(buffer: ArrayBuffer, columns: number, rows: number): TraceGrid {
  const view = new DataView(buffer);
  const rayCount = view.getInt32(0, true); const fields = view.getInt32(4, true);
  if (rayCount !== columns * rows) throw new Error(`Trace grid mismatch: ${rayCount} rays for ${columns}×${rows}.`);
  if (fields < 8) throw new Error('Trace thiếu trường dữ liệu.');
  const offsets = new Int32Array(buffer, 8, rayCount + 1);
  const body = 8 + 4 * (rayCount + 1);
  if (buffer.byteLength < body + offsets[rayCount] * fields * 2) throw new Error('Trace bị cắt ngắn.');
  const read = (sample: number, field: number) => halfToFloat(view.getUint16(body + (sample * fields + field) * 2, true));
  const rays: TraceSample[][] = [];
  for (let ray = 0; ray < rayCount; ray++) {
    const samples: TraceSample[] = [];
    for (let sample = offsets[ray]; sample < offsets[ray + 1]; sample++) {
      samples.push({ t: read(sample, 0), sigma: read(sample, 1), alpha: read(sample, 2), transmittance: read(sample, 3), weight: read(sample, 4), color: [read(sample, 5), read(sample, 6), read(sample, 7)] });
    }
    rays.push(samples);
  }
  return { columns, rows, rays };
}

// Map a normalized canvas position to the nearest baked ray of a columns × rows grid (ray centres at cell centres).
export function nearestRay(grid: { columns: number; rows: number }, position: [number, number]) {
  const column = Math.max(0, Math.min(grid.columns - 1, Math.floor(position[0] * grid.columns)));
  const row = Math.max(0, Math.min(grid.rows - 1, Math.floor(position[1] * grid.rows)));
  return { index: row * grid.columns + column, centre: [(column + 0.5) / grid.columns, (row + 0.5) / grid.rows] as [number, number] };
}

export function summarize(samples: { weight: number; transmittance: number; alpha: number; color: Triple }[]) {
  const color: Triple = [0, 0, 0];
  let total = 0;
  for (const sample of samples) { total += sample.weight; for (let channel = 0; channel < 3; channel++) color[channel] += sample.weight * sample.color[channel]; }
  const last = samples.at(-1);
  return { color, weightSum: total, finalTransmittance: last ? last.transmittance * (1 - last.alpha) : 1 };
}

export function traceToPixelSamples(samples: TraceSample[]): PixelSample[] {
  return samples.map(sample => ({ depth: sample.t, density: sample.sigma, alpha: sample.alpha, weight: sample.weight, transmittance: sample.transmittance }));
}

export function contributors(pixel: GaussianPixel): ContributorSample[] {
  return pixel.samples.map(([depth, opacity, alpha, transmittance, weight, red, green, blue]) => ({ depth, opacity, alpha, transmittance, weight, color: [red, green, blue] }));
}

export function contributorsToPixelSamples(samples: ContributorSample[]): PixelSample[] {
  return samples.map(sample => ({ depth: sample.depth, density: sample.opacity, alpha: sample.alpha, weight: sample.weight, transmittance: sample.transmittance }));
}

// InsTaG displacement: float16 [count, 3] per baked frame.
export function decodeDisplacements(buffer: ArrayBuffer) {
  const view = new DataView(buffer); const values = new Float32Array(buffer.byteLength / 2);
  for (let index = 0; index < values.length; index++) values[index] = halfToFloat(view.getUint16(index * 2, true));
  return values;
}
