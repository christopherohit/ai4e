import { readFileSync, readdirSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { decodeDisplacements, decodeTraces, halfToFloat, nearestRay, summarize, contributors } from '../../../tools/_shared/radiance/trained';

const data = new URL('../../../tools/_shared/radiance/data/trained/', import.meta.url);
const bytes = (path: string) => { const file = readFileSync(new URL(path, data)); return file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength); };
const json = (path: string) => JSON.parse(readFileSync(new URL(path, data), 'utf8'));

function packed(rays: number[][][]) {
  const total = rays.reduce((sum, ray) => sum + ray.length, 0);
  const buffer = new ArrayBuffer(8 + 4 * (rays.length + 1) + total * 8 * 2);
  const view = new DataView(buffer); view.setInt32(0, rays.length, true); view.setInt32(4, 8, true);
  let offset = 0; let cursor = 8 + 4 * (rays.length + 1);
  rays.forEach((ray, index) => { view.setInt32(8 + 4 * index, offset, true); offset += ray.length; });
  view.setInt32(8 + 4 * rays.length, offset, true);
  // 0x3c00 = 1.0, 0x3800 = 0.5, 0x0000 = 0
  for (const ray of rays) for (const sample of ray) for (const value of sample) { view.setUint16(cursor, value === 1 ? 0x3c00 : value === 0.5 ? 0x3800 : 0, true); cursor += 2; }
  return buffer;
}

describe('baked trained-model data', () => {
  it('decodes IEEE half floats', () => {
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x3555)).toBeCloseTo(1 / 3, 3);
    expect(halfToFloat(0x0001)).toBeCloseTo(2 ** -24, 30);
  });
  it('decodes windowed traces and rejects truncated or mismatched files', () => {
    const buffer = packed([[[1, 1, 0.5, 1, 0.5, 1, 0, 0], [1, 1, 1, 0.5, 0.5, 0, 1, 0]], []]);
    const grid = decodeTraces(buffer, 2, 1);
    expect(grid.rays[0]).toHaveLength(2);
    expect(grid.rays[1]).toHaveLength(0);
    const summary = summarize(grid.rays[0]);
    expect(summary.weightSum).toBe(1);
    expect(summary.color).toEqual([0.5, 0.5, 0]);
    expect(summary.finalTransmittance).toBe(0);
    expect(() => decodeTraces(buffer, 3, 1)).toThrow('mismatch');
    expect(() => decodeTraces(buffer.slice(0, buffer.byteLength - 2), 2, 1)).toThrow('cắt ngắn');
  });
  it('maps canvas positions to the nearest baked ray', () => {
    expect(nearestRay({ columns: 4, rows: 2 }, [0, 0]).index).toBe(0);
    expect(nearestRay({ columns: 4, rows: 2 }, [0.99, 0.99]).index).toBe(7);
    expect(nearestRay({ columns: 4, rows: 2 }, [0.3, 0.6])).toEqual({ index: 5, centre: [0.375, 0.75] });
  });
  it('NeRF traces from the trained model conserve energy: Σw + T_end = 1', () => {
    const meta = json('nerf/meta.json');
    const grid = decodeTraces(bytes('nerf/traces-00.bin'), meta.grid[0], meta.grid[1]);
    const hits = grid.rays.map(ray => summarize(ray)).filter(summary => summary.weightSum > 0.01);
    expect(hits.length).toBeGreaterThan(grid.rays.length * 0.9);
    // float16 storage keeps each term to ~1e-3; residual light past the last sample is T_end, not lost weight.
    for (const summary of hits) expect(summary.weightSum + summary.finalTransmittance).toBeCloseTo(1, 2);
  });
  it('SyncTalk traces cover the head and leave background rays empty', () => {
    const meta = json('synctalk/meta.json');
    const grid = decodeTraces(bytes('synctalk/traces-0100.bin'), meta.grid[0], meta.grid[1]);
    const centre = grid.rays[nearestRay(grid, [0.5, 0.4]).index];
    const corner = grid.rays[nearestRay(grid, [0.02, 0.02]).index];
    expect(summarize(centre).weightSum).toBeGreaterThan(0.95);
    expect(summarize(corner).weightSum).toBeLessThan(0.05);
  });
  it('3DGS pixel contributors reproduce the CUDA rasterizer colour', () => {
    const meta = json('gaussian/meta.json');
    for (const view of meta.views) expect(view.maxAbsDiffVsCuda).toBeLessThan(0.002);
    const pixels = json('gaussian/pixels-08.json');
    const pixel = pixels.find((candidate: { n: number }) => candidate.n > 10);
    const samples = contributors(pixel);
    const depths = samples.map(sample => sample.depth);
    expect(depths).toEqual([...depths].sort((first, second) => first - second));
    for (const sample of samples) expect(sample.weight).toBeCloseTo(sample.alpha * sample.transmittance, 4);
  });
  it('InsTaG displacements and their projections cover every Gaussian and stay on the face', () => {
    const meta = json('instag/meta.json');
    const files = readdirSync(new URL('instag/', data)).filter(name => name.startsWith('dxyz-face-'));
    expect(files.length).toBe(meta.frames.length);
    const face = decodeDisplacements(bytes('instag/' + files[0]));
    expect(face.length).toBe(meta.faceCount * 3);
    expect(Math.max(...Array.from(face, Math.abs))).toBeLessThan(0.5);
    const projected = decodeDisplacements(bytes('instag/' + files[0].replace('dxyz-', 'proj-')));
    expect(projected.length).toBe(meta.faceCount * 4);
    for (let index = 0; index < projected.length; index += 4) {
      expect(projected[index]).toBeGreaterThan(0); expect(projected[index]).toBeLessThan(meta.width);
      expect(projected[index + 1]).toBeGreaterThan(0); expect(projected[index + 1]).toBeLessThan(meta.height);
    }
  });
});
