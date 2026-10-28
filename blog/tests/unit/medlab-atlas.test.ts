import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { decodeOct, dequantize, explodeOffset, FEMALE_SCALE, groupCenters, inAtlas, parseParts, rank, sideOf, warp, widthAt, type Manifest } from '../../../tools/du-an/medlab-body-atlas/atlas';
import { DEFAULT_SYSTEMS, ORGANS, QUIZ, REGION_INFO, SYSTEM_INFO, TOUR } from '../../../tools/du-an/medlab-body-atlas/organs';

const dir = resolve('../tools/du-an/medlab-body-atlas/data');
const manifest = JSON.parse(readFileSync(resolve(dir, 'atlas.json'), 'utf8')) as Manifest;
const parts = parseParts(manifest);
const files = manifest.files.map(f => readFileSync(resolve(dir, f.file)));

describe('MedLab atlas data', () => {
  it('matches the binary files part by part', () => {
    expect(parts).toHaveLength(2312);
    manifest.files.forEach((f, i) => expect(files[i].byteLength).toBe(f.bytes));
    let triangles = 0;
    for (const p of parts) {
      const file = files[p.file];
      expect(p.indexCount % 3).toBe(0);
      expect(p.vertexOffset + p.vertexCount * 8).toBeLessThanOrEqual(file.byteLength);
      expect(p.indexOffset % 2).toBe(0);
      const indices = new Uint16Array(file.buffer, file.byteOffset + p.indexOffset, p.indexCount);
      expect(Math.max(...indices)).toBeLessThan(p.vertexCount);
      triangles += p.indexCount / 3;
    }
    expect(triangles).toBe(manifest.triangles);
  });

  it('keeps vertices inside each part box after dequantising', () => {
    for (const p of parts.filter((_, i) => i % 97 === 0)) {
      const view = new DataView(files[p.file].buffer, files[p.file].byteOffset + p.vertexOffset, p.vertexCount * 8);
      for (let v = 0; v < p.vertexCount; v++) for (let a = 0; a < 3; a++) {
        const x = dequantize(view.getInt16(v * 8 + a * 2, true), a, manifest);
        expect(Math.abs(x - p.center[a])).toBeLessThanOrEqual(p.half[a] + 2e-4);
      }
    }
  });

  it('separates the male and female atlases', () => {
    const male = parts.filter(p => inAtlas(p, 'male')), female = parts.filter(p => inAtlas(p, 'female'));
    expect(male).toHaveLength(2256);
    expect(female).toHaveLength(2298);
    expect(parts.filter(p => p.sex === 1).every(p => p.system === 'reproductive' || /bladder|urethra/i.test(p.name))).toBe(true);
    expect(parts.filter(p => p.sex === 2).every(p => p.source === 'HRA')).toBe(true);
    expect(parts.filter(p => p.source === 'BodyParts3D').every(p => /^FMA\d+$/.test(p.fma))).toBe(true);
  });

  it('describes every organ in Vietnamese and resolves it to parts', () => {
    expect(Object.keys(ORGANS).sort()).toEqual(Object.keys(manifest.organs).sort());
    expect(manifest.organs.heart.parts).toHaveLength(83);
    expect(manifest.organs.heart.concepts).toEqual(['FMA7088']);
    for (const [key, organ] of Object.entries(ORGANS)) {
      expect(organ.name.length).toBeGreaterThan(1);
      expect(organ.description.length).toBeGreaterThan(20);
      const sexes = new Set(manifest.organs[key].parts.map(i => parts[i].sex));
      if (organ.sex === 'female') expect([...sexes]).toEqual([2]);
      if (organ.sex === 'male') expect([...sexes]).toEqual([1]);
    }
    for (const [key] of TOUR) expect(ORGANS[key]).toBeDefined();
    for (const key of QUIZ) expect(ORGANS[key]).toBeDefined();
    expect(Object.keys(SYSTEM_INFO).sort()).toEqual([...manifest.systems].sort());
    expect(Object.keys(REGION_INFO).sort()).toEqual([...manifest.regions].sort());
    expect(DEFAULT_SYSTEMS).not.toContain('muscular');
  });

  it('keeps the numbers quoted in the article', () => {
    const article = readFileSync(resolve('src/content/posts/du-an/medlab/ban-do-co-the-nguoi.mdx'), 'utf8');
    const visible = parts.filter(p => inAtlas(p, 'male') && DEFAULT_SYSTEMS.includes(p.system)).length;
    expect(article).toContain('**2 312 cấu trúc**');
    expect(article).toContain(`**${visible} cấu trúc**`);
    expect(article).toContain(`**${Math.round(manifest.triangles / 1000)} nghìn tam giác**`);
    expect(article).toContain('viz: medlab-body-atlas');
  });
});

describe('MedLab atlas helpers', () => {
  it('decodes octahedral normals to unit vectors', () => {
    for (const [x, y] of [[0, 0], [127, 0], [-127, 0], [60, -60], [127, 127], [-100, 90]]) {
      const n = decodeOct(x, y);
      expect(Math.hypot(...n)).toBeCloseTo(1, 6);
    }
    expect(decodeOct(0, 0)[2]).toBeCloseTo(1, 6);
  });

  it('warps the reference body to female proportions and back', () => {
    expect(warp([0.1, 1, 0.05], 0)).toEqual([0.1, 1, 0.05]);
    expect(warp([0, 1.7, 0], 1)[1]).toBeCloseTo(1.7 * FEMALE_SCALE, 9);
    expect(widthAt(0.52)).toBeGreaterThan(1);
    expect(widthAt(0.82)).toBeLessThan(0.9);
    expect(widthAt(0)).toBe(1);
  });

  it('explodes from group centres and collapses at zero', () => {
    const groups = groupCenters(parts, manifest.organs);
    const heart = manifest.organs.heart.parts[0];
    expect(explodeOffset(parts[heart], groups[heart], 0)).toEqual([0, 0, 0]);
    const far = explodeOffset(parts[heart], groups[heart], 1);
    expect(Math.hypot(...far)).toBeGreaterThan(0.01);
  });

  it('searches Vietnamese names without diacritics', () => {
    const keys = Object.keys(ORGANS);
    expect(rank(keys, 'tim', k => [ORGANS[k].name])[0]).toBe('heart');
    expect(rank(keys, 'da day', k => [ORGANS[k].name])[0]).toBe('stomach');
    expect(rank(keys, 'hepar', k => [ORGANS[k].latin])).toEqual(['liver']);
    expect(rank(keys, '', k => [ORGANS[k].name])).toEqual([]);
    expect(sideOf('Left femur')).toBe('L');
    expect(sideOf('VH_F_ovary_R')).toBe('R');
    expect(sideOf('Pancreas')).toBe('');
  });
});
