// Atlas data model: the manifest written by scripts/medlab-atlas/build.mjs and the pure helpers the viewer and the unit
// tests share (vertex decoding, the female proportion warp, explode offsets, search).

export type SystemId = 'skeletal' | 'muscular' | 'cardiac' | 'arterial' | 'venous' | 'nervous' | 'sensory' | 'respiratory' | 'digestive' | 'urinary' | 'reproductive' | 'endocrine' | 'lymphatic' | 'connective';
export type RegionId = 'head' | 'thorax' | 'abdomen' | 'arm' | 'leg';
export type Sex = 'male' | 'female';
export type Vec3 = [number, number, number];

export interface Manifest {
  version: number;
  systems: SystemId[];
  regions: RegionId[];
  quantization: { min: Vec3; max: Vec3 };
  files: { file: string; bytes: number }[];
  columns: string[];
  parts: (string | number)[][];
  organs: Record<string, { concepts: string[]; parts: number[] }>;
  triangles: number;
}

export interface Part {
  index: number;
  id: string;
  name: string;
  fma: string;
  system: SystemId;
  region: RegionId;
  /** 0 = both atlases, 1 = male only, 2 = female only */
  sex: 0 | 1 | 2;
  file: number;
  vertexOffset: number;
  vertexCount: number;
  indexOffset: number;
  indexCount: number;
  center: Vec3;
  half: Vec3;
  source: 'BodyParts3D' | 'HRA';
}

export function parseParts(manifest: Manifest): Part[] {
  return manifest.parts.map((row, index) => {
    const [id, name, fma, system, region, sex, file, vertexOffset, vertexCount, indexOffset, indexCount, cx, cy, cz, hx, hy, hz] = row as [string, string, string, ...number[]];
    return {
      index, id, name, fma,
      system: manifest.systems[system], region: manifest.regions[region], sex: sex as 0 | 1 | 2,
      file, vertexOffset, vertexCount, indexOffset, indexCount,
      center: [cx, cy, cz], half: [hx, hy, hz],
      source: id.startsWith('FJ') ? 'BodyParts3D' : 'HRA',
    };
  });
}

export const inAtlas = (part: Part, sex: Sex) => part.sex === 0 || (sex === 'male' ? part.sex === 1 : part.sex === 2);

/** Positions are int16 over the manifest box; this mirrors the vertex shader. */
export function dequantize(q: number, axis: number, manifest: Pick<Manifest, 'quantization'>) {
  const { min, max } = manifest.quantization;
  return min[axis] + (q + 32768) / 65535 * (max[axis] - min[axis]);
}

/** Octahedral normal decoding (two int8 values), as in the vertex shader. */
export function decodeOct(ox: number, oy: number): Vec3 {
  let x = Math.max(-1, ox / 127), y = Math.max(-1, oy / 127);
  const z = 1 - Math.abs(x) - Math.abs(y);
  if (z < 0) { const px = x; x = (1 - Math.abs(y)) * Math.sign(px || 1); y = (1 - Math.abs(px)) * Math.sign(y || 1); }
  const l = Math.hypot(x, y, z) || 1;
  return [x / l, y / l, z / l];
}

// ---- female proportions -------------------------------------------------------------------------------------------
// The female atlas reuses the male reference skeleton and organs with a smooth proportion warp: 93% stature, narrower
// shoulders and chest, wider pelvis. These are teaching approximations of typical adult differences, not measurements
// of one body, and the article says so. The vertex shader implements the same function (keep both in sync).
export const STATURE = 1.714;
export const FEMALE_SCALE = 0.93;
export const FEMALE_DEPTH = 0.96;
/** Lateral (x) scale as a function of height / stature. */
export const FEMALE_WIDTH: [number, number][] = [[0, 1], [0.28, 1], [0.42, 1.04], [0.52, 1.075], [0.6, 0.98], [0.66, 0.94], [0.74, 0.93], [0.82, 0.88], [0.88, 0.93], [1, 0.95]];

export function widthAt(yN: number) {
  const k = FEMALE_WIDTH;
  if (yN <= k[0][0]) return k[0][1];
  for (let i = 1; i < k.length; i++) if (yN <= k[i][0]) {
    const t = (yN - k[i - 1][0]) / (k[i][0] - k[i - 1][0]), s = t * t * (3 - 2 * t);
    return k[i - 1][1] + (k[i][1] - k[i - 1][1]) * s;
  }
  return k[k.length - 1][1];
}

export function warp(p: Vec3, amount: number): Vec3 {
  const sx = widthAt(p[1] / STATURE);
  const female: Vec3 = [p[0] * sx * FEMALE_SCALE, p[1] * FEMALE_SCALE, p[2] * FEMALE_DEPTH * FEMALE_SCALE];
  return [p[0] + (female[0] - p[0]) * amount, p[1] + (female[1] - p[1]) * amount, p[2] + (female[2] - p[2]) * amount];
}

// ---- exploded view ------------------------------------------------------------------------------------------------
/** Group centre of every part: its organ when it belongs to one, otherwise its system within its region. */
export function groupCenters(parts: Part[], organs: Record<string, { parts: number[] }>) {
  const groupOf = new Map<number, string>();
  for (const [key, organ] of Object.entries(organs)) for (const i of organ.parts) if (!groupOf.has(i)) groupOf.set(i, `organ:${key}`);
  const sums = new Map<string, { s: Vec3; n: number }>();
  const keys = parts.map(p => groupOf.get(p.index) ?? `${p.system}:${p.region}`);
  parts.forEach((p, i) => {
    const entry = sums.get(keys[i]) ?? { s: [0, 0, 0], n: 0 };
    for (let a = 0; a < 3; a++) entry.s[a] += p.center[a];
    entry.n++;
    sums.set(keys[i], entry);
  });
  return keys.map(k => { const { s, n } = sums.get(k)!; return [s[0] / n, s[1] / n, s[2] / n] as Vec3; });
}

/** Offset for an explode amount in [0, 1]: parts spread from their group centre and groups spread from the body axis. */
export function explodeOffset(part: Part, group: Vec3, amount: number): Vec3 {
  const local = [0, 1, 2].map(a => (part.center[a] - group[a]) * 1.25 * amount);
  const body: Vec3 = [group[0] * 1.1 * amount, (group[1] - 0.95) * 0.35 * amount, group[2] * 2.2 * amount];
  return [local[0] + body[0], local[1] + body[1], local[2] + body[2]];
}

// ---- search -------------------------------------------------------------------------------------------------------
export const fold = (s: string) => s.normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/đ/g, 'd').replace(/Đ/g, 'D').toLowerCase().trim();

/** Rank `items` by how well `query` matches their fields: prefix of a word > substring. */
export function rank<T>(items: T[], query: string, fields: (item: T) => string[], limit = 30): T[] {
  const q = fold(query);
  if (!q) return [];
  const scored: [number, T][] = [];
  for (const item of items) {
    let best = 0;
    for (const field of fields(item)) {
      const f = fold(field);
      if (f === q) best = Math.max(best, 4);
      else if (f.startsWith(q)) best = Math.max(best, 3);
      else if (f.includes(` ${q}`)) best = Math.max(best, 2);
      else if (f.includes(q)) best = Math.max(best, 1);
    }
    if (best) scored.push([best, item]);
  }
  return scored.sort((a, b) => b[0] - a[0]).slice(0, limit).map(([, item]) => item);
}

/** Left/right markers as in the source names: "Left femur" → "Left femur (L)". */
export function sideOf(name: string): 'L' | 'R' | '' {
  if (/\bleft\b|_L$/i.test(name)) return 'L';
  if (/\bright\b|_R$/i.test(name)) return 'R';
  return '';
}
