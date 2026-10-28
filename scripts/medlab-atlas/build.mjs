// Builds the compact anatomy atlas for tools/du-an/medlab-body-atlas from two open sources:
//   - BodyParts3D 4.0 (adult male reference, CC BY 4.0) as packaged by ashemag/human-atlas (MIT), commit 1c38bf3
//   - Human Reference Atlas united-female v1.5 (CC BY 4.0) as packaged by ashemag/human-atlas, commit d72b4f6
// See README.md for how to fetch both. Usage: node build.mjs <male-models-dir> <female-models-dir>
import fs from 'node:fs';
import path from 'node:path';
import { MeshoptSimplifier } from 'meshoptimizer';

await MeshoptSimplifier.ready;

const [maleDir, femaleDir] = process.argv.slice(2);
if (!maleDir || !femaleDir) throw new Error('Usage: node build.mjs <male-models-dir> <female-models-dir>');
const out = path.resolve(import.meta.dirname, '../../tools/du-an/medlab-body-atlas/data');
fs.mkdirSync(out, { recursive: true });

const male = JSON.parse(fs.readFileSync(path.join(maleDir, 'atlas.json'), 'utf8'));
const female = JSON.parse(fs.readFileSync(path.join(femaleDir, 'atlas-female.json'), 'utf8'));
const chunkOf = (dir, atlas) => atlas.chunks.map(c => fs.readFileSync(path.join(dir, path.basename(c.url))));
const maleChunks = chunkOf(maleDir, male);
const femaleChunks = chunkOf(femaleDir, female);

const SYSTEMS = ['skeletal', 'muscular', 'cardiac', 'arterial', 'venous', 'nervous', 'sensory', 'respiratory', 'digestive', 'urinary', 'reproductive', 'endocrine', 'lymphatic', 'connective'];
const REGIONS = ['head', 'thorax', 'abdomen', 'arm', 'leg'];
// Vessels and muscles are large and hidden by default, so they ship in a second file loaded after the first view.
const EXTRA = new Set(['muscular', 'arterial', 'venous']);
// Share of source triangles kept per system. Small structures keep at least MIN_TRIANGLES.
const KEEP = { skeletal: 0.3, muscular: 0.12, cardiac: 0.5, arterial: 0.16, venous: 0.16, nervous: 0.3, sensory: 0.25, respiratory: 0.4, digestive: 0.45, urinary: 0.6, reproductive: 0.6, endocrine: 0.6, lymphatic: 0.6, connective: 0.3 };
const MIN_TRIANGLES = 48;
const MAX_ERROR = 0.02;

// The source places a few brain ventricles and the heart walls in a "cardiac" group; reclassify by name.
function systemFor(part) {
  if (part.system === 'cardiac' && /ventricle|foramen/i.test(part.name) && !/(cavity|wall) of|leaflet|cusp/i.test(part.name)) return 'nervous';
  if (part.system === 'sensory' && /choroid plexus/i.test(part.name)) return 'nervous';
  if (part.system === 'sensory' && /lacrimal bone/i.test(part.name)) return 'skeletal';
  if (part.system === 'sensory' && /retinaculum/i.test(part.name)) return 'connective';
  if (/^Hepatovenous segment/.test(part.name)) return 'digestive'; // liver parenchyma segments
  if (/papillary muscle/i.test(part.name)) return 'cardiac';
  return part.system;
}

// Regions from the part centre in the reference pose (metres, y up, +x = body's left, +z = front).
function regionFor(c) {
  const [x, y] = c, ax = Math.abs(x);
  if ((y > 0.95 && y < 1.5 && ax > 0.168) || (y <= 0.95 && ax > 0.2)) return 'arm';
  if (y > 1.445) return 'head';
  if (y > 1.15) return 'thorax';
  if (y > 0.84) return 'abdomen';
  return 'leg';
}

function readPart(chunks, p) {
  const b = chunks[p.chunk];
  return {
    positions: new Float32Array(b.buffer.slice(b.byteOffset + p.positions, b.byteOffset + p.positions + p.vertexCount * 12)),
    normals: new Int16Array(b.buffer.slice(b.byteOffset + p.normals, b.byteOffset + p.normals + p.vertexCount * 6)),
    indices: new Uint32Array(b.buffer.slice(b.byteOffset + p.indices, b.byteOffset + p.indices + p.indexCount * 4)),
  };
}

// HRA meshes duplicate vertices along triangle seams; weld before simplifying (same as ashemag/human-atlas).
function weld(mesh) {
  const map = new Map(), remap = new Uint32Array(mesh.positions.length / 3), wp = [], wn = [];
  for (let i = 0; i < remap.length; i++) {
    const key = `${mesh.positions[i * 3]},${mesh.positions[i * 3 + 1]},${mesh.positions[i * 3 + 2]}`;
    let index = map.get(key);
    if (index === undefined) { index = wp.length / 3; map.set(key, index); wp.push(mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2]); wn.push(0, 0, 0); }
    remap[i] = index;
    for (let a = 0; a < 3; a++) wn[index * 3 + a] += mesh.normals[i * 3 + a];
  }
  for (let i = 0; i < wn.length; i += 3) { const l = Math.hypot(wn[i], wn[i + 1], wn[i + 2]) || 1; for (let a = 0; a < 3; a++) wn[i + a] = Math.round(wn[i + a] / l * 32767); }
  return { positions: new Float32Array(wp), normals: new Int16Array(wn), indices: Uint32Array.from(mesh.indices, i => remap[i]) };
}

function simplify(mesh, keep) {
  const target = Math.min(mesh.indices.length, Math.max(MIN_TRIANGLES * 3, Math.floor(mesh.indices.length * keep / 3) * 3));
  const [indices] = MeshoptSimplifier.simplify(mesh.indices, mesh.positions, 3, target, MAX_ERROR);
  const [remap, count] = MeshoptSimplifier.compactMesh(indices);
  const positions = new Float32Array(count * 3), normals = new Int16Array(count * 3);
  for (let old = 0; old < remap.length; old++) {
    const n = remap[old];
    if (n === 0xffffffff) continue;
    positions.set(mesh.positions.subarray(old * 3, old * 3 + 3), n * 3);
    normals.set(mesh.normals.subarray(old * 3, old * 3 + 3), n * 3);
  }
  return { positions, normals, indices };
}

function boundsOf(list) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const b of list) for (let a = 0; a < 3; a++) { min[a] = Math.min(min[a], b[0][a]); max[a] = Math.max(max[a], b[1][a]); }
  return [min, max];
}
// Axis-wise affine map taking box `from` onto box `to`.
function boxMap(from, to) {
  const s = [0, 1, 2].map(a => (to[1][a] - to[0][a]) / (from[1][a] - from[0][a]));
  const t = [0, 1, 2].map(a => to[0][a] - from[0][a] * s[a]);
  return { s, t };
}
function applyMap(mesh, { s, t }, shiftZ = 0) {
  for (let i = 0; i < mesh.positions.length; i += 3) for (let a = 0; a < 3; a++) mesh.positions[i + a] = mesh.positions[i + a] * s[a] + t[a] + (a === 2 ? shiftZ : 0);
  // Non-uniform scale: transform normals by the inverse scale and renormalise.
  for (let i = 0; i < mesh.normals.length; i += 3) {
    const n = [0, 1, 2].map(a => mesh.normals[i + a] / s[a]), l = Math.hypot(...n) || 1;
    for (let a = 0; a < 3; a++) mesh.normals[i + a] = Math.round(n[a] / l * 32767);
  }
}

const maleById = new Map(male.parts.map(p => [p.id, p]));
const conceptByName = new Map(male.concepts.map(c => [c.name.toLowerCase(), c]));
const conceptById = new Map(male.concepts.map(c => [c.id, c]));
const conceptBounds = id => boundsOf(conceptById.get(id).elements.map(e => maleById.get(e)).filter(Boolean).map(p => p.bounds));

// ---- female and shared HRA structures -------------------------------------------------------------
const femaleById = new Map(female.parts.map(p => [p.id, p]));
const hraSelect = test => female.parts.filter(test);
const HRA_LUNG = hraSelect(p => p.system === 'respiratory' && /bronchopulmonary_segm|hilum/.test(p.id));
const HRA_PELVIC = hraSelect(p => p.system === 'reproductive' || (p.system === 'urinary' && /bladder/.test(p.id)));
const HRA_BREAST = hraSelect(p => /mammary_lobes|lactiferous|nipple|areola|suspensory_ligaments_[LR]$/.test(p.id));
const HRA_PELVIS_BONES = hraSelect(p => /VH_F_(ilium|ischium|pubis)_|VH_F_sacrum/.test(p.id));

// Lungs: BodyParts3D carries the bronchial tree and pulmonary vessels but no lung surface. The HRA bronchopulmonary
// segments are mapped onto the box of the male bronchial tree (padded by 6 mm) so both sexes show lung lobes.
const maleLungs = boundsOf([conceptBounds('FMA7309'), conceptBounds('FMA7310')]);
const lungTarget = [maleLungs[0].map(v => v - 0.006), maleLungs[1].map(v => v + 0.006)];
const lungMap = boxMap(boundsOf(HRA_LUNG.map(p => p.bounds)), lungTarget);
// Pelvic organs: HRA hip bones + sacrum mapped onto the BodyParts3D bony pelvis.
const pelvisMap = boxMap(boundsOf(HRA_PELVIS_BONES.map(p => p.bounds)), conceptBounds('FMA16580'));
// Breasts: height follows the thoracic spine (T1 top to T12 bottom in both atlases), width the lung map; depth keeps
// the HRA scale and puts the back of the glands 10 mm behind the front of the sternum (the lateral chest wall is
// further back than the sternum).
const hraSpine = boundsOf(hraSelect(p => /VH_F_thoracic_vertebra_\d+$/.test(p.id)).map(p => p.bounds));
const maleSpine = conceptBounds('FMA9139');
const spineScale = (maleSpine[1][1] - maleSpine[0][1]) / (hraSpine[1][1] - hraSpine[0][1]);
const breastMap = { s: [lungMap.s[0], spineScale, 1], t: [lungMap.t[0], maleSpine[0][1] - hraSpine[0][1] * spineScale, 0] };
const ribFront = conceptBounds('FMA7480')[1][2];
const breastBox = boundsOf(HRA_BREAST.map(p => p.bounds));
const breastShift = ribFront - 0.01 - breastBox[0][2];

// ---- assemble -------------------------------------------------------------------------------------
const items = [];
for (const p of male.parts) {
  if (p.system === 'integumentary') continue;
  const system = systemFor(p);
  const maleOnly = system === 'reproductive' || ['FJ3148', 'FJ3149'].includes(p.id); // urethra, bladder
  items.push({ id: p.id, name: p.name, fma: p.conceptId, system, sex: maleOnly ? 1 : 0, mesh: simplify(weld(readPart(maleChunks, p)), KEEP[system]) });
}
function addHra(list, sex, transform, system, keep) {
  for (const p of list) {
    const mesh = simplify(weld(readPart(femaleChunks, p)), keep ?? KEEP[system ?? p.system]);
    transform(mesh);
    items.push({ id: p.id, name: p.name.charAt(0).toUpperCase() + p.name.slice(1), fma: '', system: system ?? p.system, sex, mesh, source: 'HRA' });
  }
}
addHra(HRA_LUNG, 0, m => applyMap(m, lungMap));
addHra(HRA_PELVIC, 2, m => applyMap(m, pelvisMap));
addHra(HRA_BREAST, 2, m => applyMap(m, breastMap, breastShift), 'reproductive', 0.15);

// Quantise positions to int16 over the global box; normals to octahedral int8 pairs.
const all = boundsOf(items.map(it => {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < it.mesh.positions.length; i += 3) for (let a = 0; a < 3; a++) { min[a] = Math.min(min[a], it.mesh.positions[i + a]); max[a] = Math.max(max[a], it.mesh.positions[i + a]); }
  it.bounds = [min, max];
  return it.bounds;
}));
const qMin = all[0].map(v => v - 0.001), qMax = all[1].map(v => v + 0.001);
const q = (v, a) => Math.round((v - qMin[a]) / (qMax[a] - qMin[a]) * 65535) - 32768;
function oct(nx, ny, nz) {
  const l = Math.abs(nx) + Math.abs(ny) + Math.abs(nz) || 1;
  let x = nx / l, y = ny / l;
  if (nz < 0) { const ox = x; x = (1 - Math.abs(y)) * Math.sign(ox || 1); y = (1 - Math.abs(ox)) * Math.sign(y || 1); }
  return [Math.round(x * 127), Math.round(y * 127)];
}

// Organs opened from the sidebar should not wait for the second file: their vessels and muscles ship in the first.
const CORE_IDS = new Set(['FMA7088', 'FMA7197', 'FMA54449', 'FMA54450', 'FMA13295'].flatMap(id => conceptById.get(id).elements));
const files = [[], []], sizes = [0, 0];
const parts = [];
for (const it of items) {
  const chunk = EXTRA.has(it.system) && !CORE_IDS.has(it.id) ? 1 : 0;
  const count = it.mesh.positions.length / 3;
  if (count > 65535) throw new Error(`${it.id} has too many vertices`);
  const vertex = Buffer.alloc(count * 8);
  for (let i = 0; i < count; i++) {
    for (let a = 0; a < 3; a++) vertex.writeInt16LE(q(it.mesh.positions[i * 3 + a], a), i * 8 + a * 2);
    const [ox, oy] = oct(it.mesh.normals[i * 3], it.mesh.normals[i * 3 + 1], it.mesh.normals[i * 3 + 2]);
    vertex.writeInt8(ox, i * 8 + 6); vertex.writeInt8(oy, i * 8 + 7);
  }
  const index = Buffer.from(Uint16Array.from(it.mesh.indices).buffer);
  const vertexOffset = sizes[chunk];
  files[chunk].push(vertex); sizes[chunk] += vertex.length;
  const indexOffset = sizes[chunk];
  files[chunk].push(index); sizes[chunk] += index.length;
  if (sizes[chunk] % 4) { const pad = Buffer.alloc(4 - sizes[chunk] % 4); files[chunk].push(pad); sizes[chunk] += pad.length; }
  const center = [0, 1, 2].map(a => (it.bounds[0][a] + it.bounds[1][a]) / 2);
  const half = [0, 1, 2].map(a => (it.bounds[1][a] - it.bounds[0][a]) / 2);
  const r = v => Math.round(v * 10000) / 10000;
  parts.push({
    row: [it.id, it.name, it.fma, SYSTEMS.indexOf(it.system), REGIONS.indexOf(regionFor(center)), it.sex, chunk, vertexOffset, count, indexOffset, it.mesh.indices.length, ...center.map(r), ...half.map(r)],
    item: it,
  });
}
for (const p of parts) if (p.row[3] < 0) throw new Error(`Unknown system for ${p.row[0]}`);

const indexOf = new Map(parts.map((p, i) => [p.item.id, i]));
const concept = id => conceptById.get(id).elements.filter(e => indexOf.has(e)).map(e => indexOf.get(e));
const byIds = list => list.map(p => indexOf.get(p.id));
const hraIds = test => parts.map((p, i) => [p, i]).filter(([p]) => p.item.source === 'HRA' && test(p.item.id)).map(([, i]) => i);
const unique = list => [...new Set(list)];

// Organs shown in the sidebar. `concepts` lists the BodyParts3D FMA concepts used (empty for HRA-only organs);
// Vietnamese texts live in the tool (organs.ts).
const organs = {
  heart: { concepts: ['FMA7088'], parts: concept('FMA7088') },
  brain: { concepts: ['FMA50801'], parts: concept('FMA50801') },
  lungs: { concepts: ['FMA7309', 'FMA7310'], parts: unique([...concept('FMA7309'), ...concept('FMA7310'), ...byIds(HRA_LUNG)]) },
  liver: { concepts: ['FMA7197'], parts: concept('FMA7197') },
  kidneys: { concepts: ['FMA7203'], parts: concept('FMA7203') },
  stomach: { concepts: ['FMA7148'], parts: concept('FMA7148') },
  'small-intestine': { concepts: ['FMA7200'], parts: concept('FMA7200') },
  'large-intestine': { concepts: ['FMA7201'], parts: concept('FMA7201') },
  pancreas: { concepts: ['FMA7198'], parts: concept('FMA7198') },
  spleen: { concepts: ['FMA7196'], parts: concept('FMA7196') },
  bladder: { concepts: ['FMA15900'], parts: unique([...concept('FMA15900'), ...hraIds(id => /bladder/.test(id))]) },
  gallbladder: { concepts: ['FMA7202'], parts: concept('FMA7202') },
  esophagus: { concepts: ['FMA7131'], parts: concept('FMA7131') },
  trachea: { concepts: ['FMA7394'], parts: concept('FMA7394') },
  ureters: { concepts: ['FMA9704'], parts: concept('FMA9704') },
  thymus: { concepts: ['FMA9607'], parts: concept('FMA9607') },
  adrenals: { concepts: ['FMA9604'], parts: concept('FMA9604') },
  eyes: { concepts: ['FMA54449', 'FMA54450'], parts: unique([...concept('FMA54449'), ...concept('FMA54450')]) },
  skull: { concepts: ['FMA46565'], parts: concept('FMA46565') },
  spine: { concepts: ['FMA13478'], parts: concept('FMA13478') },
  ribcage: { concepts: ['FMA7480'], parts: concept('FMA7480') },
  pelvis: { concepts: ['FMA16580'], parts: concept('FMA16580') },
  diaphragm: { concepts: ['FMA13295'], parts: concept('FMA13295') },
  prostate: { concepts: ['FMA9600'], parts: concept('FMA9600') },
  testes: { concepts: ['FMA7210'], parts: concept('FMA7210') },
  uterus: { concepts: [], parts: hraIds(id => /uterus|cervi|cornua|uterine_segment|cervical_os/.test(id) && !/ligament|tube/.test(id)) },
  ovaries: { concepts: [], parts: hraIds(id => /ovary$/.test(id)) },
  'uterine-tubes': { concepts: [], parts: hraIds(id => /uterine_tube|fallopian/.test(id) && !/ligament|ostium/.test(id)) },
  vagina: { concepts: [], parts: hraIds(id => /vagina/.test(id)) },
  breasts: { concepts: [], parts: byIds(HRA_BREAST) },
};
for (const [key, organ] of Object.entries(organs)) if (!organ.parts.length) throw new Error(`Organ ${key} is empty`);

const atlas = {
  version: 1,
  source: { male: 'BodyParts3D 4.0 via ashemag/human-atlas@1c38bf3', female: 'HRA united-female v1.5 via ashemag/human-atlas@d72b4f6' },
  systems: SYSTEMS,
  regions: REGIONS,
  quantization: { min: qMin, max: qMax },
  files: ['core.bin', 'extra.bin'].map((file, i) => ({ file, bytes: sizes[i] })),
  columns: ['id', 'name', 'fma', 'system', 'region', 'sex', 'file', 'vertexOffset', 'vertexCount', 'indexOffset', 'indexCount', 'cx', 'cy', 'cz', 'hx', 'hy', 'hz'],
  parts: parts.map(p => p.row),
  organs,
  alignment: {
    lungs: { scale: lungMap.s, translate: lungMap.t },
    pelvis: { scale: pelvisMap.s, translate: pelvisMap.t },
    breasts: { scale: breastMap.s, translate: breastMap.t, depthShift: breastShift },
  },
  triangles: parts.reduce((s, p) => s + p.row[10] / 3, 0),
};
fs.writeFileSync(path.join(out, 'atlas.json'), JSON.stringify(atlas));
files.forEach((list, i) => fs.writeFileSync(path.join(out, atlas.files[i].file), Buffer.concat(list)));

const count = (test) => parts.filter(test).length;
console.log(`parts ${parts.length} (male-only ${count(p => p.row[5] === 1)}, female-only ${count(p => p.row[5] === 2)})`);
console.log(`triangles ${atlas.triangles}; core ${(sizes[0] / 1e6).toFixed(2)} MB, extra ${(sizes[1] / 1e6).toFixed(2)} MB`);
console.log('regions', REGIONS.map((r, i) => `${r}:${count(p => p.row[4] === i)}`).join(' '));
console.log('organs', Object.entries(organs).map(([k, o]) => `${k}:${o.parts.length}`).join(' '));
