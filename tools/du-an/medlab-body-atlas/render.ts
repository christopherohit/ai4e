// MedLab body atlas: loads the atlas, wires the sidebar (search, organ cards, regions, systems, organs), the stage
// (orbit camera, picking, labels, gizmo, views, sex toggle, exploded view), the detail panel, the guided tour, the quiz
// and the attribution panel.
import manifestUrl from './data/atlas.json?url';
import coreUrl from './data/core.bin?url';
import extraUrl from './data/extra.bin?url';
import { explodeOffset, groupCenters, inAtlas, parseParts, rank, sideOf, warp, type Manifest, type RegionId, type Sex, type SystemId, type Vec3 } from './atlas';
import { AtlasRenderer, MODE, OrbitCamera, project } from './gl';
import { DEFAULT_SYSTEMS, ORGANS, QUIZ, REGION_INFO, SYSTEM_INFO, TOUR } from './organs';

type Selection = { kind: 'organ'; key: string } | { kind: 'part'; index: number } | null;

const ORGAN_COLORS: Record<string, string> = {
  heart: '#d9544c', brain: '#ebb7aa', lungs: '#f0a3ad', liver: '#a84e3e', kidneys: '#b9624b', stomach: '#e3a07e', 'small-intestine': '#ecb58d',
  'large-intestine': '#d79670', pancreas: '#e9c27b', spleen: '#9c4b5d', bladder: '#e2b06f', gallbladder: '#7fb069', thymus: '#c9a0b8', adrenals: '#dba75c',
  eyes: '#e3ecf2', uterus: '#e59aab', ovaries: '#f0b8c4', 'uterine-tubes': '#eaa6b6', vagina: '#d98a9c', breasts: '#f3c6b4', prostate: '#d7a08e', testes: '#e2b4a4',
};
const CARD_ORGANS = Object.keys(ORGANS).filter(k => ORGANS[k].card);
const LABEL_ORGANS = ['brain', 'skull', 'eyes', 'trachea', 'lungs', 'heart', 'liver', 'stomach', 'spleen', 'pancreas', 'kidneys', 'small-intestine', 'large-intestine', 'bladder', 'uterus', 'prostate', 'pelvis', 'spine'];
const VIEWS = { front: { yaw: 0, pitch: 0.04 }, back: { yaw: Math.PI, pitch: 0.04 }, side: { yaw: -Math.PI / 2, pitch: 0.04 }, perspective: { yaw: 0.55, pitch: 0.14 } } as const;
const REGION_ORDER: RegionId[] = ['head', 'thorax', 'abdomen', 'arm', 'leg'];

const hex = (c: string): [number, number, number] => [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
const esc = (s: string) => s.replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]!));
const nf = new Intl.NumberFormat('vi-VN');

function jitter(index: number, rgb: [number, number, number]): [number, number, number] {
  let h = Math.imul(index ^ 0x9e3779b9, 0x85ebca6b);
  h ^= h >>> 13;
  const k = 0.9 + ((h >>> 0) % 1000) / 1000 * 0.18;
  return rgb.map(v => Math.min(255, Math.round(v * k))) as [number, number, number];
}

async function fetchWithProgress(url: string, expected: number, onProgress: (f: number) => void) {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`Không tải được ${url}`);
  const reader = response.body.getReader(), chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(Math.min(1, received / expected));
  }
  const out = new Uint8Array(received);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  if (out.byteLength !== expected) throw new Error('Tệp giải phẫu bị thiếu dữ liệu, hãy tải lại trang.');
  return out.buffer;
}

export function initBodyAtlas() {
  const root = document.querySelector<HTMLElement>('[data-medlab]');
  if (!root || root.dataset.ready) return;
  root.dataset.ready = '1';
  const $ = <T extends Element = HTMLElement>(s: string) => root.querySelector<T>(s)!;
  const $$ = <T extends Element = HTMLElement>(s: string) => [...root.querySelectorAll<T>(s)];
  start(root, $, $$).catch(error => {
    console.error(error);
    $('[data-loading-text]').textContent = error instanceof Error ? error.message : String(error);
    root.querySelector('[data-loading]')?.classList.add('is-error');
    root.setAttribute('aria-busy', 'false');
  });
}

async function start(root: HTMLElement, $: <T extends Element = HTMLElement>(s: string) => T, $$: <T extends Element = HTMLElement>(s: string) => T[]) {
  const manifest = await (await fetch(manifestUrl)).json() as Manifest;
  const parts = parseParts(manifest);
  const organs = manifest.organs;
  const groups = groupCenters(parts, organs);
  const organOf = new Map<number, string>();
  for (const [key, organ] of Object.entries(organs)) for (const i of organ.parts) if (!organOf.has(i)) organOf.set(i, key);
  const baseColor = parts.map(p => {
    const organ = organOf.get(p.index);
    const color = organ && ORGAN_COLORS[organ] ? ORGAN_COLORS[organ] : SYSTEM_INFO[p.system].color;
    return jitter(p.index, hex(color));
  });

  const canvas = $<HTMLCanvasElement>('[data-canvas]');
  const stage = $('[data-stage]');
  const renderer = new AtlasRenderer(canvas, manifest, parts.length);
  const camera = new OrbitCamera();
  const loaded = [false, false];

  const state = {
    sex: 'male' as Sex,
    female: 0,
    femaleGoal: 0,
    systems: new Set<SystemId>(DEFAULT_SYSTEMS),
    region: null as RegionId | null,
    selection: null as Selection,
    isolate: false,
    explode: 0,
    labels: false,
    pinned: new Set<string>(),
    hover: -1,
    quiz: null as null | { i: number; score: number; answered: boolean },
    tour: -1,
  };
  const modes = new Uint8Array(parts.length);
  const offsets: Vec3[] = parts.map(() => [0, 0, 0]);

  // ---- per-part state -----------------------------------------------------------------------------------------------
  const visibleSex = () => (state.female > 0.5 ? 'female' : 'male') as Sex;
  const organParts = (key: string) => organs[key].parts.filter(i => inAtlas(parts[i], state.sex));
  const selectedParts = (): number[] => {
    const s = state.selection;
    if (!s) return [];
    return s.kind === 'organ' ? organParts(s.key) : [s.index];
  };
  function updateParts() {
    const sex = visibleSex(), selected = new Set(selectedParts());
    let count = 0;
    for (const p of parts) {
      const base = loaded[p.file] && inAtlas(p, sex) && state.systems.has(p.system);
      let mode: number = MODE.hidden;
      if (selected.size) mode = selected.has(p.index) && loaded[p.file] && inAtlas(p, sex) ? MODE.focus : state.isolate || !base ? MODE.hidden : MODE.ghost;
      else if (state.region) mode = !base ? MODE.hidden : p.region === state.region ? MODE.solid : MODE.ghost;
      else mode = base ? MODE.solid : MODE.hidden;
      modes[p.index] = mode;
      if (mode === MODE.solid || mode === MODE.focus) count++;
      offsets[p.index] = state.explode ? explodeOffset(p, groups[p.index], state.explode) : [0, 0, 0];
      renderer.setPart(p.index, offsets[p.index], mode, baseColor[p.index]);
    }
    renderer.commit();
    $('[data-visible-count]').textContent = nf.format(count);
    invalidate();
  }

  // Where a part is drawn now (explode offset + female warp), for labels and camera fitting.
  const placed = (i: number, blend = state.female): Vec3 => { const w = warp(parts[i].center, blend); const o = offsets[i]; return [w[0] + o[0], w[1] + o[1], w[2] + o[2]]; };
  function sphereOf(list: number[], blend = state.female) {
    if (!list.length) return { center: [0, 0.86, 0] as Vec3, radius: 0.95 };
    const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
    for (const i of list) {
      const c = placed(i, blend), h = parts[i].half;
      for (let a = 0; a < 3; a++) { min[a] = Math.min(min[a], c[a] - h[a]); max[a] = Math.max(max[a], c[a] + h[a]); }
    }
    const center: Vec3 = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    return { center, radius: Math.max(0.03, Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2) };
  }
  const aspect = () => canvas.width / Math.max(1, canvas.height);
  // Fit a set of parts into the band of the stage left free by the top chips and the bottom bar.
  function fitView(list: number[], yaw: number, pitch: number) {
    const { center, radius } = sphereOf(list, state.femaleGoal);
    const h = Math.max(1, stage.clientHeight), w = Math.max(1, stage.clientWidth);
    const top = 50, bottom = $('.ml-bottom').offsetHeight + 24, usable = Math.max(0.35, (h - top - bottom) / h);
    // On medium widths the detail panel floats over the right of the stage.
    const info = $('.ml-info'), covered = state.selection && getComputedStyle(info).position === 'absolute' ? info.offsetWidth : 0;
    const usableW = Math.max(0.35, (w - covered) / w);
    const t = Math.tan(camera.fov / 2);
    const distance = Math.max(radius / (usable * t), radius / (t * usableW * w / h)) * 1.02;
    const px = 2 * distance * t / h, shift = (bottom - top) / 2 * px, side = covered / 2 * px;
    const up: Vec3 = [-Math.sin(yaw) * Math.sin(pitch), Math.cos(pitch), -Math.cos(yaw) * Math.sin(pitch)];
    const right: Vec3 = [Math.cos(yaw), 0, -Math.sin(yaw)];
    return { target: [0, 1, 2].map(a => center[a] - up[a] * shift + right[a] * side) as Vec3, distance };
  }
  function frame(list: number[], view?: { yaw: number; pitch: number }, room = 1) {
    const yaw = view?.yaw ?? camera.yaw, pitch = view?.pitch ?? camera.pitch;
    const fit = fitView(list, yaw, pitch);
    camera.animateTo({ target: fit.target, distance: fit.distance * room, yaw, pitch });
    invalidate();
  }
  const visibleList = () => parts.filter(p => modes[p.index] === MODE.solid || modes[p.index] === MODE.focus).map(p => p.index);
  const frameBody = (view?: { yaw: number; pitch: number }) => frame(parts.filter(p => inAtlas(p, visibleSex()) && p.system === 'skeletal').map(p => p.index), view);

  // ---- render loop --------------------------------------------------------------------------------------------------
  let dirty = true, raf = 0, last = performance.now(), pulseUntil = 0;
  function invalidate() { dirty = true; if (!raf) raf = requestAnimationFrame(loop); }
  function resize() {
    const r = stage.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; invalidate(); }
  }
  function loop(now: number) {
    raf = 0;
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    let moving = camera.step(dt);
    if (state.female !== state.femaleGoal) {
      const before = visibleSex();
      state.female += Math.sign(state.femaleGoal - state.female) * Math.min(Math.abs(state.femaleGoal - state.female), dt / 0.7);
      if (visibleSex() !== before) updateParts();
      moving = true;
    }
    const pulsing = now < pulseUntil;
    if (dirty || moving || pulsing) {
      dirty = false;
      const vp = camera.matrix(aspect());
      const pulse = pulsing ? 0.5 + 0.5 * Math.sin(now / 140) : 0;
      renderer.render(vp, camera.eye(), state.female, { ghosts: modes.some(m => m === MODE.ghost), pulse, hover: state.hover + 1 });
      overlay(vp);
      gizmo();
    }
    if (moving || pulsing) raf = requestAnimationFrame(loop);
  }

  // ---- overlays: selection leader, smart labels, gizmo ---------------------------------------------------------------
  const leaders = $<SVGSVGElement>('[data-leaders]');
  const labelLayer = $('[data-labels-layer]');
  function overlay(vp: Float32Array) {
    const w = stage.clientWidth, h = stage.clientHeight;
    const items: { key: string; text: string; sub?: string; x: number; y: number; strong: boolean; reach: number }[] = [];
    const add = (key: string, list: number[], text: string, sub: string | undefined, strong: boolean) => {
      const shown = list.filter(i => modes[i] === MODE.solid || modes[i] === MODE.focus);
      if (!shown.length) return;
      const sphere = sphereOf(shown), p = project(vp, sphere.center, w, h);
      const right: Vec3 = [Math.cos(camera.yaw), 0, -Math.sin(camera.yaw)];
      const q = project(vp, [sphere.center[0] + right[0] * sphere.radius * 0.7, sphere.center[1], sphere.center[2] + right[2] * sphere.radius * 0.7], w, h);
      if (!p.behind && p.x > -20 && p.x < w + 20 && p.y > -20 && p.y < h + 20) items.push({ key, text, sub, x: p.x, y: p.y, strong, reach: Math.abs(q.x - p.x) });
    };
    const s = state.selection;
    if (s?.kind === 'organ') add(`o:${s.key}`, organParts(s.key), ORGANS[s.key].name, ORGANS[s.key].latin, true);
    if (s?.kind === 'part') add(`p:${s.index}`, [s.index], parts[s.index].name, organOf.has(s.index) ? ORGANS[organOf.get(s.index)!]?.name : undefined, true);
    for (const key of state.pinned) {
      if (items.some(it => it.key === key)) continue;
      if (key.startsWith('o:')) add(key, organParts(key.slice(2)), ORGANS[key.slice(2)].name, undefined, false);
      else add(key, [Number(key.slice(2))], parts[Number(key.slice(2))].name, undefined, false);
    }
    if (state.labels && !s) for (const key of LABEL_ORGANS) if (inAtlasOrgan(key) && !items.some(it => it.key === `o:${key}`)) add(`o:${key}`, organParts(key), ORGANS[key].name, undefined, false);
    // Lay labels out in two columns beside the body, sorted by height, with a minimum gap.
    const cx = w / 2, gap = 30, left = items.filter(it => it.x < cx).sort((a, b) => a.y - b.y), right = items.filter(it => it.x >= cx).sort((a, b) => a.y - b.y);
    const place = (list: typeof items, side: -1 | 1) => {
      let y = 40;
      const edge = side < 0 ? Math.min(...list.map(it => it.x - it.reach), cx) - 40 : Math.max(...list.map(it => it.x + it.reach), cx) + 40;
      return list.map(it => { y = Math.max(y, Math.min(h - 90, it.y)); const out = { ...it, lx: Math.max(120, Math.min(w - 120, edge)), ly: y }; y += gap; return out; });
    };
    const laid = [...place(left, -1), ...place(right, 1)];
    leaders.setAttribute('viewBox', `0 0 ${w} ${h}`);
    leaders.innerHTML = laid.map(l => `<line x1="${l.x.toFixed(1)}" y1="${l.y.toFixed(1)}" x2="${l.lx.toFixed(1)}" y2="${l.ly.toFixed(1)}" class="${l.strong ? 'strong' : ''}"/><circle cx="${l.x.toFixed(1)}" cy="${l.y.toFixed(1)}" r="${l.strong ? 4 : 2.6}" class="${l.strong ? 'strong' : ''}"/>`).join('');
    labelLayer.innerHTML = laid.map(l => `<span class="ml-label${l.strong ? ' strong' : ''}${l.lx < cx ? ' left' : ''}" style="left:${l.lx.toFixed(1)}px;top:${l.ly.toFixed(1)}px">${esc(l.text)}${l.sub ? `<small>${esc(l.sub)}</small>` : ''}</span>`).join('');
  }
  const inAtlasOrgan = (key: string) => organParts(key).length > 0;

  const gizmoCanvas = $<HTMLCanvasElement>('[data-gizmo]');
  function gizmo() {
    const g = gizmoCanvas.getContext('2d')!, dpr = Math.min(devicePixelRatio || 1, 2), size = 84;
    if (gizmoCanvas.width !== size * dpr) { gizmoCanvas.width = gizmoCanvas.height = size * dpr; }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size, size);
    const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw), cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
    // Camera basis: right = (cos yaw, 0, -sin yaw); up = (-sin yaw sin pitch, cos pitch, -cos yaw sin pitch).
    const toScreen = (v: Vec3) => ({ x: v[0] * cy - v[2] * sy, y: -(-v[0] * sy * sp + v[1] * cp - v[2] * cy * sp), z: v[0] * sy * cp + v[1] * sp + v[2] * cy * cp });
    const axes: [string, Vec3, string][] = [['S', [0, 1, 0], '#7af0a0'], ['I', [0, -1, 0], '#4f8a64'], ['L', [1, 0, 0], '#ff7a8a'], ['R', [-1, 0, 0], '#8a4752'], ['A', [0, 0, 1], '#6cc8ff'], ['P', [0, 0, -1], '#3d6b8c']];
    const c = size / 2, r = 27;
    g.beginPath(); g.arc(c, c, 38, 0, Math.PI * 2); g.fillStyle = 'rgba(10,16,34,0.55)'; g.fill(); g.strokeStyle = 'rgba(255,255,255,0.1)'; g.stroke();
    const list = axes.map(([t, v, col]) => ({ t, col, ...toScreen(v) })).sort((a, b) => a.z - b.z);
    for (const a of list) {
      const x = c + a.x * r, y = c + a.y * r, front = a.z >= -0.01;
      g.strokeStyle = a.col; g.globalAlpha = front ? 0.9 : 0.35; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(c, c); g.lineTo(x, y); g.stroke();
      g.beginPath(); g.arc(x, y, 8, 0, Math.PI * 2); g.fillStyle = front ? a.col : 'rgba(20,28,50,0.9)'; g.fill();
      g.fillStyle = front ? '#081022' : a.col; g.font = '700 9px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(a.t, x, y + 0.5);
    }
    g.globalAlpha = 1;
  }

  // ---- pointer: orbit, pan, zoom, pick, hover -------------------------------------------------------------------------
  const pointers = new Map<number, { x: number; y: number }>();
  let downAt: { x: number; y: number; t: number } | null = null, moved = 0, pinch = 0;
  const tooltip = $('[data-tooltip]');
  canvas.addEventListener('pointerdown', e => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 1) { downAt = { x: e.clientX, y: e.clientY, t: performance.now() }; moved = 0; }
    if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = Math.hypot(a.x - b.x, a.y - b.y); }
    camera.stop();
  });
  canvas.addEventListener('pointermove', e => {
    const prev = pointers.get(e.pointerId);
    if (!prev) { hoverAt(e); return; }
    const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    moved += Math.abs(dx) + Math.abs(dy);
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()], d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch) camera.distance = Math.max(0.08, Math.min(8, camera.distance * pinch / d));
      pinch = d;
    } else if (e.buttons === 2 || e.shiftKey) {
      const k = camera.distance * 0.0012, cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
      camera.target[0] -= dx * k * cy; camera.target[2] += dx * k * sy; camera.target[1] += dy * k;
    } else {
      camera.yaw -= dx * 0.008;
      camera.pitch = Math.max(-1.35, Math.min(1.35, camera.pitch + dy * 0.006));
    }
    tooltip.hidden = true;
    invalidate();
  });
  const end = (e: PointerEvent) => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch = 0;
    if (pointers.size === 0 && downAt && moved < 6 && performance.now() - downAt.t < 600) tap(e);
    if (pointers.size === 0) downAt = null;
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', e => { pointers.delete(e.pointerId); downAt = null; });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('pointerleave', () => { if (state.hover >= 0) { state.hover = -1; tooltip.hidden = true; invalidate(); } });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    camera.stop();
    camera.distance = Math.max(0.08, Math.min(8, camera.distance * Math.exp(e.deltaY * 0.0012)));
    invalidate();
  }, { passive: false });

  function pickAt(e: { clientX: number; clientY: number }) {
    const r = canvas.getBoundingClientRect(), k = canvas.width / r.width;
    const vp = camera.matrix(aspect());
    return renderer.pick(vp, camera.eye(), state.female, (e.clientX - r.left) * k, (e.clientY - r.top) * k);
  }
  let hoverPending = false;
  function hoverAt(e: PointerEvent) {
    if (e.pointerType !== 'mouse' || hoverPending) return;
    hoverPending = true;
    requestAnimationFrame(() => {
      hoverPending = false;
      const i = pickAt(e);
      if (i !== state.hover) { state.hover = i; invalidate(); }
      if (i >= 0) {
        const r = stage.getBoundingClientRect();
        const organ = organOf.get(i);
        tooltip.innerHTML = `<b>${esc(parts[i].name)}</b>${organ && ORGANS[organ] ? `<span>${esc(ORGANS[organ].name)} · ${esc(SYSTEM_INFO[parts[i].system].label)}</span>` : `<span>${esc(SYSTEM_INFO[parts[i].system].label)}</span>`}`;
        tooltip.style.left = `${e.clientX - r.left + 14}px`;
        tooltip.style.top = `${e.clientY - r.top + 14}px`;
        tooltip.hidden = false;
      } else tooltip.hidden = true;
    });
  }
  function tap(e: PointerEvent) {
    const i = pickAt(e);
    if (state.quiz) { answerQuiz(i); return; }
    if (i < 0) return;
    const organ = organOf.get(i);
    const s = state.selection;
    if (s?.kind === 'organ' && organs[s.key].parts.includes(i)) select({ kind: 'part', index: i }, false);
    else if (!s && organ && ORGANS[organ]) select({ kind: 'organ', key: organ }, true);
    else select({ kind: 'part', index: i }, false);
  }

  // ---- selection and detail panel ------------------------------------------------------------------------------------
  const home = $('[data-home]'), detail = $('[data-detail]');
  function select(sel: Selection, reframe: boolean) {
    state.selection = sel;
    state.isolate = false;
    if (sel?.kind === 'organ' && ORGANS[sel.key].sex && ORGANS[sel.key].sex !== state.sex) setSex(ORGANS[sel.key].sex!);
    updateParts();
    renderDetail();
    if (sel) { pulseUntil = performance.now() + 900; if (reframe) frame(selectedParts(), undefined, 1.45); invalidate(); }
    syncActive();
  }
  function back() {
    state.selection = null;
    state.isolate = false;
    updateParts();
    renderDetail();
    if (state.region) frame(parts.filter(p => p.region === state.region && modes[p.index] === MODE.solid).map(p => p.index));
    else frameBody();
    syncActive();
  }

  function renderDetail() {
    const s = state.selection;
    home.hidden = Boolean(s);
    detail.hidden = !s;
    if (!s) return;
    let html = '<button type="button" class="ml-back" data-back>← Quay lại toàn thân</button>';
    if (s.kind === 'organ') {
      const o = ORGANS[s.key], list = organParts(s.key), concepts = organs[s.key].concepts;
      html += `<p class="ml-eyebrow" style="--tone:${SYSTEM_INFO[o.system].color}">${esc(SYSTEM_INFO[o.system].label)}</p>
        <h3>${esc(o.name)}</h3>
        <p class="ml-latin"><i>${esc(o.latin)}</i>${concepts.length ? ` · <span title="Foundational Model of Anatomy">${concepts.map(esc).join(', ')}</span>` : ' · <span>Human Reference Atlas</span>'}</p>
        <div class="ml-actions">
          <button type="button" class="ml-primary" data-d-focus>Khám phá ở dạng 3D</button>
          <button type="button" data-d-isolate aria-pressed="${state.isolate}">Chỉ hiện cấu trúc này</button>
          <button type="button" class="ml-icon" data-d-pin aria-pressed="${state.pinned.has(`o:${s.key}`)}" aria-label="Ghim nhãn" title="Ghim nhãn">⌖</button>
        </div>
        <p class="ml-desc">${esc(o.description)}</p>
        <dl class="ml-facts"><div><dt>Chức năng</dt><dd>${esc(o.function)}</dd></div><div><dt>Vị trí</dt><dd>${esc(o.location)}</dd></div></dl>
        <aside class="ml-callout"><b>Bạn có biết?</b><p>${esc(o.fact)}</p></aside>
        <section class="ml-included"><h4>Các cấu trúc được bao gồm · <span>${list.length}</span></h4>
          <input type="search" data-d-filter placeholder="Lọc cấu trúc…" aria-label="Lọc cấu trúc của ${esc(o.name)}" />
          <ul data-d-list>${structureList(list)}</ul></section>`;
    } else {
      const p = parts[s.index], organ = organOf.get(s.index), side = sideOf(p.name);
      html += `<p class="ml-eyebrow" style="--tone:${SYSTEM_INFO[p.system].color}">${esc(SYSTEM_INFO[p.system].label)}</p>
        <h3 class="ml-part-name">${esc(p.name)}</h3>
        <p class="ml-latin">${p.fma ? `<span title="Foundational Model of Anatomy">${esc(p.fma)}</span> · ` : ''}${p.source === 'HRA' ? 'Human Reference Atlas' : 'BodyParts3D'}${side ? ` · ${side === 'L' ? 'L – bên trái' : 'R – bên phải'}` : ''}</p>
        <div class="ml-actions">
          <button type="button" class="ml-primary" data-d-focus>Khám phá ở dạng 3D</button>
          <button type="button" data-d-isolate aria-pressed="${state.isolate}">Chỉ hiện cấu trúc này</button>
          <button type="button" class="ml-icon" data-d-pin aria-pressed="${state.pinned.has(`p:${s.index}`)}" aria-label="Ghim nhãn" title="Ghim nhãn">⌖</button>
        </div>
        <p class="ml-desc">${esc(SYSTEM_INFO[p.system].note)}</p>
        <dl class="ml-facts"><div><dt>Vùng</dt><dd>${esc(REGION_INFO[p.region].label)}</dd></div>${organ && ORGANS[organ] ? `<div><dt>Thuộc cơ quan</dt><dd><button type="button" class="ml-link" data-organ="${organ}">${esc(ORGANS[organ].name)} →</button></dd></div>` : ''}</dl>
        <p class="ml-note">Tên cấu trúc giữ nguyên tiếng Anh theo dữ liệu gốc; L/R là bên trái/bên phải của chính cơ thể.</p>`;
    }
    detail.innerHTML = html;
    detail.scrollTop = 0;
  }
  function structureList(list: number[], filter = '') {
    const shown = filter ? rank(list, filter, i => [parts[i].name, parts[i].fma], 400) : list;
    return shown.map(i => `<li><button type="button" data-part="${i}"><span>${esc(parts[i].name)}</span>${parts[i].fma ? `<small>${esc(parts[i].fma)}</small>` : ''}</button></li>`).join('') || '<li class="ml-empty">Không có cấu trúc khớp.</li>';
  }
  detail.addEventListener('click', e => {
    const t = (e.target as HTMLElement).closest<HTMLElement>('button');
    if (!t) return;
    const s = state.selection;
    if (t.matches('[data-back]')) back();
    else if (t.matches('[data-d-focus]')) { pulseUntil = performance.now() + 900; frame(selectedParts(), undefined, 1.1); }
    else if (t.matches('[data-d-isolate]')) { state.isolate = !state.isolate; t.setAttribute('aria-pressed', String(state.isolate)); updateParts(); frame(selectedParts(), undefined, 1.3); }
    else if (t.matches('[data-d-pin]') && s) {
      const key = s.kind === 'organ' ? `o:${s.key}` : `p:${s.index}`;
      if (state.pinned.has(key)) state.pinned.delete(key); else state.pinned.add(key);
      t.setAttribute('aria-pressed', String(state.pinned.has(key)));
      invalidate();
    } else if (t.dataset.part) select({ kind: 'part', index: Number(t.dataset.part) }, true);
    else if (t.dataset.organ) select({ kind: 'organ', key: t.dataset.organ }, true);
  });
  detail.addEventListener('input', e => {
    const t = e.target as HTMLInputElement;
    if (t.matches('[data-d-filter]') && state.selection?.kind === 'organ') $('[data-d-list]').innerHTML = structureList(organParts(state.selection.key), t.value);
  });

  // ---- sidebar: cards, tabs, regions, systems, organs ----------------------------------------------------------------
  const cardRow = $('[data-cards]');
  cardRow.innerHTML = CARD_ORGANS.map(k => `<button type="button" class="ml-card" data-organ="${k}"><canvas width="152" height="112" aria-hidden="true"></canvas><span>${esc(ORGANS[k].name)}</span></button>`).join('');
  $('[data-card-count]').textContent = String(CARD_ORGANS.length);
  const featured = $('[data-featured]');
  featured.innerHTML = ['heart', 'brain', 'liver'].map(k => `<button type="button" class="ml-feature" data-organ="${k}"><canvas width="152" height="112" aria-hidden="true"></canvas><span><b>${esc(ORGANS[k].name)}</b><small>${esc(ORGANS[k].description.split('. ')[0])}.</small></span></button>`).join('');

  const regionList = $('[data-region-list]');
  regionList.innerHTML = `<li><button type="button" data-region="" aria-pressed="true">${bodyIcon(null)}<span><b>Toàn thân</b><small>Hiện mọi vùng</small></span></button></li>` +
    REGION_ORDER.map(r => `<li><button type="button" data-region="${r}" aria-pressed="false">${bodyIcon(r)}<span><b>${esc(REGION_INFO[r].label)}</b><small>${esc(REGION_INFO[r].hint)}</small></span><em data-region-count="${r}"></em></button></li>`).join('');
  const systemList = $('[data-system-list]');
  systemList.innerHTML = manifest.systems.map(s => `<li><label><input type="checkbox" data-system="${s}" ${state.systems.has(s) ? 'checked' : ''}/><i style="--tone:${SYSTEM_INFO[s].color}"></i><span><b>${esc(SYSTEM_INFO[s].label)}</b><small>${esc(SYSTEM_INFO[s].note)}</small></span><em data-system-count="${s}"></em></label></li>`).join('');
  const organList = $('[data-organ-list]');
  const bySystem = new Map<SystemId, string[]>();
  for (const [k, o] of Object.entries(ORGANS)) bySystem.set(o.system, [...(bySystem.get(o.system) ?? []), k]);
  organList.innerHTML = [...bySystem].map(([sys, keys]) => `<li class="ml-group"><p style="--tone:${SYSTEM_INFO[sys].color}">${esc(SYSTEM_INFO[sys].label)}</p><ul>${keys.map(k => `<li><button type="button" data-organ="${k}"><span>${esc(ORGANS[k].name)}</span><small>${ORGANS[k].sex === 'female' ? 'Nữ · ' : ORGANS[k].sex === 'male' ? 'Nam · ' : ''}${organs[k].parts.length} cấu trúc</small></button></li>`).join('')}</ul></li>`).join('');
  function counts() {
    const sex = visibleSex();
    for (const r of REGION_ORDER) $(`[data-region-count="${r}"]`).textContent = nf.format(parts.filter(p => p.region === r && inAtlas(p, sex) && state.systems.has(p.system)).length);
    for (const s of manifest.systems) $(`[data-system-count="${s}"]`).textContent = nf.format(parts.filter(p => p.system === s && inAtlas(p, sex)).length);
  }

  root.addEventListener('click', e => {
    const t = (e.target as HTMLElement).closest<HTMLElement>('[data-organ]');
    if (t && !detail.contains(t)) { stopTour(); select({ kind: 'organ', key: t.dataset.organ! }, true); }
  });
  $$('[data-tab]').forEach(tab => tab.addEventListener('click', () => {
    $$('[data-tab]').forEach(t => t.setAttribute('aria-selected', String(t === tab)));
    $$('[data-panel]').forEach(p => { p.hidden = p.dataset.panel !== tab.dataset.tab; });
  }));
  regionList.addEventListener('click', e => {
    const t = (e.target as HTMLElement).closest<HTMLElement>('[data-region]');
    if (!t) return;
    state.region = (t.dataset.region || null) as RegionId | null;
    state.selection = null;
    updateParts();
    renderDetail();
    syncActive();
    if (state.region) frame(parts.filter(p => p.region === state.region && modes[p.index] === MODE.solid).map(p => p.index));
    else frameBody();
  });
  systemList.addEventListener('change', e => {
    const t = e.target as HTMLInputElement;
    const s = t.dataset.system as SystemId;
    if (t.checked) state.systems.add(s); else state.systems.delete(s);
    if (t.checked && ['muscular', 'arterial', 'venous'].includes(s) && !loaded[1]) status('Đang tải cơ và mạch máu…');
    updateParts();
    counts();
  });
  $('[data-systems-default]').addEventListener('click', () => { state.systems = new Set(DEFAULT_SYSTEMS); syncSystems(); });
  $('[data-systems-all]').addEventListener('click', () => { state.systems = new Set(manifest.systems); if (!loaded[1]) status('Đang tải cơ và mạch máu…'); syncSystems(); });
  function syncSystems() {
    $$<HTMLInputElement>('[data-system]').forEach(i => { i.checked = state.systems.has(i.dataset.system as SystemId); });
    updateParts();
    counts();
  }
  function syncActive() {
    $$('[data-region]').forEach(b => b.setAttribute('aria-pressed', String((b.dataset.region || null) === state.region)));
    const key = state.selection?.kind === 'organ' ? state.selection.key : state.selection?.kind === 'part' ? organOf.get(state.selection.index) : undefined;
    $$('.ml-card, [data-organ-list] [data-organ]').forEach(b => b.classList.toggle('is-active', b.dataset.organ === key));
  }

  // ---- search ----------------------------------------------------------------------------------------------------------
  const search = $<HTMLInputElement>('[data-search]'), results = $('[data-results]');
  type Hit = { kind: 'organ'; key: string } | { kind: 'part'; index: number };
  const organHits: Hit[] = Object.keys(ORGANS).map(key => ({ kind: 'organ', key }));
  const partHits: Hit[] = parts.map(p => ({ kind: 'part', index: p.index }));
  let hits: Hit[] = [], active = 0;
  function runSearch() {
    const q = search.value;
    const sex = state.sex;
    hits = [...rank(organHits, q, h => h.kind === 'organ' ? [ORGANS[h.key].name, ORGANS[h.key].latin, h.key.replace('-', ' ')] : [], 6),
      ...rank(partHits.filter(h => h.kind === 'part' && inAtlas(parts[h.index], sex)), q, h => h.kind === 'part' ? [parts[h.index].name, parts[h.index].fma] : [], 24)];
    active = 0;
    results.hidden = !q.trim();
    results.innerHTML = hits.length ? hits.map((h, i) => h.kind === 'organ'
      ? `<button type="button" role="option" data-hit="${i}" aria-selected="${i === active}"><b>${esc(ORGANS[h.key].name)}</b><small>Cơ quan · ${esc(ORGANS[h.key].latin)}</small></button>`
      : `<button type="button" role="option" data-hit="${i}" aria-selected="${i === active}"><b>${esc(parts[h.index].name)}</b><small>${esc(SYSTEM_INFO[parts[h.index].system].label)}${parts[h.index].fma ? ` · ${esc(parts[h.index].fma)}` : ''}</small></button>`).join('')
      : '<p class="ml-empty">Không tìm thấy cấu trúc. Thử “tim”, “femur”, “gan”…</p>';
  }
  function choose(i: number) {
    const h = hits[i];
    if (!h) return;
    results.hidden = true;
    search.value = '';
    stopTour();
    select(h.kind === 'organ' ? { kind: 'organ', key: h.key } : { kind: 'part', index: h.index }, true);
  }
  search.addEventListener('input', runSearch);
  search.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      active = (active + (e.key === 'ArrowDown' ? 1 : hits.length - 1)) % Math.max(1, hits.length);
      results.querySelectorAll('[data-hit]').forEach((b, i) => b.setAttribute('aria-selected', String(i === active)));
    } else if (e.key === 'Enter') { e.preventDefault(); choose(active); }
    else if (e.key === 'Escape') { search.value = ''; results.hidden = true; search.blur(); }
  });
  results.addEventListener('click', e => { const t = (e.target as HTMLElement).closest<HTMLElement>('[data-hit]'); if (t) choose(Number(t.dataset.hit)); });
  document.addEventListener('keydown', e => {
    const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement;
    if (e.key === '/' && !typing && root.getBoundingClientRect().bottom > 0 && root.getBoundingClientRect().top < innerHeight) { e.preventDefault(); search.focus(); }
    else if (e.key === 'Escape' && !typing && root.contains(document.activeElement ?? null)) { if (!$('[data-credits-panel]').hidden) closeCredits(); else if (state.selection) back(); }
  });

  // ---- bottom bar: views, sex, explode; zoom ---------------------------------------------------------------------------
  $$('[data-view]').forEach(b => b.addEventListener('click', () => {
    $$('[data-view]').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    const view = VIEWS[b.dataset.view as keyof typeof VIEWS];
    if (state.selection) frame(selectedParts(), view, 1.45); else if (state.region) frame(visibleList(), view); else frameBody(view);
  }));
  function setSex(sex: Sex) {
    state.sex = sex;
    state.femaleGoal = sex === 'female' ? 1 : 0;
    $$('[data-sex]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.sex === sex)));
    root.dataset.sex = sex;
    setTimeout(() => { counts(); if (state.selection?.kind === 'organ' && !organParts(state.selection.key).length) back(); }, 750);
    invalidate();
  }
  $$('[data-sex]').forEach(b => b.addEventListener('click', () => setSex(b.dataset.sex as Sex)));
  const explode = $<HTMLInputElement>('[data-explode]'), explodeValue = $('[data-explode-value]');
  explode.addEventListener('input', () => { state.explode = Number(explode.value) / 100; explodeValue.textContent = `${explode.value}%`; updateParts(); });
  $('[data-explode-reset]').addEventListener('click', () => { explode.value = '0'; explode.dispatchEvent(new Event('input')); });
  $$('[data-zoom]').forEach(b => b.addEventListener('click', () => {
    const z = Number(b.dataset.zoom);
    if (z === 0) { if (state.selection) frame(selectedParts(), undefined, 1.45); else frameBody(VIEWS.perspective); return; }
    camera.animateTo({ distance: Math.max(0.08, Math.min(8, camera.distance * (z > 0 ? 0.78 : 1.28))) });
    invalidate();
  }));

  // ---- footer buttons: guided tour, smart labels, credits; header: quiz, fullscreen ------------------------------------
  const tourPanel = $('[data-tour-panel]');
  function showTour(i: number) {
    state.tour = i;
    tourPanel.hidden = false;
    const [key, text] = TOUR[i];
    tourPanel.querySelector('[data-tour-step]')!.textContent = `Bước ${i + 1}/${TOUR.length} · ${ORGANS[key].name}`;
    tourPanel.querySelector('[data-tour-text]')!.textContent = text;
    tourPanel.querySelector<HTMLButtonElement>('[data-tour-prev]')!.disabled = i === 0;
    tourPanel.querySelector('[data-tour-next]')!.textContent = i === TOUR.length - 1 ? 'Kết thúc' : 'Tiếp →';
    select({ kind: 'organ', key }, true);
  }
  function stopTour() { if (state.tour >= 0) { state.tour = -1; tourPanel.hidden = true; } }
  $$('[data-learn]').forEach(b => b.addEventListener('click', () => { stopQuiz(); showTour(0); }));
  tourPanel.addEventListener('click', e => {
    const t = (e.target as HTMLElement).closest('button');
    if (!t) return;
    if (t.matches('[data-tour-prev]')) showTour(Math.max(0, state.tour - 1));
    else if (t.matches('[data-tour-next]')) { if (state.tour >= TOUR.length - 1) { stopTour(); back(); } else showTour(state.tour + 1); }
    else if (t.matches('[data-tour-close]')) stopTour();
  });
  const labelsButton = $('[data-labels]');
  labelsButton.addEventListener('click', () => { state.labels = !state.labels; labelsButton.setAttribute('aria-pressed', String(state.labels)); invalidate(); });

  const credits = $('[data-credits-panel]');
  const openCredits = () => { credits.hidden = false; credits.querySelector<HTMLElement>('[data-credits-close]')!.focus(); };
  const closeCredits = () => { credits.hidden = true; $('[data-credits]').focus(); };
  $('[data-credits]').addEventListener('click', openCredits);
  credits.addEventListener('click', e => { if (e.target === credits || (e.target as HTMLElement).closest('[data-credits-close]')) closeCredits(); });

  const quizPanel = $('[data-quiz-panel]');
  function quizQuestion() {
    const q = state.quiz!;
    const key = QUIZ[q.i];
    quizPanel.hidden = false;
    quizPanel.querySelector('[data-quiz-step]')!.textContent = `Câu ${q.i + 1}/${QUIZ.length} · Điểm ${q.score}`;
    quizPanel.querySelector('[data-quiz-text]')!.innerHTML = `Hãy chạm vào <b>${esc(ORGANS[key].name)}</b> trên mô hình.`;
    quizPanel.querySelector('[data-quiz-feedback]')!.textContent = '';
    quizPanel.querySelector<HTMLButtonElement>('[data-quiz-next]')!.hidden = true;
  }
  function startQuiz() {
    stopTour();
    state.quiz = { i: 0, score: 0, answered: false };
    state.selection = null;
    state.region = null;
    state.explode = 0; explode.value = '0'; explodeValue.textContent = '0%';
    renderDetail();
    updateParts();
    frameBody(VIEWS.front);
    quizQuestion();
  }
  function answerQuiz(i: number) {
    const q = state.quiz!;
    if (q.answered) return;
    const key = QUIZ[q.i];
    const right = i >= 0 && organs[key].parts.includes(i);
    if (i < 0) { quizPanel.querySelector('[data-quiz-feedback]')!.textContent = 'Chưa chạm trúng cấu trúc nào, thử lại nhé.'; return; }
    q.answered = true;
    if (right) q.score++;
    const picked = organOf.get(i);
    quizPanel.querySelector('[data-quiz-feedback]')!.innerHTML = right ? `✓ Chính xác! ${esc(ORGANS[key].fact)}` : `✗ Đó là ${esc(picked && ORGANS[picked] ? ORGANS[picked].name : parts[i].name)}. ${esc(ORGANS[key].name)} đang được tô sáng.`;
    quizPanel.querySelector('[data-quiz-step]')!.textContent = `Câu ${q.i + 1}/${QUIZ.length} · Điểm ${q.score}`;
    state.selection = { kind: 'organ', key };
    updateParts();
    pulseUntil = performance.now() + 1400;
    invalidate();
    const next = quizPanel.querySelector<HTMLButtonElement>('[data-quiz-next]')!;
    next.hidden = false;
    next.textContent = q.i === QUIZ.length - 1 ? 'Xem kết quả' : 'Câu tiếp →';
  }
  function stopQuiz() { if (state.quiz) { state.quiz = null; quizPanel.hidden = true; } }
  $('[data-quiz]').addEventListener('click', () => (state.quiz ? (stopQuiz(), back()) : startQuiz()));
  quizPanel.addEventListener('click', e => {
    const t = (e.target as HTMLElement).closest('button');
    if (!t) return;
    if (t.matches('[data-quiz-close]')) { stopQuiz(); back(); return; }
    if (!t.matches('[data-quiz-next]') || !state.quiz) return;
    const q = state.quiz;
    if (q.i === QUIZ.length - 1) {
      quizPanel.querySelector('[data-quiz-step]')!.textContent = 'Hoàn thành';
      quizPanel.querySelector('[data-quiz-text]')!.innerHTML = `Bạn tìm đúng <b>${q.score}/${QUIZ.length}</b> cơ quan.`;
      quizPanel.querySelector('[data-quiz-feedback]')!.textContent = q.score === QUIZ.length ? 'Xuất sắc!' : 'Mở các thẻ cơ quan bên trái để ôn lại vị trí, rồi thử lần nữa.';
      t.hidden = true;
      state.quiz = { ...q, answered: true };
      return;
    }
    q.i++;
    q.answered = false;
    state.selection = null;
    updateParts();
    quizQuestion();
  });

  const fullscreen = $('[data-fullscreen]');
  fullscreen.addEventListener('click', () => { if (document.fullscreenElement) void document.exitFullscreen(); else void root.requestFullscreen?.(); });
  document.addEventListener('fullscreenchange', () => { fullscreen.setAttribute('aria-pressed', String(document.fullscreenElement === root)); requestAnimationFrame(resize); });
  const menu = $('[data-menu-list]');
  $('[data-menu]').addEventListener('click', () => { menu.hidden = !menu.hidden; $('[data-menu]').setAttribute('aria-expanded', String(!menu.hidden)); });
  menu.addEventListener('click', e => {
    const t = (e.target as HTMLElement).closest<HTMLElement>('[data-action]');
    if (!t) return;
    menu.hidden = true;
    if (t.dataset.action === 'reset') {
      stopQuiz(); stopTour();
      state.region = null; state.selection = null; state.isolate = false; state.pinned.clear(); state.systems = new Set(DEFAULT_SYSTEMS);
      explode.value = '0'; explode.dispatchEvent(new Event('input'));
      syncSystems(); renderDetail(); syncActive(); setSex('male'); frameBody(VIEWS.perspective);
    } else if (t.dataset.action === 'credits') openCredits();
  });

  function status(text: string) { const el = $('[data-status]'); el.textContent = text; el.hidden = !text; }

  // ---- loading ---------------------------------------------------------------------------------------------------------
  new ResizeObserver(resize).observe(stage);
  resize();
  const progress = $('[data-progress]'), loadingText = $('[data-loading-text]');
  const core = await fetchWithProgress(coreUrl, manifest.files[0].bytes, f => { progress.style.width = `${(f * 100).toFixed(1)}%`; loadingText.textContent = `Đang tải giải phẫu… ${Math.round(f * 100)}%`; });
  renderer.addFile(core, parts.filter(p => p.file === 0));
  loaded[0] = true;
  camera.yaw = VIEWS.perspective.yaw; camera.pitch = VIEWS.perspective.pitch;
  updateParts();
  counts();
  const start = fitView(parts.filter(p => p.system === 'skeletal').map(p => p.index), camera.yaw, camera.pitch);
  camera.target = start.target; camera.distance = start.distance;
  $('[data-loading]').hidden = true;
  root.setAttribute('aria-busy', 'false');
  invalidate();
  await thumbnails();
  status('Đang tải cơ và mạch máu…');
  const extra = await fetchWithProgress(extraUrl, manifest.files[1].bytes, () => undefined);
  renderer.addFile(extra, parts.filter(p => p.file === 1));
  loaded[1] = true;
  status('');
  updateParts();

  // Organ thumbnails for the cards, rendered one per frame with the organ alone in a three-quarter view.
  async function thumbnails() {
    const canvases = $$<HTMLCanvasElement>('[data-organ] canvas');
    const keys = [...new Set(canvases.map(c => (c.closest('[data-organ]') as HTMLElement).dataset.organ!))];
    for (const key of keys) {
      await new Promise(r => requestAnimationFrame(r));
      const list = organParts(key);
      for (const p of parts) renderer.setPart(p.index, [0, 0, 0], MODE.hidden, baseColor[p.index]);
      for (const i of list) renderer.setPart(i, [0, 0, 0], MODE.solid, baseColor[i]);
      renderer.commit();
      const { center, radius } = sphereOf(list);
      const cam = new OrbitCamera();
      cam.yaw = key === 'kidneys' ? 0.4 : 0.35; cam.pitch = 0.12; cam.target = center; cam.distance = cam.fitDistance(radius * 0.92, 152 / 112);
      const image = renderer.snapshot(cam.matrix(152 / 112), cam.eye(), 0, 152, 112);
      for (const c of canvases) if ((c.closest('[data-organ]') as HTMLElement).dataset.organ === key) c.getContext('2d')!.putImageData(image, 0, 0);
      for (const p of parts) renderer.setPart(p.index, offsets[p.index], modes[p.index], baseColor[p.index]);
      renderer.commit();
    }
  }
}

function bodyIcon(region: RegionId | null) {
  const on = (r: RegionId) => (region === null || region === r ? 'on' : '');
  return `<svg class="ml-body-icon" viewBox="0 0 40 64" aria-hidden="true">
    <circle class="${on('head')}" cx="20" cy="7" r="5.5"/><rect class="${on('head')}" x="17.5" y="12" width="5" height="4" rx="1.5"/>
    <path class="${on('thorax')}" d="M11 17h18l-1.5 13h-15z"/>
    <path class="${on('abdomen')}" d="M12.5 31h15l1 9h-17z"/>
    <path class="${on('arm')}" d="M9.5 17.5l-3.5 14 -1.5 11 2.6.4 2.6-11 2.3-9.5zM30.5 17.5l3.5 14 1.5 11-2.6.4-2.6-11-2.3-9.5z"/>
    <path class="${on('leg')}" d="M12 41h7.4l-1 21h-3.6zM20.6 41H28l-2.8 21h-3.6z"/>
  </svg>`;
}
