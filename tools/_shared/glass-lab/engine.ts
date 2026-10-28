// Glass lab runtime: each chapter is a short animated scene with a timeline (play, pause, scrub, replay), subtitles
// that follow the timeline, live metrics and a data table. Labs only describe scenes and wire their controls.
import { Gfx } from './gfx';

export interface Beat { at: number; text: string }
export interface Metric { label: string; value: string; tone?: 'cyan' | 'magenta' | 'amber' | 'lime' | 'violet' | 'red' }
import type { Rect } from './gfx';
export interface Frame { g: Gfx; w: number; h: number; t: number; time: number; narrow: boolean; progress: number; /** Area left free by the HUD and the subtitle. */ safe: Rect }
export interface PointerEvent2 { type: 'down' | 'move' | 'up' | 'leave' | 'click'; x: number; y: number; dx: number; dy: number; dragging: boolean }
export interface Table { headers: string[]; rows: (string | number)[][]; caption?: string }

export interface Scene {
  title: string;
  /** Seconds of animation; after that the scene holds its final state (and keeps idling if `idle`). */
  duration: number;
  beats(): Beat[];
  formula?: string | (() => string);
  /** Keep redrawing after the timeline ends (camera drift, pulsing markers). */
  idle?: boolean;
  render(frame: Frame): void;
  metrics?(): Metric[];
  table?(): Table;
  pointer?(event: PointerEvent2, frame: Frame): boolean | void;
  enter?(): void;
}

export interface LabApi {
  root: HTMLElement;
  /** Redraw on the next frame (call after any parameter change). */
  invalidate(): void;
  /** Restart the current chapter's timeline from 0 and play. */
  replay(): void;
  /** Jump the timeline to the end of the current chapter. */
  finish(): void;
  chapter(): number;
  go(index: number): void;
  range(name: string, format: (value: number) => string, onInput: (value: number) => void): { value: number; set(value: number): void; input: HTMLInputElement };
  on(selector: string, event: string, handler: (element: HTMLElement, event: Event) => void): void;
}

const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;

export function glassLab(id: string, scenes: Scene[], setup: (lab: LabApi) => void) {
  const run = () => document.querySelectorAll<HTMLElement>(`[data-glass-lab="${id}"]`).forEach(root => {
    if (root.dataset.initialized) return;
    root.dataset.initialized = 'true';
    try { start(root, scenes, setup); } finally { root.setAttribute('aria-busy', 'false'); }
  });
  run();
  document.addEventListener('astro:page-load', run);
}

function start(root: HTMLElement, scenes: Scene[], setup: (lab: LabApi) => void) {
  const $ = <T extends Element = HTMLElement>(selector: string) => root.querySelector<T>(selector)!;
  const canvas = $<HTMLCanvasElement>('[data-canvas]');
  const screen = $('[data-screen]');
  const ctx = canvas.getContext('2d')!;
  const hudTop = $('.hud-top');
  const play = $<HTMLButtonElement>('[data-play]');
  const scrub = $<HTMLInputElement>('[data-scrub]');
  const time = $('[data-time]');
  const speedButton = $<HTMLButtonElement>('[data-speed]');
  const chapters = [...root.querySelectorAll<HTMLButtonElement>('[data-chapter]')];
  const state = { chapter: 0, t: 0, playing: false, speed: 1, dirty: true, visible: true, width: 0, height: 0, last: 0, drag: false, px: 0, py: 0, moved: 0, beat: -1 };
  let g = new Gfx(ctx, 1, 1);

  const scene = () => scenes[state.chapter];
  const value = <T,>(v: T | (() => T) | undefined) => (typeof v === 'function' ? (v as () => T)() : v);

  function resize() {
    const rect = screen.getBoundingClientRect();
    const narrow = rect.width < 720;
    const height = narrow ? Math.min(Math.max(rect.width * 1.6, 600), 820) : Math.min(Math.max(rect.width * 0.5625, 440), 720);
    screen.style.height = `${Math.round(height)}px`;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    state.width = rect.width; state.height = height;
    canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(height * dpr);
    canvas.style.width = `${rect.width}px`; canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    g = new Gfx(ctx, rect.width, height);
    state.dirty = true;
  }

  function frame(now = performance.now()): Frame {
    const narrow = state.width < 720;
    const top = Math.max(52, (hudTop.getBoundingClientRect().bottom - screen.getBoundingClientRect().top) + 8), bottom = narrow ? 104 : 66;
    return { g, w: state.width, h: state.height, t: state.t, time: now / 1000, narrow, progress: Math.min(1, state.t / scene().duration), safe: { x: 14, y: top, w: state.width - 28, h: state.height - top - bottom } };
  }

  function draw(now: number) {
    const s = scene();
    g.background();
    s.render(frame(now));
    updateHud();
  }

  let hudCache = '';
  function updateHud() {
    const s = scene();
    const beats = s.beats();
    let active = 0;
    beats.forEach((beat, index) => { if (state.t >= beat.at - 1e-6) active = index; });
    if (active !== state.beat || hudCache === '') {
      state.beat = active;
      $('[data-subtitle]').textContent = beats[active]?.text ?? '';
      root.querySelectorAll('[data-beats] li').forEach((li, index) => li.classList.toggle('active', index === active));
    }
    const metrics = s.metrics?.() ?? [];
    const html = metrics.map(m => `<li class="tone-${m.tone ?? 'cyan'}"><span>${m.label}</span><b>${m.value}</b></li>`).join('');
    if (html !== hudCache) { $('[data-metrics]').innerHTML = html; hudCache = html || ' '; }
    const duration = s.duration;
    time.textContent = `${clock(state.t)} / ${clock(duration)}`;
    scrub.value = String(Math.round(state.t / duration * 1000));
    chapters.forEach((button, index) => button.querySelector<HTMLElement>('[data-progress]')!.style.width = `${index < state.chapter ? 100 : index === state.chapter ? state.t / duration * 100 : 0}%`);
  }

  function loop(now: number) {
    const dt = Math.min(0.05, (now - (state.last || now)) / 1000);
    state.last = now;
    const s = scene();
    if (state.playing) {
      state.t = Math.min(s.duration, state.t + dt * state.speed);
      if (state.t >= s.duration) setPlaying(false);
      state.dirty = true;
    }
    if (state.visible && (state.dirty || (s.idle && !reduced()))) { state.dirty = false; draw(now); }
    requestAnimationFrame(loop);
  }

  function setPlaying(on: boolean) {
    state.playing = on;
    play.textContent = on ? '❚❚' : '▶';
    play.setAttribute('aria-label', on ? 'Tạm dừng' : 'Phát');
    play.setAttribute('aria-pressed', String(on));
  }

  function enterChapter(index: number) {
    state.chapter = index;
    const s = scene();
    s.enter?.();
    chapters.forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
    $('[data-title]').textContent = `${index + 1}. ${s.title}`;
    $('[data-chapter-label]').textContent = `Chương ${index + 1} / ${scenes.length} · ${s.title}`;
    root.querySelectorAll<HTMLElement>('[data-chapters]').forEach(element => { element.hidden = !element.dataset.chapters!.split(/[ ,]+/).map(Number).includes(index); });
    refreshText();
    if (reduced()) { state.t = s.duration; setPlaying(false); } else { state.t = 0; setPlaying(true); }
    fillTable();
    state.dirty = true;
  }

  /** Transcript, formula and canvas label depend on parameters, so rebuild them whenever those change. */
  function refreshText() {
    const s = scene();
    const formula = value(s.formula);
    const box = $('[data-formula]'); box.textContent = formula ?? ''; box.hidden = !formula;
    $('[data-beats]').replaceChildren(...s.beats().map(beat => { const li = document.createElement('li'); li.textContent = beat.text; li.dataset.at = String(beat.at); return li; }));
    canvas.setAttribute('aria-label', `${s.title}. ${s.beats().map(beat => beat.text).join(' ')}`);
    state.beat = -1; hudCache = '';
  }

  function fillTable() {
    const table = scene().table?.();
    const element = $<HTMLTableElement>('[data-table]');
    const details = element.closest('details') as HTMLElement | null;
    if (details) details.hidden = !table;
    element.replaceChildren();
    if (!table) return;
    if (table.caption) element.createCaption().textContent = table.caption;
    const head = element.createTHead().insertRow();
    for (const header of table.headers) { const th = document.createElement('th'); th.scope = 'col'; th.textContent = header; head.append(th); }
    const body = element.createTBody();
    for (const row of table.rows) { const tr = body.insertRow(); for (const cell of row) tr.insertCell().textContent = String(cell); }
  }

  // Player
  play.addEventListener('click', () => { if (state.playing) setPlaying(false); else { if (state.t >= scene().duration) state.t = 0; setPlaying(true); } });
  $('[data-replay]').addEventListener('click', () => { state.t = 0; setPlaying(true); state.dirty = true; });
  scrub.addEventListener('input', () => { state.t = Number(scrub.value) / 1000 * scene().duration; setPlaying(false); state.dirty = true; });
  speedButton.addEventListener('click', () => { state.speed = state.speed === 1 ? 2 : state.speed === 2 ? 0.5 : 1; speedButton.textContent = `${String(state.speed).replace('.', ',')}×`; });
  chapters.forEach((button, index) => button.addEventListener('click', () => enterChapter(index)));
  root.querySelectorAll<HTMLElement>('[data-beats]').forEach(list => list.addEventListener('click', event => {
    const li = (event.target as HTMLElement).closest('li');
    if (li?.dataset.at) { state.t = Number(li.dataset.at); setPlaying(false); state.dirty = true; }
  }));

  // Pointer on the canvas, in CSS pixels
  const local = (event: PointerEvent) => { const rect = canvas.getBoundingClientRect(); return [event.clientX - rect.left, event.clientY - rect.top] as const; };
  const send = (type: PointerEvent2['type'], x: number, y: number, dx = 0, dy = 0) => { if (scene().pointer?.({ type, x, y, dx, dy, dragging: state.drag }, frame())) state.dirty = true; };
  canvas.addEventListener('pointerdown', event => { const [x, y] = local(event); state.drag = true; state.moved = 0; state.px = x; state.py = y; canvas.setPointerCapture(event.pointerId); send('down', x, y); });
  canvas.addEventListener('pointermove', event => { const [x, y] = local(event); const dx = x - state.px, dy = y - state.py; state.px = x; state.py = y; state.moved += Math.abs(dx) + Math.abs(dy); send('move', x, y, dx, dy); });
  canvas.addEventListener('pointerup', event => { const [x, y] = local(event); state.drag = false; send('up', x, y); if (state.moved < 6) send('click', x, y); });
  canvas.addEventListener('pointerleave', () => send('leave', -1, -1));

  new ResizeObserver(() => resize()).observe(screen);
  new IntersectionObserver(entries => { state.visible = entries[0].isIntersecting; if (state.visible) state.dirty = true; }).observe(screen);
  document.addEventListener('astro:before-swap', () => setPlaying(false), { once: true });

  const api: LabApi = {
    root,
    invalidate: () => { state.dirty = true; refreshText(); fillTable(); },
    replay: () => { state.t = 0; if (!reduced()) setPlaying(true); else state.t = scene().duration; state.dirty = true; refreshText(); fillTable(); },
    finish: () => { state.t = scene().duration; setPlaying(false); state.dirty = true; },
    chapter: () => state.chapter,
    go: index => enterChapter(index),
    range(name, format, onInput) {
      const input = $<HTMLInputElement>(`input[name="${name}"]`);
      const output = root.querySelector<HTMLOutputElement>(`[data-value="${name}"]`);
      const show = () => { if (output) output.textContent = format(Number(input.value)); };
      input.addEventListener('input', () => { show(); onInput(Number(input.value)); state.dirty = true; refreshText(); fillTable(); });
      show();
      return { input, get value() { return Number(input.value); }, set(v: number) { input.value = String(v); show(); } };
    },
    on(selector, event, handler) { root.querySelectorAll<HTMLElement>(selector).forEach(element => element.addEventListener(event, e => { handler(element, e); state.dirty = true; refreshText(); fillTable(); })); },
  };
  setup(api);
  resize();
  enterChapter(0);
  requestAnimationFrame(loop);
}
