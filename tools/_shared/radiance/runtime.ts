import * as nerf from '../../kien-thuc/nerf-explainer/render';
import * as gaussian from '../../kien-thuc/gaussian-splatting-explainer/render';
import * as synctalk from '../../du-an/synctalk/render';
import * as instag from '../../du-an/instag/render';
import { asset, type LabState, type LabView, type Mode, type Root } from './lab-core';

// Every number shown here was produced by the trained models at bake time (see docs/development/radiance-labs.md).
// Tool-specific views and step texts live in each tool's render.ts; this file wires the shared lab chrome.
const tools: Record<Mode, typeof nerf> = { nerf, 'gaussian-splatting': gaussian, synctalk, instag };

export function initRadianceLabs() {
  document.querySelectorAll<HTMLElement>('[data-radiance-mode]').forEach(element => {
    if (element.dataset.initialized) return;
    element.dataset.initialized = 'true';
    const controller = new AbortController();
    const root: Root = { element, mode: element.dataset.radianceMode as Mode, signal: controller.signal };
    const cleanups: (() => void)[] = [];
    const dispose = () => { controller.abort(); cleanups.forEach(cleanup => cleanup()); };
    document.addEventListener('astro:before-swap', dispose, { once: true });
    window.addEventListener('pagehide', dispose, { once: true });
    start(root, cleanups).catch(error => { if (!controller.signal.aborted) showError(element, error instanceof Error ? error.message : 'Không tải được dữ liệu.'); });
  });
}

function showError(element: HTMLElement, message: string) {
  const error = element.querySelector<HTMLElement>('[data-error]')!;
  error.hidden = false; error.textContent = message;
  element.querySelector<HTMLElement>('[data-loading]')!.hidden = true;
  element.setAttribute('aria-busy', 'false');
}

async function start(root: Root, cleanups: (() => void)[]) {
  const { element, mode, signal } = root;
  const $ = <T extends Element = HTMLElement>(selector: string) => element.querySelector<T>(selector);
  const canvas = $<HTMLCanvasElement>('[data-scene-canvas]')!;
  const overlay = $<HTMLCanvasElement>('[data-overlay]')!;
  const loading = $('[data-loading]')!; const progress = $<HTMLProgressElement>('[data-progress]')!;
  const inputs = [...element.querySelectorAll<HTMLInputElement>('input[type=range]')];
  const fetchJson = async <T>(path: string) => { const response = await fetch(asset(path), { signal }); if (!response.ok) throw new Error('Không đọc được ' + path); return response.json() as Promise<T>; };
  const fetchBytes = async (path: string) => { const response = await fetch(asset(path), { signal }); if (!response.ok) throw new Error('Không đọc được ' + path); return response.arrayBuffer(); };
  const state = { stage: 3, selected: [0.5, 0.45] as [number, number], layer: 'rgb' };

  const setStep = () => {
    const [title, description, formula] = tools[mode].explanations[state.stage];
    $('[data-step-number]')!.textContent = `Bước ${state.stage + 1} / 4`;
    $('[data-step-title]')!.textContent = title;
    $('[data-step-description]')!.textContent = description;
    $('[data-formula]')!.textContent = formula;
  };
  const valueLabel = (input: HTMLInputElement) => { const output = $(`[data-value="${input.name}"]`); if (output) output.textContent = input.value + (input.dataset.unit ? ' ' + input.dataset.unit : ''); };
  inputs.forEach(valueLabel);

  // Overlay: selection marker in canvas-relative coordinates (works for both WebGL and 2D views).
  const drawOverlay = (extra?: (context: CanvasRenderingContext2D, width: number, height: number) => void) => {
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const { clientWidth: width, clientHeight: height } = overlay;
    if (overlay.width !== Math.round(width * ratio)) { overlay.width = Math.round(width * ratio); overlay.height = Math.round(height * ratio); }
    const context = overlay.getContext('2d')!; context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, width, height);
    extra?.(context, width, height);
    const [x, y] = [state.selected[0] * width, state.selected[1] * height];
    context.lineWidth = 2; context.strokeStyle = '#ffffff'; context.beginPath(); context.arc(x, y, 9, 0, Math.PI * 2); context.stroke();
    context.strokeStyle = '#0ea5e9'; context.beginPath(); context.moveTo(x - 16, y); context.lineTo(x - 6, y); context.moveTo(x + 6, y); context.lineTo(x + 16, y); context.moveTo(x, y - 16); context.lineTo(x, y - 6); context.moveTo(x, y + 6); context.lineTo(x, y + 16); context.stroke();
  };

  const view = await tools[mode].createView(root, { $, canvas, fetchJson, fetchBytes, onProgress: fraction => { progress.value = fraction; } });
  cleanups.push(() => view.dispose());
  loading.hidden = true; element.setAttribute('aria-busy', 'false');

  const refresh = async () => {
    setStep();
    await view.render(state);
    drawOverlay((context, width, height) => view.overlay?.(context, width, height, state));
    await view.inspect(state);
  };
  let pending: Promise<void> | undefined; let queued = false;
  // A refresh can fail transiently (aborted fetch, data for a frame the user already left); hide any old error on the
  // next successful refresh instead of leaving a blocking overlay on top of the scene.
  const error = element.querySelector<HTMLElement>('[data-error]')!;
  const schedule = () => {
    if (pending) { queued = true; return; }
    pending = refresh().then(() => { error.hidden = true; }).catch(caught => { if (!signal.aborted) showError(element, caught instanceof Error ? caught.message : 'Không tải được dữ liệu.'); })
      .finally(() => { pending = undefined; if (queued) { queued = false; schedule(); } });
  };

  element.querySelectorAll<HTMLButtonElement>('[data-stage]').forEach(button => button.addEventListener('click', () => {
    state.stage = Number(button.dataset.stage);
    element.querySelectorAll('[data-stage]').forEach(other => other.setAttribute('aria-pressed', String(other === button)));
    view.onStage?.(state.stage); schedule();
  }, { signal }));
  inputs.forEach(input => input.addEventListener('input', () => { valueLabel(input); view.onInput(input.name, Number(input.value)); schedule(); }, { signal }));
  element.querySelectorAll<HTMLInputElement>('input[name=layer]').forEach(input => input.addEventListener('change', () => { state.layer = input.value; schedule(); }, { signal }));
  $<HTMLInputElement>('[data-vectors]')?.addEventListener('change', schedule, { signal });
  $<HTMLSelectElement>('[data-view]')?.addEventListener('change', event => { view.onView?.(Number((event.target as HTMLSelectElement).value)); schedule(); }, { signal });
  const reset = (all: boolean) => {
    view.resetCamera();
    if (all) {
      inputs.forEach(input => { input.value = input.defaultValue; valueLabel(input); view.onInput(input.name, Number(input.value)); });
      state.stage = 3; state.selected = [0.5, 0.45]; state.layer = 'rgb';
      element.querySelectorAll('[data-stage]').forEach(button => button.setAttribute('aria-pressed', String((button as HTMLElement).dataset.stage === '3')));
      element.querySelectorAll<HTMLInputElement>('input[name=layer]').forEach((input, index) => { input.checked = index === 0; });
      const select = $<HTMLSelectElement>('[data-view]'); if (select) { select.selectedIndex = 0; view.onView?.(-1); }
      view.onStage?.(3);
    }
    schedule();
  };
  $('[data-camera-reset]')!.addEventListener('click', () => reset(false), { signal });
  $('[data-reset]')!.addEventListener('click', () => reset(true), { signal });

  // Pointer: drag rotates (if the view supports it), a click without movement selects a pixel.
  let pointer: { x: number; y: number; moved: boolean } | undefined;
  canvas.addEventListener('pointerdown', event => { pointer = { x: event.clientX, y: event.clientY, moved: false }; canvas.setPointerCapture(event.pointerId); }, { signal });
  canvas.addEventListener('pointermove', event => {
    if (!pointer) return;
    const dx = event.clientX - pointer.x; const dy = event.clientY - pointer.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) pointer.moved = true;
    if (pointer.moved && view.drag) { view.drag(dx, dy); pointer.x = event.clientX; pointer.y = event.clientY; schedule(); }
  }, { signal });
  canvas.addEventListener('pointerup', event => {
    if (pointer && !pointer.moved) { const bounds = canvas.getBoundingClientRect(); state.selected = [Math.max(0, Math.min(0.999, (event.clientX - bounds.left) / bounds.width)), Math.max(0, Math.min(0.999, (event.clientY - bounds.top) / bounds.height))]; schedule(); }
    pointer = undefined;
  }, { signal });
  canvas.addEventListener('pointercancel', () => { pointer = undefined; }, { signal });
  canvas.addEventListener('wheel', event => { if (!view.zoom) return; event.preventDefault(); view.zoom(event.deltaY > 0 ? 1.06 : 1 / 1.06); schedule(); }, { passive: false, signal });
  canvas.addEventListener('keydown', event => {
    const step = 0.04;
    if (event.key === '+' || event.key === '=') view.zoom?.(1 / 1.1);
    else if (event.key === '-') view.zoom?.(1.1);
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { if (view.drag) view.drag(event.key === 'ArrowLeft' ? -18 : 18, 0); else state.selected[0] = Math.max(0, Math.min(0.999, state.selected[0] + (event.key === 'ArrowLeft' ? -step : step))); }
    else if (event.key === 'ArrowUp' || event.key === 'ArrowDown') state.selected[1] = Math.max(0, Math.min(0.999, state.selected[1] + (event.key === 'ArrowUp' ? -step : step)));
    else return;
    event.preventDefault(); schedule();
  }, { signal });
  const resize = new ResizeObserver(() => schedule()); resize.observe(canvas); cleanups.push(() => resize.disconnect());

  if (mode === 'synctalk' || mode === 'instag') wireClips(root, view, schedule, cleanups);
  schedule();
}

// Clip playback: the reference video is the clock; the render clip, audio and frame slider follow it.
function wireClips(root: Root, view: LabView, schedule: () => void, cleanups: (() => void)[]) {
  const $ = <T extends Element = HTMLElement>(selector: string) => root.element.querySelector<T>(selector);
  const reference = $<HTMLVideoElement>('[data-reference]')!; const rendered = $<HTMLVideoElement>('[data-rendered]')!; const audio = $<HTMLAudioElement>('[data-audio]')!;
  const play = $<HTMLButtonElement>('[data-play]')!; const sound = $<HTMLInputElement>('[data-sound]')!; const slider = $<HTMLInputElement>('input[name=position]')!;
  const fps = 25; let playing = false; let seeking = false;
  const seek = (frame: number) => { seeking = true; const time = (frame + 0.5) / fps; for (const media of [reference, rendered, audio]) if (Number.isFinite(media.duration) || media.readyState >= 1) media.currentTime = Math.min(time, (media.duration || time) - 0.02); seeking = false; };
  const pause = () => { playing = false; reference.pause(); rendered.pause(); audio.pause(); play.textContent = 'Phát'; play.setAttribute('aria-pressed', 'false'); };
  play.addEventListener('click', async () => {
    if (playing) { pause(); return; }
    if (reference.currentTime >= reference.duration - 0.05) { slider.value = '0'; view.setFrame?.(0); seek(0); }
    playing = true; play.textContent = 'Tạm dừng'; play.setAttribute('aria-pressed', 'true'); audio.muted = !sound.checked;
    try { await Promise.all([reference.play(), rendered.play(), audio.play()]); } catch { pause(); }
  }, { signal: root.signal });
  sound.addEventListener('change', () => { audio.muted = !sound.checked; }, { signal: root.signal });
  reference.addEventListener('timeupdate', () => {
    // Only playback drives the slider. A timeupdate that arrives late from an earlier seek must not overwrite
    // a value the user (or Đặt lại) has set since.
    if (seeking || !playing) return;
    const frame = Math.max(0, Math.min(249, Math.floor(reference.currentTime * fps)));
    if (Math.abs(rendered.currentTime - reference.currentTime) > 0.08) rendered.currentTime = reference.currentTime;
    if (Math.abs(audio.currentTime - reference.currentTime) > 0.08) audio.currentTime = reference.currentTime;
    if (Number(slider.value) !== frame) { slider.value = String(frame); root.element.querySelector('[data-value="position"]')!.textContent = String(frame); view.setFrame?.(frame); schedule(); }
  }, { signal: root.signal });
  reference.addEventListener('ended', pause, { signal: root.signal });
  rendered.addEventListener('seeked', schedule, { signal: root.signal });
  slider.addEventListener('input', () => { pause(); seek(Number(slider.value)); }, { signal: root.signal });
  // Đặt lại restores the slider default (handled in start); bring the clips to that frame too.
  $('[data-reset]')!.addEventListener('click', () => { pause(); seek(Number(slider.value)); }, { signal: root.signal });
  for (const media of [reference, rendered, audio]) media.addEventListener('loadedmetadata', () => { if (!playing) seek(Number(slider.value)); }, { signal: root.signal });
  cleanups.push(pause);
}
