import { scenarios, type Scenario } from './model';
export const explainerKind = 'educational-simulation';
export function initBswmd() {
  const root = document.querySelector<HTMLElement>('[data-bswmd]'); if (!root) return;
  const result = root.querySelector<HTMLElement>('[data-result]')!;
  const draw = (key: Scenario) => { const s = scenarios[key]; root.querySelectorAll<HTMLButtonElement>('button').forEach(b => { const selected = b.dataset.scenario === key; b.classList.toggle('active', selected); b.setAttribute('aria-pressed', String(selected)); }); result.innerHTML = `<div class="result-top"><span class="status ${s.status.toLowerCase()}">${s.status}</span><code>${s.authority}</code></div><h3>${s.label}: DcmDspSecurityLevel = ${s.value}</h3><p>${s.message}</p><div class="facets">${s.facets.map(f => `<span>${f}</span>`).join('')}</div><p class="note">Mô phỏng deterministic: không đọc hoặc ghi file, không gọi parser thật.</p>`; };
  root.querySelectorAll<HTMLButtonElement>('button').forEach(b => b.addEventListener('click', () => draw((b.dataset.scenario || 'valid') as Scenario))); draw('valid');
  root.setAttribute('aria-busy', 'false');
}
