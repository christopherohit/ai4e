import { localizeTool } from './tool-locale';

interface Adapter {
  html: string;
  css: string;
  scripts: { src?: string; source?: string; type: string }[];
}

const readySelectors: Record<string, string> = {
  dodrio: '.graph-container svg', fairvis: '#root button', energyvis: '#energymap svg',
  viscuit: '#selection .subgroup-entry-image-display img', ganlab: 'gan-lab #iteration-count',
  'transformer-explainer': '.main-section.active .nodes',
  'diffusion-explainer': '#architecture-container img',
};

export async function mountNativeTool(root: HTMLElement, slug: string, status: HTMLElement) {
  let readyObserver: MutationObserver | undefined;
  root.addEventListener('keydown', event => {
    if (event.target !== root || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    root.scrollLeft += event.key === 'ArrowRight' ? 80 : -80;
  });
  try {
    const response = await fetch(`/native/${slug}/adapter.json`);
    if (!response.ok) throw new Error(`Adapter ${response.status}`);
    const adapter: Adapter = await response.json();
    const style = document.createElement('style');
    style.textContent = `@scope (#native-tool-root) { ${adapter.css} }`;
    root.appendChild(style);
    const content = document.createElement('div');
    content.className = 'native-content';
    content.innerHTML = adapter.html;
    root.appendChild(content);
    let sizingFrame = 0;
    let previousWidth = -1;
    const fitContent = () => {
      cancelAnimationFrame(sizingFrame);
      sizingFrame = requestAnimationFrame(() => {
        const available = root.clientWidth;
        const minimum = slug === 'ganlab' ? 1442 : slug === 'transformer-explainer' ? 1300 : 1200;
        const width = `${Math.max(available, minimum)}px`;
        if (content.style.width !== width) content.style.width = width;
        const contentHeight = slug === 'transformer-explainer' ? content.offsetHeight : content.scrollHeight;
        const height = `${Math.ceil(contentHeight) + 8}px`;
        if (root.style.height !== height) root.style.setProperty('height', height, 'important');
      });
    };
    const sizingObserver = new ResizeObserver(entries => {
      const contentChanged = entries.some(entry => entry.target === content);
      const widthChanged = root.clientWidth !== previousWidth;
      previousWidth = root.clientWidth;
      if (contentChanged || widthChanged) fitContent();
    });
    sizingObserver.observe(content);
    sizingObserver.observe(root);
    window.addEventListener('pagehide', () => {
      sizingObserver.disconnect();
      cancelAnimationFrame(sizingFrame);
    }, { once: true });
    localizeTool(root);
    const ready = () => {
      if (root.querySelector(readySelectors[slug])) {
        root.setAttribute('aria-busy', 'false');
        status.hidden = true;
        readyObserver?.disconnect();
      }
    };
    readyObserver = new MutationObserver(ready);
    readyObserver.observe(root, { childList: true, subtree: true });
    for (const descriptor of adapter.scripts) {
      if (descriptor.type && !['module', 'text/javascript', 'application/javascript'].includes(descriptor.type)) continue;
      await new Promise<void>((resolve, reject) => {
        const script = document.createElement('script');
        if (descriptor.type === 'module') script.type = 'module';
        if (descriptor.src) {
          script.src = descriptor.src;
          script.onload = () => resolve();
          script.onerror = () => reject(new Error(`Không tải được ${descriptor.src}`));
        } else {
          script.textContent = descriptor.source ?? '';
        }
        root.appendChild(script);
        if (!descriptor.src) resolve();
      });
    }
    ready();
    fitContent();
    window.setTimeout(() => {
      if (root.getAttribute('aria-busy') === 'true') {
        status.textContent = 'Công cụ vẫn đang tải dữ liệu. Vui lòng kiểm tra kết nối hoặc tải lại trang.';
      }
    }, 90000);
  } catch (error) {
    readyObserver?.disconnect();
    root.setAttribute('aria-busy', 'false');
    status.textContent = 'Không khởi động được công cụ. Vui lòng tải lại trang để thử lại.';
    console.error(error);
  }
}
