export const nativeTools = {
  'gradient-descent-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/gradient-descent-explainer', article: 'gradient-descent' },
  'mini-batch-sgd-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/mini-batch-sgd-explainer', article: 'batch-mini-batch-sgd' },
  'qwen3-8-flash-next-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/qwen3-8-flash-next-explainer', article: 'qwen3-8-flash-next' },
  'gated-delta-net-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/gated-delta-net-explainer', article: 'gated-delta-net' },
  'sparse-attention-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/sparse-attention-explainer', article: 'sparse-attention' },
  'nerf-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/nerf-explainer', article: 'nerf' },
  'gaussian-splatting-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/gaussian-splatting-explainer', article: 'gaussian-splatting' },
  'sdvx-modeler': { category: 'du-an', source: 'tools/du-an/sdvx-modeler', article: 'sdvx-modeler' },
  bswmd: { category: 'du-an', source: 'tools/du-an/bswmd', article: 'bswmd' },
  synctalk: { category: 'du-an', source: 'tools/du-an/synctalk', article: 'synctalk' },
  instag: { category: 'du-an', source: 'tools/du-an/instag', article: 'instag' },
};

export function validToolName(category, slug) {
  return !slug || category !== 'kien-thuc' || slug === 'ganlab' || slug.endsWith('-explainer');
}
