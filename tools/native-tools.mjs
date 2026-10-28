export const nativeTools = {
  'gradient-descent-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/gradient-descent-explainer', article: 'gradient-descent' },
  'naive-bayes-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/naive-bayes-explainer', article: 'naive-bayes' },
  'svm-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/svm-explainer', article: 'svm' },
  'bias-variance-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/bias-variance-explainer', article: 'bias-variance' },
  'regularization-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/regularization-explainer', article: 'l1-l2-regularization' },
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
  'medlab-body-atlas': { category: 'du-an', source: 'tools/du-an/medlab-body-atlas', article: 'ban-do-co-the-nguoi' },
  'activation-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/activation-explainer', article: 'ham-kich-hoat' },
  'pca-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/pca-explainer', article: 'pca' },
  'jev-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/jev-explainer', article: 'jev' },
  'paged-attention-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/paged-attention-explainer', article: 'paged-attention' },
  'inference-engineering-explainer': { category: 'kien-thuc', source: 'tools/kien-thuc/inference-engineering-explainer', article: 'inference-engineering' },
};

export function validToolName(category, slug) {
  return !slug || category !== 'kien-thuc' || slug === 'ganlab' || slug.endsWith('-explainer');
}
