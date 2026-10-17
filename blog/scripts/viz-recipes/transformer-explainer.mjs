// Build recipe for /viz/transformer-explainer/ — field docs in ../viz-recipes.mjs.
/** @type {any} */
export default {
  slug: "transformer-explainer",
  src: "tools/kien-thuc/transformer-explainer",
  replacements: [
    { file: "svelte.config.js", from: "'/transformer-explainer'", to: "'/viz/transformer-explainer'" },
    { file: "package.json", from: '"vite": "^5.4.21"', to: '"vite": "6.3.6"' },
    { file: "vite.config.ts", from: 'scss: {', to: 'scss: {\n                loadPaths: [process.cwd()],' },
    { file: "src/components/WeightPopovers.svelte", from: "./popovers/", to: "./Popovers/", all: true },
    { file: "src/components/LinearSoftmax.svelte", from: "./popovers/", to: "./Popovers/", all: true },
    { file: "src/components/OperationGroup.svelte", from: "./popovers/", to: "./Popovers/", all: true },
    { file: "src/routes/+page.ts", from: 'export const prerender = true;', to: 'export const prerender = true;\nexport const ssr = false;' },
    { file: "src/routes/+page.svelte", from: "import { AutoTokenizer } from '@xenova/transformers';", to: "import { AutoTokenizer, env } from '@xenova/transformers';\n\tenv.localModelPath = `${base}/models/`;\n\tenv.allowRemoteModels = false;" },
    { file: "src/routes/+page.svelte", from: "'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/'", to: "`${base}/onnxruntime/`" },
    {
      file: "src/utils/fetchChunks.js",
      from: `const fetchPromises = chunkUrls.map((url, index) => {
		if (!cachedResponses[index]) {
			// console.log(\`Fetching and caching: \${url}\`);
			return fetch(url).then((response) => {
				if (response.ok) {
					cache.put(url, response.clone());
					return response.arrayBuffer();
				} else {
					throw new Error(\`Failed to fetch \${url}\`);
				}
			});
		} else {
			hasCache = true;
			// console.log(\`Using cached version: \${url}\`);
			return cachedResponses[index].arrayBuffer();
		}
	});

	const modelBuffers = await Promise.all(fetchPromises);`,
      to: `const modelBuffers = new Array(chunkUrls.length);
	let nextIndex = 0;
	const fetchNext = async () => {
		while (nextIndex < chunkUrls.length) {
			const index = nextIndex++;
			const url = chunkUrls[index];
			if (cachedResponses[index]) {
				hasCache = true;
				modelBuffers[index] = await cachedResponses[index].arrayBuffer();
				continue;
			}
			const response = await fetch(url);
			if (!response.ok) throw new Error(\`Failed to fetch \${url}\`);
			await cache.put(url, response.clone());
			modelBuffers[index] = await response.arrayBuffer();
		}
	};
	await Promise.all(Array.from({ length: Math.min(6, chunkUrls.length) }, fetchNext));`,
    },
  ],
  install: "npm install --include=dev --ignore-scripts --no-audit --no-fund",
  build: "mkdir -p static/models/Xenova/gpt2 static/onnxruntime && curl -fsSL --retry 2 --max-time 60 https://huggingface.co/Xenova/gpt2/resolve/main/tokenizer.json -o static/models/Xenova/gpt2/tokenizer.json && curl -fsSL --retry 2 --max-time 60 https://huggingface.co/Xenova/gpt2/resolve/main/tokenizer_config.json -o static/models/Xenova/gpt2/tokenizer_config.json && cp node_modules/onnxruntime-web/dist/*.mjs node_modules/onnxruntime-web/dist/*.wasm static/onnxruntime/ && npm run build",
  env: { NODE_ENV: "production", NODE_EXTRA_CA_CERTS: "/etc/ssl/certs/ca-certificates.crt" },
  out: "build",
  verified: true,
  notes: "Verified 2026-10-03: SvelteKit/adapter-static, 738MB output under /viz/transformer-explainer/. Build-copy fixes: Vite 6.3.6 matches plugin peer dependency; Sass loadPaths includes project root; Popovers imports match Linux case; browser-only SSR disabled; dev dependencies included and unused native sharp install skipped. Build downloads GPT-2 tokenizer/config from Hugging Face, sets localModelPath, disables remote tokenizer fallback and copies ONNX WASM/modules from npm to /onnxruntime/. Model chunks already self-hosted. Full main view and model readiness confirmed after 60s: zero failed same-origin requests and zero console errors. First downloads may take time (600MB model); transient host network-change errors occurred on an earlier run but clean retry passed. External Google Fonts and optional article embeds remain; no inference backend required."
};
