import { writable } from '../../.viz-build/transformer-explainer/node_modules/svelte/src/store/index-client.js';
export const page = writable({ url: new URL(window.location.href) });
