/** Prefix a root-relative URL with the deployment base (e.g. /ai4e on GitHub Pages); unchanged when served at /. */
export const withBase = (path: string) => `${import.meta.env.BASE_URL.replace(/\/$/, '')}${path}`;
