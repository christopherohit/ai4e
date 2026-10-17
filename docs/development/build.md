# Development

Run workspace commands from the repository root:

```sh
node scripts/workspace.mjs verify
node scripts/workspace.mjs check
node scripts/workspace.mjs test
node scripts/workspace.mjs build
```

Rebuild a source tool after modifying `tools/`:

```sh
node scripts/workspace.mjs build:tools cnn-explainer --skip-install
```

Recipes copy source into `blog/.viz-build/` before applying integration changes.
Then rebuild the blog to refresh adapters and deployment assets. Building
Transformer requires its dependencies in the patched build copy.

Browser tests: `cd blog && npm run test:e2e` after a completed build. Do not run
tests against the deployment directory while a build is replacing its files.

## Browser tests behind the office proxy

With HTTP_PROXY/HTTPS_PROXY set, Playwright's webServer check reports http://127.0.0.1:4322/ as "already used" even
when nothing listens there, and refuses to start. Run browser tests without the proxy variables:

    env -u http_proxy -u HTTP_PROXY -u https_proxy -u HTTPS_PROXY npx playwright test

On this machine Docker creates and removes many virtual interfaces; Chrome can then abort same-origin requests with
net::ERR_NETWORK_CHANGED. The dodrio smoke test is the most exposed (many small figure requests); rerun it before
treating it as a regression.
