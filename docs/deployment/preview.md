# Preview And Tunnel

```sh
node scripts/workspace.mjs preview
```

The preview listens on `127.0.0.1:4321`. The existing remote tunnel forwards this
port to `https://interesting-field-8smklgt-4321.asse.devtunnels.ms/`.

Do not start another preview if port 4321 is already occupied. Rebuilding
`blog/dist/` updates the existing preview; a build briefly replaces its files.
No change to the 9router service or its port 20128 is needed.
