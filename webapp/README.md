# Ollama Tunnel Console

This Vite app is a static frontend for calling an Ollama API exposed through the zrok tunnel scripts in `../deploy/zrok/`.

It does not hardcode the zrok URL or API key. Paste them into the UI at runtime after running the host script.

## Develop

```bash
npm ci
npm run dev
```

## Build

```bash
npm run build
```

Deploy `dist/` to any static host.

## Host API

On the machine running Ollama:

```bash
../deploy/zrok/expose-ollama-zrok.sh
```

Or on Windows:

```powershell
..\deploy\zrok\expose-ollama-zrok.ps1
```

If the deployed browser cannot reach the endpoint, restart Ollama with `OLLAMA_ORIGINS` set to the deployed frontend origin. See `../docs/LOCAL_MODEL_TUNNEL.md`.
