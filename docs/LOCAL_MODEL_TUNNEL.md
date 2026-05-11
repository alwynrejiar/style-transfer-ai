# Local Model Tunnel Runbook

This runbook turns a host machine running Ollama into an authenticated HTTPS API endpoint through zrok, then uses the deployable Vite frontend in `webapp/` to call it.

## Host Machine

Run these commands on the machine that owns the local models.

### Bash

```bash
chmod +x deploy/zrok/expose-ollama-zrok.sh
./deploy/zrok/expose-ollama-zrok.sh
```

### PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\deploy\zrok\expose-ollama-zrok.ps1
```

The script checks for `ollama` and `zrok`, prompts for a zrok enable token when needed, starts `ollama serve` if Ollama is installed but not reachable, generates a strong Basic Auth password, starts the zrok share in the background, and prints:

- Public zrok URL
- Basic Auth username
- API key / Basic Auth password
- `curl` commands for `/api/tags` and `/api/generate`
- Tunnel PID and log path

If the default state/log folder is not writable, set `OLLAMA_ZROK_STATE_DIR` to a folder you own before running the script.

## Browser CORS

`curl` can call the zrok URL immediately. A deployed browser frontend may also need Ollama to allow that frontend origin.

Prefer an exact origin:

```bash
OLLAMA_ORIGINS=https://your-frontend.example.com ./deploy/zrok/expose-ollama-zrok.sh
```

```powershell
$env:OLLAMA_ORIGINS = "https://your-frontend.example.com"
.\deploy\zrok\expose-ollama-zrok.ps1
```

If Ollama is already running as a desktop app or service, restart Ollama with the same `OLLAMA_ORIGINS` value before running the tunnel script again.

## Deploy The Frontend

The `webapp/` frontend is static. It does not compile the zrok URL or API key into the build. Users paste those values at runtime.

```bash
cd webapp
npm ci
npm run build
```

Deploy `webapp/dist/` to any static host. Open the deployed page, paste the zrok URL, username, and API key from the host script, then click `Check`.

## Stop The Tunnel

The script prints the tunnel PID.

Bash:

```bash
kill <PID>
```

PowerShell:

```powershell
Stop-Process -Id <PID>
```

## Security Notes

- Do not commit the generated API key.
- Do not bake the API key into a public frontend build.
- Anyone with both the public zrok URL and API key can use the exposed Ollama API while the tunnel is running.
- On shared hosts, same-user processes or local admins may be able to inspect command arguments. Use a machine/account you control.
- zrok public shares are ephemeral by default; stopping the `zrok share` process removes access.

## Troubleshooting

- `zrok is not installed`: install zrok and open a new terminal so PATH updates apply.
- `Ollama is installed but not reachable`: run `ollama serve`, then open `http://127.0.0.1:11434/api/tags`.
- `Basic Auth rejected`: rerun the host script and paste the newest API key into the frontend.
- `Browser could not reach endpoint`: verify the zrok process is still running, then check `OLLAMA_ORIGINS`.
- Model request fails: run `ollama pull gemma3:1b` or select a model returned by `/api/tags`.
