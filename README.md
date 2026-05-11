# Stylomex / Style Transfer AI

Style Transfer AI is a stylometry and style-transfer project with a FastAPI backend, a static browser app, and legacy CLI tooling.

Current primary runtime:
- FastAPI server in `api.py`
- Static app in `app/` served at `/app`
- Static docs site in `docs/` served at `/docs`

Legacy tooling still exists:
- CLI entry via `python scripts/run.py`
- Package console script `style-transfer-ai`

## Current Architecture

- API server: `api.py`
- Core Python package: `src/`
- Browser app (vanilla JS): `app/`
- Docs/marketing static site: `docs/`
- Optional Vite React frontend workspace: `webapp/`

## Run Locally (Primary)

### 1. Create and activate a Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r install/requirements.txt
pip install fastapi uvicorn python-dotenv supabase
```

### 3. Optional local model setup (Ollama)

```bash
ollama serve
ollama pull gemma3:1b
```

### 4. Start API + static app

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open:
- App: http://127.0.0.1:8000/app
- Docs page: http://127.0.0.1:8000/docs/index.html
- OpenAPI Swagger: http://127.0.0.1:8000/docs

## API Surface

All responses follow:

```json
{ "success": true, "data": {}, "error": null }
```

### Auth
- `POST /api/auth/signup`
- `POST /api/auth/signin`
- `POST /api/auth/google/start`
- `POST /api/auth/signout`
- `GET /api/auth/me`
- `POST /api/auth/password`
- `POST /api/account/delete`

### Analysis / Generation
- `POST /api/analyze` (NDJSON stream)
- `POST /api/generate` (text stream)
- `POST /api/analogy`
- `POST /api/transfer`

### Profiles / Comparisons
- `GET /api/profiles`
- `POST /api/profiles`
- `GET /api/profiles/{profile_id}`
- `DELETE /api/profiles/{profile_id}`
- `POST /api/comparisons`
- `GET /api/comparisons`

### Health
- `GET /api/health`

## Frontend Notes

### Primary app (`app/`)

`app/` is plain JavaScript and is served directly by FastAPI. It does not require a frontend build step.

### Local model tunnel + deployable frontend

Use `deploy/zrok/expose-ollama-zrok.sh` or `deploy/zrok/expose-ollama-zrok.ps1` to expose a host machine's local Ollama API through an authenticated zrok tunnel. The optional Vite frontend in `webapp/` is now an Ollama Tunnel Console that can be deployed as static files and configured at runtime with the zrok URL and generated API key.

See `docs/LOCAL_MODEL_TUNNEL.md` for the full runbook.

### Recent UI updates (`app/`)

- Generate and Transfer page now uses an open layout (no outer form card), aligned with Analyze page content width.
- Generate page form structure is standardized:
  - Top two-row, two-column layout:
    - `Style Profile | Content Type`
    - `Desired Tone | Word Count`
  - `Topic / Subject`, `Additional Context`, `Generate Content`, and `Generated Output` remain full width.
- Generate page inputs/textareas now use consistent, more visible outlines in light mode; dark mode styling remains unchanged.
- Generate textareas (`Topic / Subject`, `Additional Context`) are fixed-height and non-resizable.
- Analyze and Compare pages had background panel/card wrappers removed behind primary form sections for a cleaner open layout.
- Compare page refinements:
  - Larger radio controls for mode selection visibility.
  - Mode label renamed from `New Text Data` to `Text Box`.
  - Textareas are fixed-height and non-resizable.
  - Extra spacing added before the empty-state message.
- Page subtitles/descriptions were removed from:
  - Analyze
  - Generate
  - Compare
  - Student Analogy
  - Profiles
  - Settings
- Student Analogy page title updated from `Student Analogy (v2)` to `Student Analogy`.
- Additional heading-to-form vertical spacing was added for Analyze, Generate, and Compare pages.

### Optional Vite app (`webapp/`)

There is also a separate React + TypeScript + Vite workspace in `webapp/`.

```bash
cd webapp
npm install
npm run dev
npm run build
```

This is separate from the FastAPI-served `app/` experience and is intended for direct calls to an authenticated Ollama/zrok endpoint.

## Legacy CLI

Legacy CLI remains available:

```bash
python scripts/run.py
```

If installed as a package:

```bash
style-transfer-ai
```

## Project Structure

```text
style-transfer-ai/
|- api.py
|- app/
|- docs/
|- scripts/
|  |- run.py
|  |- style_analyzer_enhanced.py
|- src/
|  |- analysis/
|  |- config/
|  |- database/
|  |- generation/
|  |- models/
|  |- main.py
|- install/
|- webapp/
|- README.md
```

## Configuration

Supabase is optional but required for authenticated profile/content persistence features.

Create `.env` in the project root when using Supabase:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
```

## Development Notes

- Use constants from `src/config/settings.py` for model and mode values.
- Keep API responses consistent with the `{ success, data, error }` shape.
- For UI feature additions in the primary app, wire routes in `app/js/router.js` and nav links in `app/js/components/navbar.js`.

## License

See `LICENSE`.
