# Stylomex / Style Transfer AI

Style Transfer AI is a stylometry and style-transfer project with a FastAPI backend and a static browser app.

Current primary runtime:
- FastAPI server in `api.py`
- Static app in `app/` served at `/app`
- Static docs site in `docs/` served at `/docs`

## Current Architecture

- API server: `api.py`
- Core Python package: `src/`
- Browser app: `app/`
- Docs/marketing static site: `docs/`
- Local persistence: `local_data/` (ignored by Git)
- Deployment helpers: `deploy/`
- Dependency manifest: `install/requirements.txt`

## Run Locally

### 1. Create and activate a Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r install/requirements.txt
```

### 3. Optional local model setup

```bash
ollama serve
ollama pull gemma3:1b
```

### 4. Start API + static app

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Alternative:

```bash
python run.py
```

Open:
- App: http://127.0.0.1:8000/app
- Docs page: http://127.0.0.1:8000/docs/index.html
- OpenAPI Swagger: http://127.0.0.1:8000/docs

## Achievements

### InApp - IEEE CS Student Project Awards 2026

![InApp IEEE CS Student Project Awards 2026 Elevator Pitch Round](docs/assets/images/achievements/ieee-cs-student-project-awards-2026.png)

### OpenAI Academy x NxtWave Buildathon

![OpenAI Academy x NxtWave Buildathon State-Level Qualification](docs/assets/images/achievements/openai-academy-nxtwave-state-level-buildathon.png)

## API Surface

All standard JSON responses follow:

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

`app/` is plain JavaScript and is served directly by FastAPI. It does not require a frontend build step.

Recent UI updates:
- Generate and Transfer page now use an open layout aligned with Analyze page content width.
- Generate, Analyze, and Compare forms have cleaner open sections with fixed-height textareas.
- Compare mode labels and spacing were refined.
- Page subtitles were removed from Analyze, Generate, Compare, Student Analogy, Profiles, and Settings.
- Student Analogy page title is `Student Analogy`.

## Project Structure

```text
style-transfer-ai/
|- api.py
|- run.py
|- app/
|- data/
|- deploy/
|- docs/
|- install/
|  |- requirements.txt
|- scripts/
|  |- generate_frontend_config.py
|- src/
|  |- analysis/
|  |- config/
|  |- database/
|  |- generation/
|  |- models/
|  |- utils/
|  |- local_store.py
|- tests/
|- README.md
```

## Configuration

Supabase is used only for authentication. Saved profiles, generated content, comparisons, and uploaded avatar files are stored locally under `local_data/` by default.

Create `.env` in the project root for authentication and optional local storage location:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
LOCAL_DATA_DIR=local_data
```

## Development Notes

- Use constants from `src/config/settings.py` for model and mode values.
- Keep API responses consistent with the `{ success, data, error }` shape.
- For UI feature additions in the primary app, wire routes in `app/js/router.js` and nav links in `app/js/components/navbar.js`.

## License

See `LICENSE`.
