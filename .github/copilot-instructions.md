# GitHub Copilot Instructions - Style Transfer AI

## Architecture Overview

Current product surface is a FastAPI backend plus a static vanilla JS app.

- Core library: `src/`
- API server: `api.py`
- Browser app (sidebar UI): `app/`
- Public marketing/docs: `docs/`

## Active Entry Points

- API + app runtime:
  - `python -m uvicorn api:app --host 127.0.0.1 --port 8000`
  - or `python run.py`
  - App URL: `/app`
  - Docs URL: `/docs/index.html`

## Current Web App Structure

- Shell: `app/index.html`
- Styling: `app/css/app.css` (imports `docs/css/style.css`)
- Router: `app/js/router.js` (hash routes)
- Auth/session: `app/js/auth.js`
- API client: `app/js/api.js`
- Sidebar nav: `app/js/components/navbar.js`
- Pages:
  - `app/js/pages/analyze.js`
  - `app/js/pages/studentAnalogy.js`
  - `app/js/pages/profiles.js`
  - `app/js/pages/generate.js`
  - `app/js/pages/compare.js`
  - `app/js/pages/settings.js`

## FastAPI Surface (`api.py`)

### Auth
- `POST /api/auth/signup`
- `POST /api/auth/signin`
- `POST /api/auth/signout`

### Analysis / Generation
- `POST /api/analyze` (NDJSON stream)
  - event types: `pass`, `heartbeat`, `result`, `error`
- `POST /api/generate` (text stream)
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

## Core Library Map (`src/`)

| Module | Purpose | Key exports |
|---|---|---|
| `analysis/analyzer.py` | 7-pass stylometric analysis orchestration | `analyze_style`, `create_enhanced_style_profile` |
| `analysis/metrics.py` | readability + deep stylometry + similarity | `calculate_style_similarity`, `extract_deep_stylometry` |
| `analysis/analogy_engine.py` | cognitive bridging support | `detect_conceptual_density`, `AnalogyInjector` |
| `generation/content_generator.py` | style-conditioned generation | `ContentGenerator.generate_content` |
| `generation/style_transfer.py` | rewriting/transfer | `StyleTransfer.transfer_style` |
| `database/auth.py` | Supabase auth wrappers | `sign_in`, `sign_up`, `sign_out`, `get_current_user` |
| `database/db_analyses.py` | saved style profile CRUD | `save_analysis`, `list_analyses`, `get_analysis`, `delete_analysis` |
| `database/db_comparisons.py` | saved comparison CRUD | `save_comparison`, `list_comparisons` |
| `database/db_content.py` | generated content persistence | `save_generated_content` |
| `config/settings.py` | model/mode constants | `AVAILABLE_MODELS`, `PROCESSING_MODES`, URLs |

## Conventions and Rules

- Import constants from `src/config/settings.py`; do not hardcode model/mode constants.
- Keep API response shape consistent: `{ "success": bool, "data": ..., "error": ... }`.
- Preserve auth flow using bearer token from `/api/auth/signin`.
- `app/` is plain JS (no build tooling). Keep dependencies browser-native.
- Sidebar navigation is the primary feature launcher; new features should be routed via `router.js` and linked in `navbar.js`.
- Keep docs changes minimal; `docs/` is mostly read-only except explicit user requests.

## Running Locally

```bash
# dependencies
pip install fastapi uvicorn requests supabase spacy
python -m spacy download en_core_web_sm

# local model runtime
ollama serve
ollama pull gemma3:1b

# start app
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open:
- `http://127.0.0.1:8000/app`
- `http://127.0.0.1:8000/docs/index.html`

## Notes for Future Changes

- If a UI element appears unexpectedly at top-level, verify inherited styles from `docs/css/style.css` are overridden in `app/css/app.css`.
- Analysis can take over a minute depending on sample length and model speed; keep status feedback minimal unless the user asks for detailed progress.
- Student analogy/cognitive bridging should be implemented via transfer instructions and domain-specific simplification prompts.
