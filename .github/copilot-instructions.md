# GitHub Copilot Instructions - Style Transfer AI

## Architecture Overview

Modular Python package (`src/`) with three interfaces: interactive CLI, Streamlit web GUI, and a desktop CustomTkinter GUI.

### Entry Points
- **CLI**: `python scripts/run.py` → `src.main:cli_entry_point` → `src/menu/main_menu.py:run_main_menu()`
- **Streamlit GUI**: `streamlit run app.py` → sidebar routing to `gui/*.py` page modules (each exposes `show()`)
- **Desktop GUI**: `python scripts/run_gui.py` → CustomTkinter app in `src/gui/app.py`
- **PyPI console script**: `style-transfer-ai` (after `pip install style-transfer-ai`)

### Module Layout (`src/`)
| Module | Purpose | Key exports |
|---|---|---|
| `config/settings.py` | All constants: `AVAILABLE_MODELS`, `PROCESSING_MODES`, API URLs, file paths | Import constants here; never hardcode |
| `models/` | One client per provider: `ollama_client.py`, `remote_ollama_client.py` | `analyze_with_*()`, `setup_*_client()` |
| `analysis/` | `analyzer.py` (orchestrator), `prompts.py` (25-point prompt), `metrics.py` (readability + Burrows' Delta), `analogy_engine.py` (cognitive bridging) | `analyze_style()`, `create_enhanced_style_profile()`, `detect_conceptual_density()`, `AnalogyInjector` |
| `generation/` | `ContentGenerator`, `StyleTransfer`, `QualityController`, `GenerationTemplates` | `generate_content()`, `transfer_style()`, `compare_styles()` |
| `storage/local_storage.py` | Dual-format save (JSON + TXT) with personalized filenames | `save_style_profile_locally()` |
| `menu/` | `main_menu.py` (~1600 lines, 14 menu options), `model_selection.py` (global state), `navigation.py` (UI helpers) | `run_main_menu()` |
| `utils/` | `text_processing.py` (file I/O, sanitization), `formatters.py` (report output), `user_profile.py` | Shared helpers |
| `gui/app.py` | Desktop CustomTkinter app: analyze, cognitive bridging, transfer, profiles, settings | `StyleTransferApp` |

### Data Flow
```
User Input → model_selection (global state) → analyzer.analyze_style()
  → prompts.create_enhanced_deep_prompt() → models/*_client.analyze_with_*()
  → metrics (readability, Burrows' Delta) → storage.save_style_profile_locally()
  → dual output: JSON + TXT in "stylometry fingerprints/"

# Optional cognitive bridging layer:
analyzer.create_enhanced_style_profile(analogy_augmentation=True)
  → analogy_engine.detect_conceptual_density() → AnalogyInjector.augment_analysis_result()
  → [Cognitive Note: …] blocks appended to output
```

## Model Integration Patterns

Two models in `AVAILABLE_MODELS`, both type `ollama`. Uniform call pattern:

```python
# Every model client: analyze_with_*(prompt, ...) → str
# Ollama:  requests.post("http://localhost:11434/api/generate", json=payload)
```

**Global model state** in `src/menu/model_selection.py`: module-level `USE_LOCAL_MODEL`, `SELECTED_MODEL`, `USER_CHOSEN_API_KEY`. Reset via `reset_model_selection()`.

**Adding a new model**: (1) Add to `AVAILABLE_MODELS` in `settings.py`, (2) create/update client in `src/models/`, (3) add `elif` in `analyzer.analyze_style()` and `model_selection.validate_model_selection()`.

## Key Conventions

- **File encoding**: Use `read_text_file()` from `src/utils/text_processing.py` — tries UTF-8 then latin-1
- **Output filenames**: `{name}_stylometric_profile_{YYYYMMDD_HHMMSS}.json/.txt`
- **Dual output**: Every analysis saves both JSON and TXT via `save_dual_format()`
- **API keys**: Placeholder strings `"your-openai-api-key-here"` in committed code; entered interactively at runtime
- **Circular import avoidance**: `src/__init__.py` has empty `__all__`; model clients are imported lazily inside `analyzer.analyze_style()`
- **Menu handlers**: Each CLI feature is a `handle_*()` function in `main_menu.py`, dispatched by `run_main_menu()`
- **Analogy engine**: Opt-in per request via `analogy_augmentation` param; global default in `ANALOGY_AUGMENTATION_ENABLED`. Analogies are always rendered as `[Cognitive Note: …]` blocks to preserve primary output
- **Desktop GUI**: `src/gui/app.py` provides a full CustomTkinter interface with threaded LLM calls; import via `from src.gui.app import StyleTransferApp`

## Running & Testing

```bash
# Core dependencies
pip install requests spacy streamlit plotly
python -m spacy download en_core_web_sm

# Local models (privacy-first default)
ollama serve
ollama pull gemma3:1b                 # fast

# Launch
python scripts/run.py                 # CLI
streamlit run app.py                  # Streamlit GUI
python scripts/run_gui.py             # Desktop GUI (needs customtkinter matplotlib pillow)

# Tests (import-validation scripts, not pytest)
python tests/test_modular_implementation.py
python tests/test_integration_complete.py
python tests/test_analogy_engine.py
```

Sample texts: `data/samples/`. Generated profiles: `stylometry fingerprints/`.

## Critical Details

- **`main_menu.py` is ~1600 lines** — all 14 CLI features as `handle_*()` functions (option 14 = Cognitive Bridging)
- **`analogy_engine.py`** — `detect_conceptual_density()` is pure Python (no LLM); `AnalogyInjector` batches dense sentences into a single LLM call
- **`metrics.py` auto-installs spaCy** at runtime via `_ensure_spacy_model()` if missing
- **Processing modes**: `"enhanced"` (25-point, temp 0.2, 180s) vs `"statistical"` (basic, temp 0.3, 120s) in `PROCESSING_MODES`
- **Ollama tokens**: `num_predict` uses `gemma_tokens` setting from `PROCESSING_MODES` (in `ollama_client.py`)
- **Analogy domains**: 7 built-in domains in `ANALOGY_DOMAINS` (sports, gaming, cooking, nature, daily_life, tech, general_simplification); default = `general_simplification`
- **Conceptual density threshold**: `CONCEPTUAL_DENSITY_THRESHOLD = 0.45` (0-1 range; most academic text scores 0.45-0.70)
- **PyInstaller**: `scripts/run.py` and `run_gui.py` detect `sys.frozen` and use `sys._MEIPASS` for bundled paths
