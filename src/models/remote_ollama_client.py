"""
Remote Ollama client for Style Transfer AI.

Connects to a remote Ollama instance (e.g. via ngrok/zrok tunnel) and provides
the same `analyze_with_*` interface as the local ollama_client so it can be used
as a drop-in model option throughout analysis and generation pipelines.
"""

import json
import time
import requests
from typing import Tuple, List, Optional

# Default headers for tunnel proxies (skip browser warnings)
_HEADERS = {
    "ngrok-skip-browser-warning": "1",
    "Content-Type": "application/json",
}

# Module-level state — set once via setup, reused by analyze calls
_REMOTE_URL: Optional[str] = None
_REMOTE_MODEL: Optional[str] = None


# ---------------------------------------------------------------------------
# Setup / connection helpers
# ---------------------------------------------------------------------------

def setup_remote_ollama(url: str = None) -> Tuple[bool, str]:
    """
    Configure and validate the remote Ollama endpoint.

    Args:
        url: Full base URL of the remote Ollama instance
             (e.g. 'https://myollamaapi2000.share.zrok.io').
             If None, prompts the user interactively.

    Returns:
        (success, message)
    """
    global _REMOTE_URL

    if not url:
        default = "https://myollamaapi2000.share.zrok.io"
        url = input(f"\nEnter remote Ollama URL [{default}]: ").strip().rstrip("/") or default

    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        if r.status_code not in (200, 404):
            return False, f"Remote Ollama returned unexpected status {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot reach remote Ollama at {url}"
    except requests.exceptions.Timeout:
        return False, f"Connection to {url} timed out"

    _REMOTE_URL = url
    return True, f"Connected to remote Ollama at {url}"


def list_remote_models() -> Tuple[List[str], Optional[str]]:
    """
    List models available on the remote Ollama instance.

    Returns:
        (model_names, error_message)
    """
    if not _REMOTE_URL:
        return [], "Remote Ollama URL not configured. Run setup_remote_ollama() first."

    try:
        r = requests.get(f"{_REMOTE_URL}/api/tags", headers=_HEADERS, timeout=15)
        if r.status_code != 200:
            return [], f"Remote Ollama returned status {r.status_code}: {r.text[:300]}"
        models = [m["name"] for m in r.json().get("models", [])]
        return models, None
    except requests.exceptions.ConnectionError:
        return [], f"Cannot reach remote Ollama at {_REMOTE_URL}"
    except Exception as e:
        return [], f"Error listing remote models: {e}"


def select_remote_model() -> Tuple[Optional[str], str]:
    """
    Interactively select a model from the remote Ollama instance.

    Returns:
        (selected_model_name, message)
    """
    global _REMOTE_MODEL

    models, error = list_remote_models()
    if error:
        return None, error
    if not models:
        return None, "No models found on the remote Ollama instance."

    print("\nRemote Ollama Models:")
    for i, name in enumerate(models, 1):
        print(f"  [{i}] {name}")

    while True:
        choice = input("\nSelect a model (number or name): ").strip()
        if choice.isdigit() and 0 < int(choice) <= len(models):
            _REMOTE_MODEL = models[int(choice) - 1]
            return _REMOTE_MODEL, f"Selected remote model: {_REMOTE_MODEL}"
        if choice in models:
            _REMOTE_MODEL = choice
            return _REMOTE_MODEL, f"Selected remote model: {_REMOTE_MODEL}"
        print("Invalid selection. Try again.")


def check_remote_connection(model_name: str = None) -> Tuple[bool, str]:
    """
    Check if the remote Ollama instance is reachable and (optionally) if a
    specific model is available.

    Returns:
        (is_available, message)
    """
    if not _REMOTE_URL:
        return False, "Remote Ollama URL not configured"

    models, error = list_remote_models()
    if error:
        return False, error
    if not models:
        return False, "Remote Ollama has no models loaded"

    if model_name:
        found = any(model_name in m or m.startswith(model_name.split(":")[0]) for m in models)
        if found:
            return True, f"Remote model '{model_name}' is available"
        return False, f"Model '{model_name}' not found on remote. Available: {', '.join(models)}"

    return True, "Remote Ollama is reachable"


def get_remote_url() -> Optional[str]:
    """Return the currently configured remote URL (or None)."""
    return _REMOTE_URL


def get_selected_remote_model() -> Optional[str]:
    """Return the currently selected remote model name (or None)."""
    return _REMOTE_MODEL


def set_remote_model(model_name: str):
    """Programmatically set the remote model name."""
    global _REMOTE_MODEL
    _REMOTE_MODEL = model_name


# ---------------------------------------------------------------------------
# Core inference — drop-in replacement for analyze_with_ollama
# ---------------------------------------------------------------------------

def analyze_with_remote_ollama(
    prompt: str,
    model_name: str = None,
    processing_mode: str = "fast",
    *,
    max_retries: int = 4,
) -> str:
    """
    Send a prompt to the remote Ollama instance and return the text response.

    This mirrors the signature of ``analyze_with_ollama`` so it can be used
    interchangeably in the analysis and generation pipelines.

    Args:
        prompt:          The analysis/generation prompt.
        model_name:      Model to use on the remote server. Falls back to
                         the module-level ``_REMOTE_MODEL`` if not provided.
        processing_mode: 'fast', 'statistical', or 'enhanced' (used for temperature/token
                         config, same as local Ollama).
        max_retries:     Number of retry attempts on transient errors.

    Returns:
        The model's text response as a string.

    Raises:
        RuntimeError: If the remote instance is unreachable or all retries fail.
    """
    if not _REMOTE_URL:
        raise RuntimeError("Remote Ollama URL not configured. Call setup_remote_ollama() first.")

    model = model_name or _REMOTE_MODEL
    if not model:
        raise RuntimeError("No remote model selected. Call select_remote_model() first.")

    # Get settings from PROCESSING_MODES
    from ..config.settings import PROCESSING_MODES
    mode = PROCESSING_MODES.get(processing_mode, PROCESSING_MODES["fast"])
    temperature = mode.get("temperature", 0.3)
    
    # Token limit depends on model family
    num_predict = mode.get("gemma_tokens", 900)

    # Build payload — use /api/generate (single-shot, not chat)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    for attempt in range(1, max_retries + 1):
        try:
            with requests.post(
                f"{_REMOTE_URL}/api/generate",
                headers=_HEADERS,
                json=payload,
                stream=True,
                timeout=300,
            ) as r:
                if r.status_code == 504:
                    # Gateway timeout — back off and retry
                    time.sleep(15 * attempt)
                    continue
                if not r.ok:
                    raise RuntimeError(
                        f"Remote Ollama error {r.status_code}: {r.text[:300]}"
                    )

                # Stream response tokens
                full_response = ""
                for line in r.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("response", "")
                            full_response += token
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            pass

                return full_response

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(15 * attempt)
                continue
            raise RuntimeError(f"Remote Ollama timed out after {max_retries} attempts")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot reach remote Ollama at {_REMOTE_URL}")

    raise RuntimeError(f"Remote Ollama failed after {max_retries} retries")
