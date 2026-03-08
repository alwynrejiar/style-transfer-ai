"""
Wrapper for local Ollama API client — model management and inference.
"""

import json
import time
import requests
from typing import Tuple

from ..config.settings import OLLAMA_BASE_URL


def is_ollama_installed() -> bool:
    """Check if Ollama is installed and reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def check_ollama_connection(model_name: str = None) -> Tuple[bool, str]:
    """
    Check if Ollama is running and the specified model is available.

    Returns (is_available, message).
    """
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if r.status_code != 200:
            return False, f"Ollama returned status {r.status_code}"
        models = [m["name"] for m in r.json().get("models", [])]
        if model_name:
            # Check exact or prefix match (e.g. "gemma3:1b" matches "gemma3:1b")
            found = any(model_name in m or m.startswith(model_name.split(":")[0]) for m in models)
            if found:
                return True, f"Model '{model_name}' is available"
            return False, f"Model '{model_name}' not found. Available: {', '.join(models)}"
        return True, "Ollama is running"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Is it running? (ollama serve)"
    except Exception as e:
        return False, f"Ollama connection error: {e}"


def list_ollama_models() -> Tuple[list, str]:
    """
    List locally installed Ollama models.

    Returns (model_list, error_message).
    """
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if r.status_code != 200:
            return [], f"Ollama returned status {r.status_code}"
        models = [m["name"] for m in r.json().get("models", [])]
        return models, None
    except requests.exceptions.ConnectionError:
        return [], "Cannot connect to Ollama. Is it running?"
    except Exception as e:
        return [], str(e)


def pull_ollama_model(model_name: str) -> Tuple[bool, str]:
    """
    Pull/download an Ollama model.

    Returns (success, message).
    """
    try:
        print(f"Downloading {model_name}... (this may take a while)")
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=600,
        )
        if r.status_code != 200:
            return False, f"Pull failed with status {r.status_code}"
        for line in r.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "pulling" in status or "downloading" in status:
                        total = data.get("total", 0)
                        completed = data.get("completed", 0)
                        if total:
                            pct = int(completed / total * 100)
                            print(f"\r  {status} {pct}%", end="", flush=True)
                except json.JSONDecodeError:
                    pass
        print()
        return True, f"Successfully pulled {model_name}"
    except Exception as e:
        return False, f"Pull failed: {e}"


def analyze_with_ollama(
    prompt: str,
    model_name: str,
    processing_mode: str = "enhanced",
) -> str:
    """
    Send a prompt to a local Ollama model and return the text response.

    Args:
        prompt: The analysis/generation prompt.
        model_name: Ollama model name (e.g. 'gemma3:1b').
        processing_mode: 'enhanced' or 'statistical'.

    Returns:
        The model's text response.
    """
    from ..config.settings import PROCESSING_MODES

    mode = PROCESSING_MODES.get(processing_mode, PROCESSING_MODES["enhanced"])
    temperature = mode.get("temperature", 0.2)
    timeout = mode.get("timeout", 180)

    # Token limit depends on model family
    num_predict = mode.get("gemma_tokens", 2000)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,  # Enable streaming for progress feedback
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    try:
        import sys
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=(10, timeout),  # (connect_timeout, read_timeout per chunk)
        )
        if r.status_code != 200:
            raise RuntimeError(f"Ollama returned status {r.status_code}: {r.text[:300]}")
        
        # Stream response with progress dots and total time limit
        full_response = ""
        dot_count = 0
        start_time = time.time()
        max_total_time = timeout * 1.5  # Hard ceiling: 1.5x the per-chunk timeout
        
        for line in r.iter_lines():
            # Enforce total time limit (streaming timeout doesn't cover this)
            elapsed = time.time() - start_time
            if elapsed > max_total_time:
                print(f" [timeout after {int(elapsed)}s]", end="", flush=True)
                r.close()
                break
            
            if line:
                try:
                    chunk = json.loads(line)
                    if chunk.get("response"):
                        full_response += chunk["response"]
                        # Show progress dot every 10 tokens
                        dot_count += 1
                        if dot_count % 10 == 0:
                            print(".", end="", flush=True)
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
        
        elapsed = time.time() - start_time
        print(f" [{int(elapsed)}s]", end="", flush=True)
        
        if not full_response.strip():
            raise RuntimeError("Ollama returned an empty response")
        
        return full_response
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama request timed out after {timeout}s")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Is it running?")
