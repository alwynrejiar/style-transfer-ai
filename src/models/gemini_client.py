"""
Gemini API client for Style Transfer AI.
Version-safe: works with google-generativeai 0.3.x – 0.8.x+.
"""

from typing import Any, Tuple, Optional

import requests

# Version-safe generation config import
try:
    from google.generativeai.types import GenerationConfig
except ImportError:
    try:
        from google.generativeai import GenerationConfig
    except ImportError:
        GenerationConfig = None


# Cache a known-good model name once discovered, so subsequent calls avoid retries.
_RESOLVED_MODEL_NAME: Optional[str] = None

GEMINI_GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def _candidate_model_names(requested_model: str) -> list[str]:
    """Return an ordered list of model IDs to try for broad API compatibility."""
    ordered: list[str] = []

    def add(name: str):
        if name and name not in ordered:
            ordered.append(name)

    requested_model = _normalize_gemini_model_name(requested_model)

    # Prefer the requested model first.
    add(requested_model)

    # Common aliases/successors used across Gemini API versions.
    if requested_model == "gemini-1.5-flash":
        add("gemini-1.5-flash-latest")
    if requested_model == "gemini-1.5-pro":
        add("gemini-1.5-pro-latest")

    add("gemini-2.0-flash")
    add("gemini-2.5-flash")
    add("gemini-2.0-flash-exp")
    add("gemini-1.5-flash-latest")
    add("gemini-1.5-pro-latest")
    add("gemini-pro")

    return ordered


def _normalize_gemini_model_name(model_name: str | None) -> str:
    model = str(model_name or "").strip()
    if model.startswith("models/"):
        model = model.split("/", 1)[1]
    if model == "gemini":
        return "gemini-1.5-flash"
    return model or "gemini-1.5-flash"


def _is_model_not_found_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "model" in msg
        and (
            "unknown model" in msg
            or "model not found" in msg
            or "not found" in msg
            or "not supported" in msg
            or "404" in msg
        )
    )


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Gemini returned an empty response.")

    content = candidates[0].get("content") if isinstance(candidates[0], dict) else {}
    parts = content.get("parts") if isinstance(content, dict) else []
    if not isinstance(parts, list):
        raise RuntimeError("Gemini returned an empty response.")

    text = "".join(
        str(part.get("text", ""))
        for part in parts
        if isinstance(part, dict) and part.get("text")
    ).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty text response.")
    return text


def _extract_gemini_error(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("status") or "Gemini API error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    return "Gemini API error"


def _generate_with_gemini_rest(
    prompt: str,
    api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> str:
    global _RESOLVED_MODEL_NAME

    last_error = ""
    for candidate in _candidate_model_names(_RESOLVED_MODEL_NAME or model_name):
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        try:
            response = requests.post(
                GEMINI_GENERATE_CONTENT_URL.format(model=candidate),
                params={"key": api_key.strip()},
                headers={"Content-Type": "application/json"},
                json=body,
                timeout=120,
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("Gemini request timed out.") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Gemini network error: {exc}") from exc

        payload: dict[str, Any] = {}
        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if response.status_code in (401, 403):
            raise RuntimeError("Invalid Gemini API key.")

        if response.status_code >= 400:
            last_error = _extract_gemini_error(payload)
            if _is_model_not_found_error(Exception(last_error)):
                continue
            raise RuntimeError(f"Gemini API call failed ({response.status_code}): {last_error}")

        text = _extract_gemini_text(payload)
        _RESOLVED_MODEL_NAME = candidate
        return text

    raise RuntimeError(
        "Gemini API call failed: no compatible Gemini model worked. "
        f"Last error: {last_error or 'unknown model'}"
    )


def setup_gemini_client(api_key: str) -> Tuple[Optional[object], str]:
    """Initialise and return a configured Gemini model client."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Prefer a previously-resolved model to avoid repeated probing.
        if _RESOLVED_MODEL_NAME:
            return genai.GenerativeModel(_RESOLVED_MODEL_NAME), ""

        for name in _candidate_model_names("gemini-1.5-flash"):
            try:
                model = genai.GenerativeModel(name)
                return model, ""
            except Exception:
                continue

        return None, "No compatible Gemini model was found for this API key/version."
    except Exception as e:
        return None, f"Gemini setup failed: {e}"


def check_gemini_connection(api_key: str) -> Tuple[bool, str]:
    """Check if the Gemini API key works."""
    client, err = setup_gemini_client(api_key)
    if err:
        return False, err

    # Perform a minimal request so quota/billing/model issues are caught up front.
    try:
        _ = analyze_with_gemini(
            "Reply with exactly: OK",
            api_key_or_client=api_key,
            max_tokens=8,
            temperature=0.0,
        )
        return True, "Gemini API key is valid"
    except Exception as e:
        msg = str(e)
        if "429" in msg and "quota" in msg.lower():
            return False, (
                "Gemini quota exceeded for this API key/project. "
                "Please enable billing, wait for quota reset, or use a local Ollama model."
            )
        return False, f"Gemini connectivity test failed: {msg}"


def analyze_with_gemini(
    prompt: str,
    api_key_or_client=None,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.2,
    max_tokens: int = 3000,
    processing_mode: str = "fast",     # accepted but ignored (kept for compat)
) -> str:
    """
    Send a prompt to Gemini and return the text response.

    Args:
        prompt:             The analysis/generation prompt.
        api_key_or_client:  A raw API key string OR a pre-built GenerativeModel.
                            Falls back to GEMINI_API_KEY env var if None.
        model_name:         Gemini model to use (ignored when client passed).
        temperature:        Sampling temperature.
        max_tokens:         Maximum output tokens.
        processing_mode:    Accepted for compatibility, not used directly.
    """
    # Resolve api_key_or_client → model object
    if api_key_or_client is None:
        import os
        api_key_or_client = os.environ.get("GEMINI_API_KEY", "")

    if isinstance(api_key_or_client, str):
        if not api_key_or_client:
            raise RuntimeError(
                "Gemini API key is empty. Set GEMINI_API_KEY env var or pass the key explicitly."
            )
        return _generate_with_gemini_rest(
            prompt=prompt,
            api_key=api_key_or_client,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        model = api_key_or_client

    import google.generativeai as genai

    # Build generation config safely
    gen_cfg = None
    if GenerationConfig is not None:
        try:
            gen_cfg = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        except Exception:
            gen_cfg = None

    # If caller provided a pre-built client object, use it directly.
    if model is not None:
        try:
            if gen_cfg is not None:
                response = model.generate_content(prompt, generation_config=gen_cfg)
            else:
                response = model.generate_content(prompt)

            if not response or not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")

    # String API key path: use REST directly to avoid SDK-local model registry
    # mismatches like "Unknown model: gemini-1.5-flash".
    if isinstance(api_key_or_client, str):
        return _generate_with_gemini_rest(
            prompt=prompt,
            api_key=api_key_or_client,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # SDK fallback for pre-built client objects.
    global _RESOLVED_MODEL_NAME
    candidates = _candidate_model_names(_RESOLVED_MODEL_NAME or model_name)
    last_error: Optional[Exception] = None

    for candidate in candidates:
        try:
            candidate_model = genai.GenerativeModel(candidate)
            if gen_cfg is not None:
                response = candidate_model.generate_content(prompt, generation_config=gen_cfg)
            else:
                response = candidate_model.generate_content(prompt)

            if not response or not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            _RESOLVED_MODEL_NAME = candidate
            return response.text
        except Exception as e:
            last_error = e
            if _is_model_not_found_error(e):
                continue
            raise RuntimeError(f"Gemini API call failed: {e}")

    raise RuntimeError(
        "Gemini API call failed: no compatible model ID worked. "
        f"Last error: {last_error}"
    )


def generate_gemini_response(prompt: str, api_key: str, model: str = "gemini-1.5-flash") -> str:
    """
    Generate a Gemini response using an explicit API key supplied at runtime.
    """
    if not api_key or not api_key.strip():
        raise RuntimeError("Gemini API key is missing.")
    if not model or not model.strip():
        raise RuntimeError("Gemini model is required.")

    return analyze_with_gemini(
        prompt=prompt,
        api_key_or_client=api_key.strip(),
        model_name=model.strip(),
        processing_mode="fast",
    )


def get_api_key() -> str:
    """Prompt the user interactively for a Gemini API key."""
    print("\nEnter your Google Gemini API key")
    print("Get one at: https://aistudio.google.com/app/apikey")
    return input("API Key: ").strip()
