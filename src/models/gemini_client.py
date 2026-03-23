"""
Gemini API client for Style Transfer AI.
Version-safe: works with google-generativeai 0.3.x – 0.8.x+.
"""

from typing import Tuple, Optional

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


def _candidate_model_names(requested_model: str) -> list[str]:
    """Return an ordered list of model IDs to try for broad API compatibility."""
    ordered: list[str] = []

    def add(name: str):
        if name and name not in ordered:
            ordered.append(name)

    # Prefer the requested model first.
    add(requested_model)

    # Common aliases/successors used across Gemini API versions.
    if requested_model == "gemini-1.5-flash":
        add("gemini-1.5-flash-latest")
    if requested_model == "gemini-1.5-pro":
        add("gemini-1.5-pro-latest")

    add("gemini-2.0-flash")
    add("gemini-2.0-flash-exp")
    add("gemini-1.5-flash-latest")
    add("gemini-1.5-pro-latest")

    return ordered


def _is_model_not_found_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "404" in msg
        and ("not found" in msg or "not supported" in msg)
        and "model" in msg
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
    import google.generativeai as genai

    # Resolve api_key_or_client → model object
    if api_key_or_client is None:
        import os
        api_key_or_client = os.environ.get("GEMINI_API_KEY", "")

    if isinstance(api_key_or_client, str):
        if not api_key_or_client:
            raise RuntimeError(
                "Gemini API key is empty. Set GEMINI_API_KEY env var or pass the key explicitly."
            )
        genai.configure(api_key=api_key_or_client)
        model = None
    else:
        model = api_key_or_client

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

    # String API key path: probe candidate model IDs for compatibility.
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


def get_api_key() -> str:
    """Prompt the user interactively for a Gemini API key."""
    print("\nEnter your Google Gemini API key")
    print("Get one at: https://aistudio.google.com/app/apikey")
    return input("API Key: ").strip()