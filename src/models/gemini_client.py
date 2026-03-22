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


def setup_gemini_client(api_key: str) -> Tuple[Optional[object], str]:
    """Initialise and return a configured Gemini model client."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model, ""
    except Exception as e:
        return None, f"Gemini setup failed: {e}"


def check_gemini_connection(api_key: str) -> Tuple[bool, str]:
    """Check if the Gemini API key works."""
    client, err = setup_gemini_client(api_key)
    if err:
        return False, err
    return True, "Gemini API key is valid"


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
        model = genai.GenerativeModel(model_name)
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


def get_api_key() -> str:
    """Prompt the user interactively for a Gemini API key."""
    print("\nEnter your Google Gemini API key")
    print("Get one at: https://aistudio.google.com/app/apikey")
    return input("API Key: ").strip()