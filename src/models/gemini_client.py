"""
Wrapper for Google Gemini API client setup and inference.
"""

from typing import Tuple


def setup_gemini_client(api_key: str = None) -> Tuple[object, str]:
    """
    Initialize and return a Gemini API client.

    Returns (client, message). If the client library is missing or initialization fails,
    client will be None.
    """
    try:
        import google.generativeai as genai

        if not api_key:
            api_key = get_api_key()
        if not api_key:
            return None, "No API key provided"

        genai.configure(api_key=api_key)
        client = genai.GenerativeModel("gemini-1.5-flash")
        return client, "Gemini client initialized successfully"
    except ImportError:
        return None, "google-generativeai package not installed. Run: pip install google-generativeai"
    except Exception as e:
        return None, f"Error initializing Gemini client: {e}"


def analyze_with_gemini(api_client, prompt: str) -> str:
    """Send a prompt to a configured Gemini client and return text result."""
    if not api_client:
        return "Gemini Error: no client provided"
    try:
        response = api_client.generate_content(
            prompt,
            generation_config={"temperature": 0.2},
        )
        return response.text
    except Exception as e:
        return f"Gemini API Error: {e}"


def get_api_key():
    """Prompt for a Gemini API key from the user."""
    key = input("\nPlease enter your Google Gemini API key (or press Enter to cancel): \nAPI Key: ").strip()
    return key if key else None
