"""
Wrapper for OpenAI API client setup and inference.
"""

from typing import Tuple


def setup_openai_client(api_key: str = None) -> Tuple[object, str]:
    """
    Initialize and return an OpenAI client instance.

    Returns (client, message). If the openai library is missing or initialization fails,
    client will be None and message will contain the error.
    """
    try:
        import openai

        if not api_key:
            api_key = get_api_key()
        if not api_key:
            return None, "No API key provided"

        client = openai.OpenAI(api_key=api_key)
        return client, "OpenAI client initialized successfully"
    except ImportError:
        return None, "openai package not installed. Run: pip install openai"
    except Exception as e:
        return None, f"Error initializing OpenAI client: {e}"


def analyze_with_openai(api_client, prompt: str) -> str:
    """Send a prompt to a configured OpenAI client and return the model response."""
    if not api_client:
        return "OpenAI Error: no client provided"
    try:
        response = api_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API Error: {e}"


def get_api_key():
    """Prompt for an OpenAI API key (simple interactive helper)."""
    key = input("\nPlease enter your OpenAI API key (or press Enter to cancel): \nAPI Key: ").strip()
    return key if key else None
