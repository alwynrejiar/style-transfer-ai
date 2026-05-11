"""
OpenRouter client for cloud model inference using the Chat Completions API.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import requests


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


def _extract_error_message(payload: Dict[str, Any]) -> str:
    err = payload.get("error")
    if isinstance(err, dict):
        # OpenRouter may return useful nested provider details on overload/rate-limit.
        details = err.get("metadata") or err.get("details") or err.get("provider_error")
        base = err.get("message") or err.get("code") or "OpenRouter provider error"
        if details:
            return f"{base} ({details})"
        return str(base)
    if isinstance(err, str) and err.strip():
        return err.strip()
    return "OpenRouter provider error"


def generate_openrouter_response(prompt: str, api_key: str, model: str) -> str:
    """
    Generate a response from OpenRouter using an OpenAI-compatible request body.
    """
    if not api_key or not api_key.strip():
        raise RuntimeError("OpenRouter API key is missing.")
    if not model or not model.strip():
        raise RuntimeError("OpenRouter model is required.")

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000/app",
        "X-Title": "Stylomex",
    }
    body = {
        "model": model.strip(),
        "messages": [{"role": "user", "content": prompt}],
    }

    response = None
    payload: Dict[str, Any] = {}
    for attempt in range(3):
        try:
            response = requests.post(
                OPENROUTER_CHAT_COMPLETIONS_URL,
                headers=headers,
                json=body,
                timeout=120,
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("OpenRouter request timed out.") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"OpenRouter network error: {exc}") from exc

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        # Retry only transient capacity/rate-limit errors.
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(1.5 * (attempt + 1))
            continue
        break

    if response is None:
        raise RuntimeError("OpenRouter request failed before receiving a response.")

    if response.status_code in (401, 403):
        raise RuntimeError("Invalid OpenRouter API key.")
    if response.status_code == 429:
        provider_error = _extract_error_message(payload)
        raise RuntimeError(
            "OpenRouter request failed (429): rate limit or provider capacity reached. "
            f"Details: {provider_error}"
        )
    if response.status_code >= 400:
        provider_error = _extract_error_message(payload)
        raise RuntimeError(f"OpenRouter request failed ({response.status_code}): {provider_error}")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenRouter returned an empty response.")

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")

    if isinstance(content, list):
        content = " ".join(
            str(part.get("text", "")).strip()
            for part in content
            if isinstance(part, dict) and part.get("text")
        ).strip()

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenRouter returned an empty message content.")

    return content.strip()
