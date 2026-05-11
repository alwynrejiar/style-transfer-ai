"""
OpenAI client for cloud model inference using Chat Completions API.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


def _extract_error_message(payload: Dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "OpenAI API error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    return "OpenAI API error"


def generate_openai_response(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 3000,
) -> str:
    if not api_key or not api_key.strip():
        raise RuntimeError("OpenAI API key is missing.")
    if not model or not model.strip():
        raise RuntimeError("OpenAI model is required.")

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model.strip(),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    try:
        response = requests.post(
            OPENAI_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=body,
            timeout=120,
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("OpenAI request timed out.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"OpenAI network error: {exc}") from exc

    payload: Dict[str, Any] = {}
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.status_code in (401, 403):
        raise RuntimeError("Invalid OpenAI API key.")
    if response.status_code >= 400:
        provider_error = _extract_error_message(payload)
        raise RuntimeError(f"OpenAI request failed ({response.status_code}): {provider_error}")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenAI returned an empty response.")

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
        raise RuntimeError("OpenAI returned an empty message content.")

    return content.strip()


def analyze_with_openai(
    prompt: str,
    api_key: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 3000,
    processing_mode: str = "fast",
) -> str:
    _ = processing_mode
    return generate_openai_response(
        prompt=prompt,
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
