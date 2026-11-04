"""Google Gemini REST provider — direct HTTP against generateContent."""

from __future__ import annotations

from typing import Any

from ats_analyzer.domain.exceptions import ConfigError, ProviderError
from ats_analyzer.infrastructure.providers.base import BaseRestProvider


class GeminiProvider(BaseRestProvider):
    """Gemini via ``POST {base}/models/{model}:generateContent``.

    The API key goes in the ``x-goog-api-key`` header (preferred over a
    query-string parameter, which would put credentials into URLs and logs).
    """

    name = "Google Gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def generate(self, system_prompt: str, prompt: str) -> str:
        key = self._config.api_key
        if key is None or not key.get_secret_value():
            raise ConfigError("A Gemini API key is required.")
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        data = self.post(
            self._url(),
            {"x-goog-api-key": key.get_secret_value(), "Content-Type": "application/json"},
            payload,
        )
        return _gemini_text(data)

    def _url(self) -> str:
        base = (self._config.base_url or self.default_base_url).rstrip("/")
        return f"{base}/models/{self._config.model}:generateContent"


def _gemini_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("error"), dict):
        # Provider-side error text may echo request details; truncate and stay generic.
        kind = str(data["error"].get("status") or data["error"].get("code") or "error")
        raise ProviderError(
            f"The Gemini API returned an error ({kind}). Check the model configuration."
        )
    feedback = data.get("promptFeedback") or {}
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        raise ProviderError("Gemini blocked this prompt with its safety filters.")
    candidates = data.get("candidates")
    if not candidates or not isinstance(candidates[0], dict):
        raise ProviderError("Gemini returned no candidates.")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )
    if not text.strip():
        reason = candidates[0].get("finishReason") or "unknown"
        raise ProviderError(f"Gemini's response was empty (finish reason: {reason}).")
    return text
