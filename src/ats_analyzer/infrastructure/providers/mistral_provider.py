"""Mistral AI REST provider — OpenAI-dialect chat completions, direct HTTP."""

from __future__ import annotations

from ats_analyzer.domain.exceptions import ConfigError
from ats_analyzer.infrastructure.providers.base import BaseRestProvider, openai_chat_text


class MistralProvider(BaseRestProvider):
    """Mistral chat via ``POST {base}/chat/completions``."""

    name = "Mistral AI"
    default_base_url = "https://api.mistral.ai/v1"

    def generate(self, system_prompt: str, prompt: str) -> str:
        key = self._config.api_key
        if key is None or not key.get_secret_value():
            raise ConfigError("A Mistral API key is required.")
        payload = {
            "model": self._config.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        data = self.post(
            self._url(),
            {
                "Authorization": f"Bearer {key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            payload,
        )
        return openai_chat_text(data)

    def _url(self) -> str:
        base = (self._config.base_url or self.default_base_url).rstrip("/")
        return f"{base}/chat/completions"
