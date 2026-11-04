"""Shared plumbing for REST chat providers: HTTP ownership and config access."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ats_analyzer.domain.exceptions import ProviderError
from ats_analyzer.domain.models import ProviderConfig
from ats_analyzer.infrastructure.http_client import HttpJsonClient


class BaseRestProvider:
    """Base class implementing the parts every provider shares.

    Each concrete provider owns an ``HttpJsonClient`` unless one is injected
    (tests inject a transport-bearing client; the UI injects a client it can
    close after a run). Subclasses implement :meth:`generate` by building a
    URL, headers, and a provider-specific payload.
    """

    name = "REST provider"
    default_base_url = ""

    def __init__(self, config: ProviderConfig, http: HttpJsonClient | None = None) -> None:
        self._config = config
        self._owned_http: HttpJsonClient | None = None
        self._http: HttpJsonClient
        if http is None:
            self._owned_http = HttpJsonClient(timeout_seconds=config.timeout_seconds)
            self._http = self._owned_http
        else:
            self._http = http

    def close(self) -> None:
        if self._owned_http is not None:
            self._owned_http.close()
            self._owned_http = None

    def __enter__(self) -> BaseRestProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def post(self, url: str, headers: Mapping[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return self._http.post_json(url, headers=dict(headers), payload=payload)

    # Declared to satisfy the AIProvider protocol contract for all subclasses.
    def generate(self, system_prompt: str, prompt: str) -> str:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name} · {self._config.model}"


def openai_chat_text(data: dict[str, Any]) -> str:
    """Extract the assistant text from an OpenAI- dialect chat completion body."""
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ProviderError("The provider returned no completions.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ProviderError("The provider returned an empty completion.")
    return content
