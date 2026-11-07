"""Shared plumbing for REST chat providers: HTTP ownership and config access."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from ats_analyzer.domain.exceptions import ProviderError
from ats_analyzer.domain.models import ProviderConfig
from ats_analyzer.infrastructure.http_client import HttpJsonClient

_logger = logging.getLogger(__name__)


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
        parsed = urlsplit(url)
        # Model/host/path only — never headers or payload (both hold secrets/content).
        _logger.info(
            "rest_post model=%s host=%s path=%s", self._config.model, parsed.hostname, parsed.path
        )
        return self._http.post_json(url, headers=dict(headers), payload=payload)

    # Declared to satisfy the AIProvider protocol contract for all subclasses.
    def generate(self, system_prompt: str, prompt: str) -> str:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name} · {self._config.model}"


def openai_chat_text(data: dict[str, Any]) -> str:
    """Extract the assistant text from an OpenAI-dialect chat completion body.

    Text may be empty: the connection probe treats any delivered completion as
    success, and the chain layer raises its own clearer error when text is
    required for analysis.
    """
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ProviderError("The provider returned no completions.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ProviderError("The provider returned no completions.")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multi-part content in the modern dialect
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    reasoning = message.get("reasoning_content")  # gateways that hide text there
    if isinstance(reasoning, str):
        return reasoning
    return ""
