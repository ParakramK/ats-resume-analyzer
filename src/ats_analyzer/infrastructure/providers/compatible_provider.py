"""OpenAI-compatible endpoints (Ollama, self-hosted servers, other gateways).

These endpoints are a trust boundary configured by the user:
- HTTPS is required for remote hosts; plain HTTP is allowed only for
  explicitly local addresses (localhost, loopback, private IP literals).
- Credentials embedded in the URL are rejected — keys belong in the
  API-key field only.
- Only baseline chat completion is assumed; optional OpenAI features are
  never silently sent.

Known limitation (documented in the README): hostname-based addresses are
checked as literals; DNS names resolving to internal addresses are not
resolved here.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from ats_analyzer.domain.exceptions import ConfigError
from ats_analyzer.infrastructure.providers.base import BaseRestProvider, openai_chat_text

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", "host.docker.internal"})


def normalize_custom_endpoint(raw: str) -> str:
    """Validate and normalize a user-supplied OpenAI-compatible base URL."""
    endpoint = (raw or "").strip().rstrip("/")
    if not endpoint:
        raise ConfigError("A base URL is required for the OpenAI-compatible provider.")
    if len(endpoint) > 2048:
        raise ConfigError("The base URL is unexpectedly long; check the configuration.")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in ("http", "https"):
        raise ConfigError("The base URL must start with http:// or https://.")
    if parsed.username or parsed.password:
        raise ConfigError("The base URL must not embed credentials; use the API key field instead.")
    host = parsed.hostname or ""
    if not host:
        raise ConfigError("The base URL has no hostname.")
    if parsed.scheme == "http" and not _is_local(host):
        raise ConfigError(
            "Insecure http:// is allowed only for local endpoints (localhost or a private IP). "
            "Use https:// for remote endpoints."
        )
    return endpoint


def _is_local(host: str) -> bool:
    if host.lower() in _LOCAL_HOSTS or host.lower().endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private


class CompatibleProvider(BaseRestProvider):
    """Baseline ``POST {base_url}/chat/completions`` for compatible servers."""

    name = "OpenAI-compatible"
    default_base_url = ""

    def generate(self, system_prompt: str, prompt: str) -> str:
        endpoint = normalize_custom_endpoint(self._config.base_url or "")
        payload = {
            "model": self._config.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self._config.api_key and self._config.api_key.get_secret_value():
            headers["Authorization"] = f"Bearer {self._config.api_key.get_secret_value()}"
        data = self.post(f"{endpoint}/chat/completions", headers, payload)
        return openai_chat_text(data)
