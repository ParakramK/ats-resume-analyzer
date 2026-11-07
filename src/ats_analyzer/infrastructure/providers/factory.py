"""Provider factory and configuration validation (single choke point)."""

from __future__ import annotations

from ats_analyzer.config import PROVIDER_LABELS, PROVIDER_REQUIRES_KEY
from ats_analyzer.domain.exceptions import ConfigError
from ats_analyzer.domain.interfaces import AIProvider
from ats_analyzer.domain.models import ProviderConfig
from ats_analyzer.infrastructure.http_client import HttpJsonClient
from ats_analyzer.infrastructure.providers.base import BaseRestProvider
from ats_analyzer.infrastructure.providers.compatible_provider import (
    CompatibleProvider,
    normalize_custom_endpoint,
)
from ats_analyzer.infrastructure.providers.gemini_provider import GeminiProvider
from ats_analyzer.infrastructure.providers.mistral_provider import MistralProvider
from ats_analyzer.infrastructure.providers.openai_provider import OpenAIProvider

_PROVIDERS: dict[str, type[BaseRestProvider]] = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "mistral": MistralProvider,
    "compatible": CompatibleProvider,
}


def validate_provider_config(config: ProviderConfig) -> ProviderConfig:
    """Validate provider configuration; error messages never echo secret values."""
    # Normalize pasted model names: "a, b" from copying suggestion lists, stray
    # whitespace, wrapping quotes — treat the first token as the intended pick.
    config.model = config.model.strip().strip("\"'`").split(",", 1)[0].strip().strip("\"'`")
    if not config.model:
        raise ConfigError(f"A model name is required for {PROVIDER_LABELS[config.provider_id]}.")
    if PROVIDER_REQUIRES_KEY.get(config.provider_id) and not (
        config.api_key and config.api_key.get_secret_value()
    ):
        raise ConfigError(f"An API key is required for {PROVIDER_LABELS[config.provider_id]}.")
    _custom_base_url_providers = ("compatible", "mistral")
    if config.provider_id not in _custom_base_url_providers and config.base_url:
        raise ConfigError("A custom base URL is supported only for OpenAI-compatible endpoints.")
    if config.provider_id == "compatible":
        normalize_custom_endpoint(config.base_url or "")
    elif config.provider_id == "mistral" and config.base_url:
        normalize_custom_endpoint(config.base_url)
    if config.timeout_seconds <= 0:
        raise ConfigError("The request timeout must be positive.")
    return config


def create_provider(config: ProviderConfig, http: HttpJsonClient | None = None) -> AIProvider:
    """Build the provider implementing the AIProvider interface."""
    validate_provider_config(config)
    provider_class = _PROVIDERS[config.provider_id]
    provider: BaseRestProvider = provider_class(config, http=http)
    return provider
