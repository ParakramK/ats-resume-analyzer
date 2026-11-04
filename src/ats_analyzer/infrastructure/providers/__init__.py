"""AI provider implementations talking to provider REST APIs over HTTPX.

Each provider implements :class:`ats_analyzer.domain.interfaces.AIProvider`.
Provider-specific serialization lives in the concrete classes; everything
shared (HTTP plumbing, config access, description) lives in ``base``.
The factory is the single place that understands provider configuration.
"""

from ats_analyzer.infrastructure.providers.base import BaseRestProvider, openai_chat_text
from ats_analyzer.infrastructure.providers.compatible_provider import CompatibleProvider
from ats_analyzer.infrastructure.providers.factory import create_provider, validate_provider_config
from ats_analyzer.infrastructure.providers.gemini_provider import GeminiProvider
from ats_analyzer.infrastructure.providers.mistral_provider import MistralProvider
from ats_analyzer.infrastructure.providers.openai_provider import OpenAIProvider

__all__ = [
    "BaseRestProvider",
    "CompatibleProvider",
    "GeminiProvider",
    "MistralProvider",
    "OpenAIProvider",
    "create_provider",
    "openai_chat_text",
    "validate_provider_config",
]
