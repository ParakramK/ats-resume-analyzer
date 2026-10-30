"""Domain interfaces: the contract every AI provider must satisfy.

Business logic (LangChain orchestration, analyzer service, UI) depends on this
Protocol only — never on concrete provider classes or on HTTPX.
"""

from __future__ import annotations

from typing import Protocol


class AIProvider(Protocol):
    """A single-turn text generation endpoint.

    Providers implementing this Protocol are interchangeable (LSP): same inputs
    yield a plain-text completion; transport or response failures are normalized
    into :class:`AtsAnalyzerError` subclasses with user-safe messages.
    """

    name: str
    """Human-readable provider name, e.g. ``"OpenAI"``. Never contains secrets."""

    def generate(self, system_prompt: str, prompt: str) -> str:
        """Return one assistant completion for the given messages.

        Args:
            system_prompt: Instructions for the assistant.
            prompt: The user-side task content. Resume/job text is passed here
                by the orchestration layer together with its own instructions.

        Raises:
            ConfigError: Provider configuration incomplete.
            HTTPClientError: Transport failure (normalized, user-safe).
            ProviderError: Response unusable (blocked, empty, malformed).
        """
        ...  # pragma: no cover

    def describe(self) -> str:
        """Return a short non-secret description like ``OpenAI · gpt-4o-mini``."""
        ...  # pragma: no cover
