"""Domain exceptions.

Every error the application can surface extends :class:`AtsAnalyzerError` and
carries a message that is safe to display: no raw HTTP bodies, no prompt
content, no credentials. Callers that need details for logs must rely on the
``status`` attributes only.
"""

from __future__ import annotations


class AtsAnalyzerError(Exception):
    """Base class for all application errors."""


class UploadError(AtsAnalyzerError):
    """The uploaded file is unusable: wrong type, bad signature, too large, empty."""


class ParserError(AtsAnalyzerError):
    """A document could not be parsed (corrupt, empty, or image-only)."""


class ConfigError(AtsAnalyzerError):
    """Provider configuration is incomplete or invalid (missing key, bad endpoint)."""


class HTTPClientError(AtsAnalyzerError):
    """Normalized transport failure raised by the HTTP client layer."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ProviderError(AtsAnalyzerError):
    """The provider responded in an unusable way (blocked, empty, malformed)."""


class AIOutputError(AtsAnalyzerError):
    """AI output could not be parsed into the expected structured result."""
