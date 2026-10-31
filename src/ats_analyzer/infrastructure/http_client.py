"""Reusable JSON-over-HTTP client for AI provider APIs, built on HTTPX.

Security and reliability properties:

- Redirects are never followed, so authorization headers cannot leak to a
  redirect target (providers are expected to answer directly; a redirect is
  treated as a configuration anomaly).
- Responses are streamed and rejected if they exceed a bounded size, so a
  misbehaving endpoint cannot exhaust memory.
- Only transient failures are retried with bounded attempts: HTTP 429/502/503/504,
  network and timeout errors. Authentication, invalid request, redirect, and
  endpoint-configuration errors fail immediately.
- POST retries are acceptable here because every request is a stateless
  chat completion; attempts are strictly bounded (see MAX_RETRIES) and honor
  Retry-After when provided.
- Log lines contain only method, host, status and attempt numbers — never
  headers, providers' response bodies, or any request payload.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx

from ats_analyzer.config import DEFAULT_TIMEOUT_SECONDS, MAX_RESPONSE_BYTES, MAX_RETRIES
from ats_analyzer.domain.exceptions import HTTPClientError

_logger = logging.getLogger(__name__)

_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_CAP_SECONDS = 8.0
_RETRY_AFTER_CAP_SECONDS = 30.0
_JITTER_CEILING_SECONDS = 0.3


_STATUS_MESSAGES: dict[int, str] = {
    400: "The AI provider rejected the request as invalid (400). Check the selected model name.",
    401: "The AI provider rejected the credentials (401). Check the API key and permissions.",
    403: "The AI provider denied access (403). Check the API key and permissions.",
    404: "The AI provider endpoint or model was not found (404). Check model name and endpoint.",
    405: "The AI provider rejected the request method (405). The endpoint config looks wrong.",
    408: "The AI provider request timed out server-side (408). Retry shortly.",
    413: "The request was too large for the AI provider (413). Shorten the documents.",
    422: "The AI provider rejected the request as invalid (422). Check the selected model name.",
    429: "The AI provider rate limit was exceeded (429). Wait a moment or pick a smaller model.",
}


def _status_message(status: int) -> str:
    """Map status codes to user-safe messages. Never includes response bodies."""
    message = _STATUS_MESSAGES.get(status)
    if message is not None:
        return message
    if 500 <= status <= 599:
        return f"The AI provider had an internal error (HTTP {status}). Retry shortly."
    return f"The AI provider returned an unexpected HTTP status {status}."


class HttpJsonClient:
    """Synchronous POST-JSON client with bounded retries and size limits.

    One instance wraps one reusable ``httpx.Client``; use it as a context
    manager (or call :meth:`close`) so connections are cleaned up reliably.
    ``transport`` exists for tests (``httpx.MockTransport``) only.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        max_retries: int = MAX_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._max_response_bytes = max_response_bytes
        self._max_retries = max_retries
        self._sleep = sleep
        self._client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,  # never send authorization headers to redirect targets
            transport=transport,
        )

    def __enter__(self) -> HttpJsonClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def post_json(
        self, url: str, *, headers: Mapping[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST ``payload`` and return the parsed JSON object response."""
        host = urlsplit(url).hostname or "unknown"
        for attempt in range(self._max_retries + 1):
            try:
                with self._client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code in _RETRYABLE_STATUSES and attempt < self._max_retries:
                        _logger.info(
                            "http_retryable host=%s status=%s attempt=%d",
                            host,
                            response.status_code,
                            attempt + 1,
                        )
                        self._sleep(self._backoff(attempt, response.headers.get("retry-after")))
                        continue
                    return self._consume(host, response)
            except httpx.TimeoutException:
                message = (
                    "The request to the AI provider timed out. Try again or pick a smaller model."
                )
                if attempt >= self._max_retries:
                    raise HTTPClientError(message) from None
                self._sleep(self._backoff(attempt, None))
            except httpx.RequestError:
                if attempt >= self._max_retries:
                    raise HTTPClientError(
                        "The connection to the AI provider failed. Check network connectivity."
                    ) from None
                self._sleep(self._backoff(attempt, None))
        raise HTTPClientError("The connection to the AI provider failed.")  # pragma: no cover

    def _consume(self, host: str, response: httpx.Response) -> dict[str, Any]:
        status = response.status_code
        _logger.info("http_request host=%s status=%s", host, status)
        if status in _RETRYABLE_STATUSES:  # attempts exhausted, decided by caller
            raise HTTPClientError(_status_message(status), status=status)
        if 300 <= status < 400:
            raise HTTPClientError(
                "The AI provider attempted a redirect; redirects were not followed for security.",
                status=status,
            )
        if status >= 400:
            raise HTTPClientError(_status_message(status), status=status)

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > self._max_response_bytes:
                raise HTTPClientError("The AI provider response was too large and was rejected.")
            chunks.append(chunk)
        try:
            parsed: Any = json.loads(b"".join(chunks) or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise HTTPClientError(
                "The AI provider returned a malformed response (invalid JSON)."
            ) from None
        if not isinstance(parsed, dict):
            raise HTTPClientError("The AI provider returned an unexpected response format.")
        return parsed

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        delay = _JITTER_CEILING_SECONDS * random.random()  # non-cryptographic jitter
        if retry_after is not None:
            with contextlib.suppress(ValueError):
                delay = (
                    min(_RETRY_AFTER_CAP_SECONDS, float(retry_after))
                    + _JITTER_CEILING_SECONDS * random.random()
                )
        else:
            delay += min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * (2**attempt))
        return delay
