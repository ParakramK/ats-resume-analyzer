"""Tests for the reusable HTTP client: retries, size limits, error mapping.

All HTTP traffic is simulated with httpx.MockTransport — no network, no keys.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from ats_analyzer.domain.exceptions import HTTPClientError
from ats_analyzer.infrastructure.http_client import HttpJsonClient

URL = "https://api.example.test/v1/chat/completions"


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    sleep: Callable[[float], None] = lambda seconds: None,
    **client_kwargs: float,
) -> tuple[HttpJsonClient, list[httpx.Request]]:
    """Build an HttpJsonClient over a tracking MockTransport; return (client, calls)."""
    calls: list[httpx.Request] = []

    def track(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    client = HttpJsonClient(
        sleep=sleep,
        transport=httpx.MockTransport(track),
        timeout_seconds=2.0,
        **client_kwargs,  # type: ignore[arg-type]
    )
    return client, calls


def _response(
    payload: dict[str, str],
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status, json=payload, headers=headers or {}, request=httpx.Request("POST", URL)
    )


def test_post_json_returns_parsed_object() -> None:
    client, calls = _client(lambda request: _response({"ok": True}))
    try:
        result = client.post_json(URL, headers={"Authorization": "Bearer x"}, payload={"a": 1})
    finally:
        client.close()
    assert result == {"ok": True}
    assert len(calls) == 1
    assert calls[0].headers["Authorization"] == "Bearer x"


def test_auth_error_fails_immediately_without_retry() -> None:
    client, calls = _client(lambda request: _response({}, status=401))
    try:
        with pytest.raises(HTTPClientError) as excinfo:
            client.post_json(URL, headers={"Authorization": "Bearer x"}, payload={})
    finally:
        client.close()
    assert "rejected the credentials" in str(excinfo.value)
    assert len(calls) == 1  # authentication failures are never retried


def test_rate_limit_retries_then_succeeds() -> None:
    responses = [_response({}, status=429, headers={"retry-after": "0"}), _response({"ok": 1})]

    client, calls = _client(lambda request: responses.pop(0))
    try:
        assert client.post_json(URL, headers={}, payload={}) == {"ok": 1}
    finally:
        client.close()
    assert len(calls) == 2


def test_server_error_exhausts_bounded_attempts() -> None:
    client, calls = _client(lambda request: _response({}, status=503))
    try:
        with pytest.raises(HTTPClientError) as excinfo:
            client.post_json(URL, headers={}, payload={})
    finally:
        client.close()
    assert len(calls) == client._max_retries + 1  # bounded attempts, no infinite retry
    assert "HTTP 503" in str(excinfo.value)


def test_redirect_is_never_followed() -> None:
    client, calls = _client(lambda request: _response({}, status=301))
    try:
        with pytest.raises(HTTPClientError) as excinfo:
            client.post_json(URL, headers={"Authorization": "secret"}, payload={})
    finally:
        client.close()
    assert len(calls) == 1
    assert "redirect" in str(excinfo.value)


def test_oversized_response_is_rejected() -> None:
    client, calls = _client(lambda request: _response({"body": "x" * 1000}), max_response_bytes=100)
    try:
        with pytest.raises(HTTPClientError, match="too large"):
            client.post_json(URL, headers={}, payload={})
    finally:
        client.close()
    assert len(calls) == 1


def test_malformed_json_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>nope</html>", request=httpx.Request("POST", URL))

    client, calls = _client(handler)
    try:
        with pytest.raises(HTTPClientError, match="malformed"):
            client.post_json(URL, headers={}, payload={})
    finally:
        client.close()
    assert len(calls) == 1


def test_non_object_json_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=json.dumps([1, 2]).encode(), request=httpx.Request("POST", URL)
        )

    client, calls = _client(handler)
    try:
        with pytest.raises(HTTPClientError, match="unexpected response format"):
            client.post_json(URL, headers={}, payload={})
    finally:
        client.close()


def test_connect_error_is_retried_then_normalized() -> None:
    holder = httpx.Request("POST", URL)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=holder)

    client, calls = _client(handler)
    try:
        with pytest.raises(HTTPClientError, match="connection"):
            client.post_json(URL, headers={}, payload={})
    finally:
        client.close()
    assert len(calls) == client._max_retries + 1


def test_retry_after_is_respected_with_cap() -> None:
    slept: list[float] = []
    responses = [_response({}, status=429, headers={"retry-after": "2"}), _response({"ok": 1})]

    client, _calls = _client(lambda request: responses.pop(0), sleep=slept.append)
    try:
        assert client.post_json(URL, headers={}, payload={}) == {"ok": 1}
    finally:
        client.close()
    assert slept and 2.0 <= slept[0] <= 2.4  # Retry-After honored, jitter capped
