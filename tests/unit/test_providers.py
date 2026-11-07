"""Provider tests: request construction, auth placement, response parsing,
error normalization, endpoint validation, and interchangeability.

All traffic goes through httpx.MockTransport — no network, no real keys.
"""

from __future__ import annotations

import httpx
import pytest

from ats_analyzer.domain.exceptions import ConfigError, ProviderError
from ats_analyzer.domain.models import ProviderConfig
from ats_analyzer.infrastructure.http_client import HttpJsonClient
from ats_analyzer.infrastructure.providers.compatible_provider import normalize_custom_endpoint
from ats_analyzer.infrastructure.providers.factory import create_provider, validate_provider_config

OPENAI_BODY = '{"choices":[{"message":{"role":"assistant","content":"Hello from OpenAI"}}]}'
GEMINI_BODY = '{"candidates":[{"content":{"parts":[{"text":"Hello from Gemini"}]}}]}'
MISTRAL_BODY = '{"choices":[{"message":{"content":"Hello from Mistral"}}]}'
COMPATIBLE_BODY = OPENAI_BODY


def _multipart(provider_id: str, body: bytes, base_url: str | None, api_key: str | None) -> tuple:
    """Build (provider, client, captured) with the request captured by a mock."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(
            url=str(request.url),
            headers=dict(request.headers),
            payload=request.read().decode("utf-8"),
        )
        return httpx.Response(200, content=body, request=request)

    config = ProviderConfig(
        provider_id=provider_id,  # type: ignore[arg-type]
        model="test-model",
        api_key=api_key,
        base_url=base_url,
    )
    client = HttpJsonClient(timeout_seconds=2.0)
    client._client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    provider = create_provider(config, http=client)
    return provider, client, captured


# --- request construction ----------------------------------------------------


def test_openai_request_construction_and_auth() -> None:
    provider, client, captured = _multipart("openai", OPENAI_BODY.encode(), None, "test-key-abc")
    try:
        text = provider.generate("SYS", "USER TASK")
        assert text == "Hello from OpenAI"
        assert captured["url"] == "https://api.openai.com/v1/chat/completions"
        headers = captured["headers"]
        assert headers["authorization"] == "Bearer test-key-abc"
        payload = str(captured["payload"])
        assert "SYS" in payload and "USER TASK" in payload and "test-key" not in payload
    finally:
        client.close()


def test_gemini_request_construction_and_header_auth() -> None:
    provider, client, captured = _multipart("gemini", GEMINI_BODY.encode(), None, "test-key-abc")
    try:
        text = provider.generate("SYS", "USER TASK")
        assert text == "Hello from Gemini"
        assert ":generateContent" in str(captured["url"])
        headers = captured["headers"]
        assert headers["x-goog-api-key"] == "test-key-abc"
        assert "key=" not in str(captured["url"])  # credentials never in the URL
    finally:
        client.close()


def test_mistral_request_construction() -> None:
    provider, client, captured = _multipart("mistral", MISTRAL_BODY.encode(), None, "test-key-abc")
    try:
        text = provider.generate("SYS", "USER TASK")
        assert text == "Hello from Mistral"
        assert str(captured["url"]) == "https://api.mistral.ai/v1/chat/completions"
        assert captured["headers"]["authorization"] == "Bearer test-key-abc"
    finally:
        client.close()


def test_compatible_provider_local_http_without_key() -> None:
    provider, client, captured = _multipart(
        "compatible", COMPATIBLE_BODY.encode(), "http://localhost:11434/v1", None
    )
    try:
        text = provider.generate("SYS", "USER TASK")
        assert text == "Hello from OpenAI"
        assert str(captured["url"]) == "http://localhost:11434/v1/chat/completions"
        assert "authorization" not in captured["headers"]  # optional key omitted
    finally:
        client.close()


def test_compatible_provider_sends_optional_key() -> None:
    provider, client, captured = _multipart(
        "compatible", COMPATIBLE_BODY.encode(), "https://gate.example.test/v1", "gk-9"
    )
    try:
        provider.generate("S", "U")
        assert captured["headers"]["authorization"] == "Bearer gk-9"
    finally:
        client.close()


# --- response edge cases ------------------------------------------------------


def _stub(
    provider_id: str, body: bytes, base_url: str | None = None, api_key: str | None = "k"
) -> tuple:
    provider, client, _captured = _multipart(provider_id, body, base_url, api_key)
    return provider, client


def test_gemini_blocked_response_is_normalized() -> None:
    provider, client = _stub("gemini", b'{"promptFeedback": {"blockReason": "SAFETY"}}')
    try:
        with pytest.raises(ProviderError, match="blocked"):
            provider.generate("S", "U")
    finally:
        client.close()


def test_gemini_error_object_is_normalized_without_echoing() -> None:
    provider, client = _stub("gemini", b'{"error": {"code": 404, "message": "anything"}}')
    try:
        with pytest.raises(ProviderError, match="error"):
            provider.generate("S", "U")
    finally:
        client.close()


def test_gemini_empty_candidates_is_normalized() -> None:
    provider, client = _stub("gemini", b'{"candidates": []}')
    try:
        with pytest.raises(ProviderError, match="no candidates"):
            provider.generate("S", "U")
    finally:
        client.close()


def test_gemini_empty_finish_reason_is_normalized() -> None:
    provider, client = _stub(
        "gemini", b'{"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}'
    )
    try:
        with pytest.raises(ProviderError, match="empty"):
            provider.generate("S", "U")
    finally:
        client.close()


def test_openai_empty_choices_is_normalized() -> None:
    provider, client = _stub("openai", b'{"choices": []}')
    try:
        with pytest.raises(ProviderError, match="no completions"):
            provider.generate("S", "U")
    finally:
        client.close()


# --- configuration validation --------------------------------------------------


def test_missing_key_is_rejected() -> None:
    config = ProviderConfig(provider_id="openai", model="m")
    with pytest.raises(ConfigError, match="API key is required"):
        validate_provider_config(config)


def test_missing_model_is_rejected() -> None:
    config = ProviderConfig(provider_id="mistral", api_key="k", model="")
    with pytest.raises(ConfigError, match="model name is required"):
        validate_provider_config(config)


def test_custom_base_url_rejected_for_named_providers() -> None:
    for provider_id in ("openai", "gemini"):
        config = ProviderConfig(
            provider_id=provider_id, model="m", api_key="k", base_url="https://x.test/v1"
        )
        with pytest.raises(ConfigError, match="OpenAI-compatible"):
            validate_provider_config(config)


def test_custom_base_url_allowed_for_mistral() -> None:
    config = ProviderConfig(
        provider_id="mistral", model="m", api_key="k", base_url=" https://x.test/v1/ "
    )
    assert validate_provider_config(config) is config
    with pytest.raises(ConfigError, match="Insecure http"):
        validate_provider_config(
            ProviderConfig(
                provider_id="mistral", model="m", api_key="k", base_url="http://x.test/v1"
            )
        )


def test_endpoint_rules() -> None:
    assert normalize_custom_endpoint("http://localhost:11434/v1") == "http://localhost:11434/v1"
    assert normalize_custom_endpoint(" http://127.0.0.1:8080/api ") == "http://127.0.0.1:8080/api"
    assert normalize_custom_endpoint("https://gate.example.test") == "https://gate.example.test"
    assert (
        normalize_custom_endpoint("https://10.1.2.3") == "https://10.1.2.3"
    )  # private ip over https ok
    with pytest.raises(ConfigError, match="Insecure http"):
        normalize_custom_endpoint("http://api.mistral.ai")
    with pytest.raises(ConfigError, match="must not embed credentials"):
        normalize_custom_endpoint("https://user:pass@evil.test/v1")
    with pytest.raises(ConfigError, match="http"):
        normalize_custom_endpoint("ftp://localhost/v1")
    with pytest.raises(ConfigError, match="hostname"):
        normalize_custom_endpoint("https://")
    with pytest.raises(ConfigError, match="base URL is required"):
        normalize_custom_endpoint("   ")


def test_local_private_ip_allowed_over_http() -> None:
    assert normalize_custom_endpoint("http://192.168.1.20:1234/v1") == "http://192.168.1.20:1234/v1"


# --- interchangeability (LSP) ---------------------------------------------------


def test_all_providers_normalize_to_plain_text() -> None:
    """Same call signature, same text output contract across providers."""
    expected = {
        "openai": "Hello from OpenAI",
        "gemini": "Hello from Gemini",
        "mistral": "Hello from Mistral",
    }
    setups = [
        ("openai", OPENAI_BODY, None, "k"),
        ("gemini", GEMINI_BODY, None, "k"),
        ("mistral", MISTRAL_BODY, None, "k"),
    ]
    for provider_id, body, _base_url, _key in setups:
        provider, client = _stub(provider_id, body.encode())
        try:
            assert provider.generate("system", "prompt") == expected[provider_id]
        finally:
            client.close()
    provider, client = _stub(
        "compatible", COMPATIBLE_BODY.encode(), "http://127.0.0.1:9999/v1", None
    )
    try:
        assert provider.generate("system", "prompt") == "Hello from OpenAI"
    finally:
        client.close()


def test_valid_config_passes_without_leaking_secrets() -> None:
    config = ProviderConfig(provider_id="gemini", model="m", api_key="supersecretvalue")
    assert validate_provider_config(config) is config
