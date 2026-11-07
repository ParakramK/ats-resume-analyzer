"""Logging configuration: idempotent setup and metadata-only events."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from ats_analyzer.domain.models import ProviderConfig
from ats_analyzer.infrastructure.http_client import HttpJsonClient
from ats_analyzer.infrastructure.providers.factory import create_provider
from ats_analyzer.logging_setup import configure_logging


@pytest.fixture()
def package_logger():
    """Hand back the package logger with a capture handler, then clean up."""
    configure_logging()
    logger = logging.getLogger("ats_analyzer")
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    capture = Capture()
    logger.addHandler(capture)
    yield logger, records
    logger.handlers.clear()  # drop capture and the persisted handlers alike


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    configure_logging()
    logger = logging.getLogger("ats_analyzer")
    stream_handlers = [h for h in logger.handlers if type(h) is logging.StreamHandler]
    assert stream_handlers, "console handler must be installed"


def test_provider_call_logs_metadata_never_secrets(package_logger) -> None:
    logger, records = package_logger

    key = "sekrit-do-not-log-77"
    body = json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": "Hello secret test"}}]}
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, request=request)

    config = ProviderConfig(provider_id="openai", model="probe-model", api_key=key)
    client = HttpJsonClient(timeout_seconds=2.0)
    client._client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    provider = create_provider(config, http=client)
    try:
        provider.generate("S", "U")
    finally:
        client.close()

    messages = " | ".join(record.getMessage() for record in records)
    assert key not in messages, "the API key must never reach a log line"
    assert any("rest_post" in record.getMessage() for record in records)
    assert any("http_response" in record.getMessage() for record in records)
    # reply length is legitimate metadata; reply content must not appear
    assert "Hello secret test" not in messages
