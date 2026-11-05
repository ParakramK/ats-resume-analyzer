"""Tests for LangChain orchestration: prompts, adapter, repair, validation.

A FakeProvider supplies canned completions; no network, no keys, no SDKs.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ats_analyzer.ai import prompts
from ats_analyzer.ai.chains import (
    ProviderChatModel,
    build_analysis_chain,
    build_extraction_chain,
)
from ats_analyzer.domain.exceptions import AIOutputError, HTTPClientError
from ats_analyzer.domain.models import AiFeedback, JobRequirements


class FakeProvider:
    """Protocol-conforming provider that replays a canned response."""

    name = "Fake"

    def __init__(self, reply: str | Exception) -> None:
        self._reply = reply
        self.seen: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, prompt: str) -> str:
        self.seen.append((system_prompt, prompt))
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply

    def describe(self) -> str:
        return "Fake · canned"


def test_analysis_chain_composes_and_validates() -> None:
    feedback = {
        "assessment": "Solid match on backend skills.",
        "strengths": ["Python scope", "Data platform experience"],
        "recommendations": ["Mention Kafka work from the ETL migration."],
    }
    provider = FakeProvider("```json\n" + json.dumps(feedback) + "```")  # fenced on purpose
    chain = build_analysis_chain(provider)
    result = chain.invoke(
        {"resume": "RESUME BODY ABOUT PYTHON", "job_requirements": "REQUIRED: PYTHON"}
    )
    assert isinstance(result, AiFeedback)
    assert result.assessment == "Solid match on backend skills."
    # the resume text travels as a template variable into the user prompt
    system, human = provider.seen[0]
    assert "RESUME BODY ABOUT PYTHON" in human
    assert "UNTRUSTED DATA" in system  # injection-defense rule always present
    assert "Never invent" in system  # anti-fabrication rule always present
    assert "JSON" in system  # schema instruction present


def test_extraction_chain_returns_job_requirements() -> None:
    payload = {"title": "Data Eng", "required_skills": ["SQL"], "min_years_experience": 3}
    provider = FakeProvider(json.dumps(payload))
    chain = build_extraction_chain(provider)
    result = chain.invoke(
        {"job_description": "Data Eng posting. Job description BEGIN nested {braces}"}
    )
    assert isinstance(result, JobRequirements)
    assert result.title == "Data Eng"


def test_chat_model_adapter_smoke() -> None:
    provider = FakeProvider("just text")
    model = ProviderChatModel(provider=provider)
    message = model.invoke([HumanMessage(content="hello")])
    assert isinstance(message, AIMessage)
    assert message.content == "just text"


def test_provider_transport_errors_propagate_normalized() -> None:
    provider = FakeProvider(HTTPClientError("The AI provider rate limit was exceeded."))
    model = ProviderChatModel(provider=provider)
    with pytest.raises(HTTPClientError):
        model.invoke([HumanMessage(content="hi")])


# --- output repair & validation ---------------------------------------------


def test_extract_json_object_from_chatty_text() -> None:
    raw = 'Sure! Here you go: {"a": 1, "b": {"c": 2}} hope that helps'
    assert prompts.extract_json_object(raw) == {"a": 1, "b": {"c": 2}}


def test_extract_json_object_rejects_garbage() -> None:
    with pytest.raises(AIOutputError, match="did not contain a JSON object"):
        prompts.extract_json_object("no object here")
    with pytest.raises(AIOutputError, match="empty"):
        prompts.extract_json_object("")


def test_validate_missing_fields_maps_to_user_error() -> None:
    step = prompts.validate_ai_json(AiFeedback)
    with pytest.raises(AIOutputError, match="missing expected fields"):
        step({"bullet_suggestions": [{"original": 42}]})  # wrong type -> ValidationError


def test_chain_flags_unparseable_ai_output() -> None:
    provider = FakeProvider("I would rather freeform this answer, no JSON today.")
    chain = build_analysis_chain(provider)
    with pytest.raises(AIOutputError, match="did not contain a JSON object"):
        chain.invoke({"resume": "R", "job_requirements": "J"})
