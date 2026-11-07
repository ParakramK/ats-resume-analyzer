"""Unit tests: local recommendations, analyzer service flow, report exports."""

from __future__ import annotations

import json

import pytest

from ats_analyzer.ai.chains import build_analysis_chain
from ats_analyzer.application.analyzer_service import (
    extract_requirements_with_ai,
    run_ai_analysis,
    run_connection_test,
    run_local_analysis,
)
from ats_analyzer.application.extraction import extract_job_requirements
from ats_analyzer.application.recommendations import local_recommendations
from ats_analyzer.application.report import build_report, export_json_bytes, export_pdf_bytes
from ats_analyzer.domain.exceptions import HTTPClientError
from ats_analyzer.infrastructure.parsers import parse_resume


class FakeProvider:
    name = "Fake"

    def __init__(self, reply: str | Exception) -> None:
        self._reply = reply
        self.calls = 0

    def generate(self, system_prompt: str, prompt: str) -> str:
        self.calls += 1
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply

    def describe(self) -> str:
        return "Fake · canned"


def _pair() -> tuple:
    resume = parse_resume(
        "resume.txt",
        (
            b"Jane Doe\njane@example.com\n555-010-1234\nSUMMARY\n6 years of experience.\n"
            b"EXPERIENCE\n- Built 12 Python services; cut latency 45%.\n"
            b"EDUCATION\nB.S.\nSKILLS\nPython, PostgreSQL, Docker, Kubernetes, REST APIs\n"
        ),
    )
    jd = extract_job_requirements(
        "Backend Engineer\nRequired:\n- 5+ years experience\n- Python\n- Docker\n"
        "Preferred:\n- Kafka\nKeywords: latency, pipelines\n"
    )
    return resume, jd


# --- local flow ---------------------------------------------------------------


def test_local_analysis_needs_no_credentials() -> None:
    resume, jd = _pair()
    ats = run_local_analysis(resume, jd)
    assert 0 <= ats.overall <= 100
    assert "Python" in ats.matched_required


def test_local_recommendations_are_actionable() -> None:
    resume, jd = _pair()
    ats = run_local_analysis(resume, jd)
    tips = local_recommendations(resume, jd, ats)
    assert tips
    assert all(isinstance(tip, str) and tip for tip in tips)
    assert any("truthful" in tip or "If you have it" in tip for tip in tips[:3])


# --- AI flow -------------------------------------------------------------------


def test_ai_analysis_merges_deterministic_score_with_feedback() -> None:
    resume, jd = _pair()
    feedback = {"assessment": "Good fit.", "recommendations": ["Add Kafka detail."]}
    provider = FakeProvider("```json\n" + json.dumps(feedback) + "\n```")
    ats, fb = run_ai_analysis(provider, resume, jd)
    assert ats.overall == run_local_analysis(resume, jd).overall  # score never comes from AI
    assert fb.assessment == "Good fit."
    assert fb.recommendations == ["Add Kafka detail."]
    assert provider.calls == 1


def test_ai_analysis_surfaces_transport_errors() -> None:
    resume, jd = _pair()
    provider = FakeProvider(HTTPClientError("The AI provider rate limit was exceeded (429)."))
    with pytest.raises(HTTPClientError):
        run_ai_analysis(provider, resume, jd)


def test_ai_extraction_flow() -> None:
    provider = FakeProvider(json.dumps({"title": "Data Eng", "required_skills": ["SQL"]}))
    requirements = extract_requirements_with_ai(provider, "Data Eng posting")
    assert requirements.title == "Data Eng"
    assert requirements.required_skills == ["SQL"]


def test_connection_probe_sends_no_resume_content() -> None:
    provider = FakeProvider("OK")
    message = run_connection_test(provider)
    assert "Connected" in message
    assert provider.calls == 1  # exactly one fixed-sentence probe, no document content


def test_connection_probe_reports_unexpected_reply_but_connected() -> None:
    provider = FakeProvider("")  # empty completion still proves connectivity
    message = run_connection_test(provider)
    assert "Connected" in message and "unexpected" in message
    assert provider.calls == 1


def test_chain_buildable_against_fake_provider() -> None:
    provider = FakeProvider(json.dumps({"assessment": "x"}))
    chain = build_analysis_chain(provider)
    feedback = chain.invoke({"resume": "r", "job_requirements": "j"})
    assert feedback.assessment == "x"


# --- exports ---------------------------------------------------------------------


def test_json_export_has_metadata_methodology_and_no_secrets() -> None:
    resume, jd = _pair()
    ats = run_local_analysis(resume, jd)
    report = build_report(resume, jd, ats, local_recommendations(resume, jd, ats))
    data = json.loads(export_json_bytes(report))
    assert data["metadata"]["ai_provider"] == "none"
    assert "overall" in data["ats"]
    assert data["notes"]
    blob = json.dumps(data)
    assert "sk-" not in blob and "Bearer" not in blob


def test_pdf_export_produces_real_pdf() -> None:
    resume, jd = _pair()
    ats = run_local_analysis(resume, jd)
    report = build_report(resume, jd, ats, ["tip"], ai_feedback=None)
    data = export_pdf_bytes(report)
    assert data.startswith(b"%PDF")
