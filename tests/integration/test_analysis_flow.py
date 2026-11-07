"""End-to-end flow tests: bytes in, report out. No credentials, no network."""

from __future__ import annotations

import json

from ats_analyzer.application.analyzer_service import run_ai_analysis, run_local_analysis
from ats_analyzer.application.extraction import extract_job_requirements
from ats_analyzer.application.recommendations import local_recommendations
from ats_analyzer.application.report import build_report, export_json_bytes, export_pdf_bytes
from ats_analyzer.domain.models import AiFeedback
from ats_analyzer.infrastructure.parsers import parse_resume


def test_local_flow_bytes_to_report(sample_resume_text: str, sample_jd_text: str) -> None:
    resume = parse_resume("resume.txt", sample_resume_text.encode("utf-8"))
    requirements = extract_job_requirements(sample_jd_text)
    assert requirements.required_skills, "deterministic extraction must find skills"

    ats = run_local_analysis(resume, requirements)
    recs = local_recommendations(resume, requirements, ats)
    report = build_report(resume, requirements, ats, recs)

    assert report.metadata.ai_provider == "none"
    assert report.metadata.weights
    for category in ats.categories:
        assert 0.0 <= category.score <= 100.0
        if not category.applicable:
            assert category.contribution == 0.0

    as_json = json.loads(export_json_bytes(report))
    assert as_json["job_title"] == requirements.title
    assert as_json["notes"], "exporter must attach the methodology notes"

    pdf = export_pdf_bytes(report)
    assert pdf.startswith(b"%PDF") and len(pdf) > 1_000


def test_ai_flow_scores_deterministically_and_adds_feedback(
    sample_resume_text: str, sample_jd_text: str
) -> None:
    class RecordingProvider:
        name = "Recording"

        def __init__(self) -> None:
            self.seen: list[str] = []

        def generate(self, system_prompt: str, prompt: str) -> str:
            self.seen.append(prompt)
            reply = json.dumps(
                {
                    "assessment": "Good fit.",
                    "strengths": ["Python"],
                    "recommendations": ["Add metrics."],
                }
            )
            return "```json\n" + reply + "```"

    resume = parse_resume("resume.txt", sample_resume_text.encode("utf-8"))
    requirements = extract_job_requirements(sample_jd_text)

    provider = RecordingProvider()
    ats, feedback = run_ai_analysis(provider, resume, requirements)
    assert isinstance(feedback, AiFeedback)
    assert feedback.assessment == "Good fit."

    # the resume text reached the provider; the score equals the pure local one
    assert provider.seen and resume.text in provider.seen[0]
    reference = run_local_analysis(resume, requirements)
    for live, base in zip(ats.categories, reference.categories, strict=True):
        assert live.score == base.score
