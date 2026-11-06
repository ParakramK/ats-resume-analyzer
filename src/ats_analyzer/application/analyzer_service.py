"""Analyzer service: orchestrates local scoring, AI feedback, and connection tests.

AI is invoked only through the injected provider — the caller decides and
must have shown an explicit consent prompt in the UI before calling the AI
functions here. No secrets transit through this module in either direction.
"""

from __future__ import annotations

import json

from ats_analyzer.ai.chains import build_analysis_chain, build_extraction_chain
from ats_analyzer.application.scoring import score_resume
from ats_analyzer.domain.interfaces import AIProvider
from ats_analyzer.domain.models import AiFeedback, AtsResult, JobRequirements, ParsedResume


def run_local_analysis(resume: ParsedResume, requirements: JobRequirements) -> AtsResult:
    """Deterministic score. No network, no credentials."""
    return score_resume(resume, requirements)


def run_ai_analysis(
    provider: AIProvider, resume: ParsedResume, requirements: JobRequirements
) -> tuple[AtsResult, AiFeedback]:
    """Deterministic score first, then AI advice. Requires explicit consent upstream."""
    ats = score_resume(resume, requirements)
    requirements_text = json.dumps(requirements.model_dump(), ensure_ascii=False, indent=1)
    feedback: AiFeedback = build_analysis_chain(provider).invoke(
        {"resume": resume.text, "job_requirements": requirements_text}
    )
    return ats, feedback


def extract_requirements_with_ai(provider: AIProvider, job_description: str) -> JobRequirements:
    """Optional AI extraction of job requirements. Requires explicit consent upstream."""
    result: JobRequirements = build_extraction_chain(provider).invoke(
        {"job_description": job_description}
    )
    return result


def run_connection_test(provider: AIProvider) -> str:
    """Send a tiny probe (no resume content) and report a user-safe verdict."""
    reply = provider.generate(
        system_prompt="You are a connectivity probe. Answer minimally.",
        prompt="Reply with the single word: OK",
    )
    if "ok" in reply.lower():
        return f"Connected to {provider.describe()}. The provider replied 'OK'."
    return (
        f"Connected to {provider.describe()}, but the reply looked unexpected. "
        "The connection works; double-check the model if analysis misbehaves."
    )
