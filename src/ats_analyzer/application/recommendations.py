"""Deterministic, local recommendations derived from the ATS result + resume.

These complement (never substitute) AI advice and require no credentials.
"""

from __future__ import annotations

from ats_analyzer.application.scoring import (
    _missing_keywords,
    _missing_names,
    _normalized,
    _stated_years,
)
from ats_analyzer.domain.models import AtsResult, JobRequirements, ParsedResume


def local_recommendations(
    resume: ParsedResume, requirements: JobRequirements, ats: AtsResult
) -> list[str]:
    """Rule-based, actionable edits ranked roughly by impact."""
    haystack = _normalized(resume.text)
    tips: list[str] = []

    missing_required = _missing_names(requirements.required_skills, haystack)
    for skill in missing_required[:6]:
        tips.append(
            f"If you have it, add evidence for '{skill}' — "
            "name the project or role where you used it."
        )
    if len(missing_required) > 6:
        remaining = len(missing_required) - 6
        tips.append(f"{remaining} more required skill terms are unlisted; keep additions truthful.")

    missing_keywords = _missing_keywords(haystack, requirements.keywords)
    for keyword in missing_keywords[:6]:
        tips.append(
            f"Where truthful, work '{keyword}' into an achievement bullet, "
            "not only the skills list."
        )

    stated_years = _stated_years(haystack)
    if requirements.min_years_experience and stated_years is None:
        tips.append("State total years of experience explicitly (e.g. '7 years of experience').")
    elif (
        requirements.min_years_experience
        and stated_years is not None
        and stated_years < requirements.min_years_experience
    ):
        tips.append(
            f"The posting asks for {requirements.min_years_experience}+ years; make your "
            f"{stated_years}+ years unmistakable in the summary line."
        )

    if requirements.education:
        tips.append("List the exact degree name under Education; spell out abbreviations once.")
    if requirements.certifications:
        tips.append("Add a Certifications heading and list every certificate exactly as issued.")

    structure = next((c for c in ats.categories if c.key == "structure"), None)
    if structure:
        for note in structure.notes:
            _structure_tip(tips, note)

    if not tips:
        tips.append("No obvious gaps for this posting — keep tailoring the summary to it.")
    return tips[:12]


def _structure_tip(tips: list[str], note: str) -> None:
    if note.startswith("No email or phone"):
        tips.append("Add a plain-text contact block (email and phone) at the very top.")
    elif note.startswith("No summary"):
        tips.append("Open with a three-line professional summary tailored to this posting.")
    elif note.startswith("No bullet points"):
        tips.append("Break the experience sections into one-line achievement bullets.")
    elif note.startswith("Some accomplishment"):
        tips.append("Quantify every bullet you can with real outcomes (%, $, time saved).")
    elif note.startswith("Only ~"):
        tips.append("Add detail: achievements, tools and scope are keywords for matching.")
    elif note.startswith("~") and "long" in note:
        tips.append("Trim older/irrelevant roles; very long resumes often get truncated.")
