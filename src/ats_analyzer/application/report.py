"""Report assembly and local exports (PDF/JSON).

Reports contain no credentials and no raw resume text: scores, notes, matched
and missing terms, recommendations and metadata only — the resume itself stays
with the user.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime

import pymupdf

from ats_analyzer import __version__
from ats_analyzer.config import APP_NAME
from ats_analyzer.domain.models import (
    AiFeedback,
    AnalysisMetadata,
    AnalysisReport,
    AtsResult,
    JobRequirements,
    ParsedResume,
)
from ats_analyzer.domain.scoring_rules import (
    METHODOLOGY_LABEL,
    METHODOLOGY_NOTE,
    NOT_A_PREDICTION_NOTE,
)

__all__ = ["build_report", "export_json_bytes", "export_pdf_bytes"]


def build_report(
    resume: ParsedResume,
    requirements: JobRequirements,
    ats: AtsResult,
    local_recommendations: list[str],
    ai_feedback: AiFeedback | None = None,
    ai_provider_label: str = "none",
    ai_model: str = "",
) -> AnalysisReport:
    """Assemble the complete report object used by both exporters."""
    metadata = AnalysisMetadata(
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        scoring_engine=f"{METHODOLOGY_LABEL} ({APP_NAME} v{__version__})",
        ai_provider=ai_provider_label if ai_feedback is not None else "none",
        ai_model=ai_model if ai_feedback is not None else "",
        weights={c.key: c.weight for c in ats.categories},
    )
    return AnalysisReport(
        resume_filename=resume.filename,
        job_title=requirements.title,
        ats=ats,
        ai_feedback=ai_feedback,
        local_recommendations=local_recommendations,
        metadata=metadata,
    )


def export_json_bytes(report: AnalysisReport) -> bytes:
    payload = report.model_dump(mode="json")
    payload["notes"] = [METHODOLOGY_NOTE, NOT_A_PREDICTION_NOTE]
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def export_pdf_bytes(report: AnalysisReport) -> bytes:
    """Render the report as a paginated PDF built locally with PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 64.0
    for line in _pdf_lines(report):
        bold = line.startswith("#")
        text = line.lstrip("#").strip()
        if text == "":
            y += 10
            continue
        size = 13 if line.startswith("## ") else 11 if bold else 9.5
        chunks = textwrap.wrap(text, width=104, break_long_words=False, break_on_hyphens=False)
        for chunk in chunks or [""]:
            if y > 792:
                page = doc.new_page()
                y = 64.0
            page.insert_text((56.0, y), chunk, fontname="hebo" if bold else "helv", fontsize=size)
            y += size + 3.5
        if bold:
            y += 2
    content: bytes = doc.tobytes()
    doc.close()
    return content


def _pdf_lines(report: AnalysisReport) -> list[str]:
    ats = report.ats
    lines = [
        f"# {APP_NAME} — Analysis Report",
        f"# Resume: {report.resume_filename}   Job: {report.job_title or '—'}",
        f"Generated: {report.metadata.generated_at}  ·  Engine: {report.metadata.scoring_engine}",
        f"AI provider: {report.metadata.ai_provider}"
        + (f" ({report.metadata.ai_model})" if report.metadata.ai_model else ""),
        "",
        f"## Overall compatibility: {ats.overall:.1f} / 100",
        f"{NOT_A_PREDICTION_NOTE}",
        "",
        "## Category breakdown",
    ]
    for category in ats.categories:
        weight = "" if not category.applicable else f" · weight {category.weight:.0f}%"
        if category.applicable:
            state = f"score {category.score:.0f}%"
        else:
            state = "excluded (not judgeable)"
        lines.append(f"{category.label}: {state}{weight}")
        lines.extend(f"  - {note}" for note in category.notes)
    lines += [
        "",
        "## Matched / missing",
        f"Matched required skills: {', '.join(ats.matched_required) or '—'}",
        f"Missing required skills: {', '.join(ats.missing_required) or '—'}",
        f"Matched preferred skills: {', '.join(ats.matched_preferred) or '—'}",
        f"Missing preferred skills: {', '.join(ats.missing_preferred) or '—'}",
        f"Missing keywords: {', '.join(ats.missing_keywords) or '—'}",
        "",
        "## Strengths",
        *([f"- {item}" for item in ats.strengths] or ["- —"]),
        "",
        "## Weaknesses",
        *([f"- {item}" for item in ats.weaknesses] or ["- —"]),
        "",
        "## Local recommendations",
        *([f"- {item}" for item in report.local_recommendations] or ["- —"]),
    ]
    if report.ai_feedback is not None:
        feedback = report.ai_feedback
        lines += ["", "## AI feedback (advice, not verified facts)", str(feedback.assessment)]
        if feedback.strengths:
            lines += ["", "AI-noted strengths:", *([f"- {s}" for s in feedback.strengths])]
        if feedback.recommendations:
            lines += ["", "AI recommendations:", *([f"- {r}" for r in feedback.recommendations])]
        if feedback.bullet_suggestions:
            lines += ["", "Bullet rewrites (suggestions):"]
            for bullet in feedback.bullet_suggestions:
                lines.append(f"- was: {bullet.original}")
                lines.append(f"  idea: {bullet.suggested}")
        if feedback.skill_gaps:
            gap_items = [f"- {gap.skill}: {gap.note}" for gap in feedback.skill_gaps]
            lines += ["", "Skill gaps:", *(gap_items)]
    lines += ["", "## Scoring methodology", str(ats.methodology)]
    return lines
