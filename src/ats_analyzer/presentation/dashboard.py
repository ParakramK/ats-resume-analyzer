"""Results dashboard: score metrics, category breakdown, matched/missing,
strengths/weaknesses, recommendations, AI feedback, and export buttons."""

from __future__ import annotations

import streamlit as st

from ats_analyzer.application.report import export_json_bytes, export_pdf_bytes
from ats_analyzer.domain.models import AnalysisReport, AtsResult


def render_report(report: AnalysisReport) -> None:
    ats = report.ats
    _overall(ats.overall)
    st.divider()
    left, right = st.columns(2)
    with left:
        _category_chart(ats)
    with right:
        _required_lists(ats)
    _category_details(ats)
    st.divider()
    _keyed_lists("💪 Strengths", ats.strengths, "✅")
    _keyed_lists("⚠️ Weaknesses", ats.weaknesses, "⚠️")
    st.divider()
    st.subheader("🧭 Recommendations (local, rule-based)")
    for tip in report.local_recommendations:
        st.markdown(f"- {tip}")
    if report.ai_feedback is not None:
        _ai_feedback(report)
    st.divider()
    _exports(report)


def _overall(score: float) -> None:
    st.subheader("🎯 Overall ATS compatibility")
    col_metric, col_note = st.columns([1, 2])
    with col_metric:
        st.metric(label="Score", value=f"{score:.1f} / 100")
        st.progress(min(max(score, 0.0), 100.0) / 100.0)
    with col_note:
        st.caption(
            "A weighted keyword-and-structure estimate of resume-to-job fit. "
            "Not a prediction of ATS acceptance or recruiter decisions."
        )


def _category_chart(ats: AtsResult) -> None:
    st.markdown("**Category scores**")
    applicable = {c.label: c.score for c in ats.categories if c.applicable}
    if applicable:
        st.bar_chart(applicable)
    excluded = [c.label for c in ats.categories if not c.applicable]
    if excluded:
        st.caption("Excluded from the overall score (nothing to judge): " + ", ".join(excluded))


def _required_lists(ats: AtsResult) -> None:
    st.markdown("**Required skills**")
    if ats.matched_required:
        st.markdown(" ".join(f"`{skill}` ✅" for skill in ats.matched_required))
    else:
        st.caption("No required skills listed.")
    if ats.missing_required:
        st.markdown(" ".join(f"`{skill}` ❌" for skill in ats.missing_required))
    st.markdown("**Preferred skills**")
    if ats.matched_preferred:
        st.markdown(" ".join(f"`{skill}` ✅" for skill in ats.matched_preferred))
    if ats.missing_preferred:
        st.markdown(" ".join(f"`{skill}` ❌" for skill in ats.missing_preferred))


def _category_details(ats: AtsResult) -> None:
    st.markdown("**Explainable breakdown**")
    for category in ats.categories:
        if category.applicable:
            header = f"{category.label} — {category.score:.0f}%  (weight {category.weight:.0f}%)"
        else:
            header = f"{category.label} — excluded (not judgeable)"
        with st.expander(header):
            if category.applicable:
                st.progress(min(max(category.score, 0.0), 100.0) / 100.0)
            for note in category.notes:
                st.markdown(f"- {note}")


def _keyed_lists(title: str, items: list[str], marker: str) -> None:
    st.subheader(title)
    if items:
        for item in items:
            st.markdown(f"{marker} {item}")
    else:
        st.caption("—")


def _ai_feedback(report: AnalysisReport) -> None:
    feedback = report.ai_feedback
    assert feedback is not None
    st.divider()
    st.subheader(f"🤖 AI feedback ({report.metadata.ai_provider})")
    st.info(
        "AI advice is a suggestion, not verified fact. Nothing the model says "
        "changes your ATS score, which is computed deterministically."
    )
    if feedback.assessment:
        st.markdown(f"> {feedback.assessment}")
    if feedback.strengths:
        st.markdown("**AI-noted strengths**")
        for item in feedback.strengths:
            st.markdown(f"✅ {item}")
    if feedback.recommendations:
        st.markdown("**AI recommendations**")
        for item in feedback.recommendations:
            st.markdown(f"- {item}")
    if feedback.bullet_suggestions:
        st.markdown("**Bullet rewrites (suggestions only)**")
        for bullet in feedback.bullet_suggestions:
            left, right = st.columns(2)
            with left:
                st.caption("Original")
                st.markdown(bullet.original or "—")
            with right:
                st.caption("Suggested")
                st.markdown(bullet.suggested or "—")
    if feedback.skill_gaps:
        st.markdown("**Genuine skill gaps**")
        for gap in feedback.skill_gaps:
            st.markdown(f"- **{gap.skill}** — {gap.note}")


def _exports(report: AnalysisReport) -> None:
    st.subheader("⬇️ Download the report")
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download JSON report",
            data=export_json_bytes(report),
            file_name="ats-analysis-report.json",
            mime="application/json",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "Download PDF report",
            data=export_pdf_bytes(report),
            file_name="ats-analysis-report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    st.caption(
        "Reports contain scores and advice only — never your API key or the full resume text."
    )
