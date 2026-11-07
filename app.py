"""ATS Resume Analyzer — Streamlit application entry point.

Flow: upload resume -> paste job description -> review extracted data ->
run local (deterministic) and/or explicit AI-enhanced analysis -> results.

Privacy boundaries enforced here:
- Uploads are parsed in memory; nothing is written to disk.
- API keys stay in session memory and are never rendered in plain text.
- Cloud AI calls happen ONLY when the user clicks the AI action, which is
  labeled with an explicit data-transmission disclosure.
"""

from __future__ import annotations

import logging

import streamlit as st

from ats_analyzer.application.analyzer_service import (
    extract_requirements_with_ai,
    run_ai_analysis,
    run_local_analysis,
)
from ats_analyzer.application.extraction import extract_job_requirements
from ats_analyzer.application.recommendations import local_recommendations
from ats_analyzer.application.report import build_report
from ats_analyzer.config import AI_DISCLOSURE, APP_NAME, LOGO_EMOJI, MAX_UPLOAD_BYTES
from ats_analyzer.domain.exceptions import AtsAnalyzerError
from ats_analyzer.domain.models import JobRequirements, ParsedResume, ProviderConfig
from ats_analyzer.infrastructure.http_client import HttpJsonClient
from ats_analyzer.infrastructure.parsers import parse_resume
from ats_analyzer.infrastructure.providers.factory import create_provider
from ats_analyzer.logging_setup import configure_logging
from ats_analyzer.presentation.dashboard import render_report
from ats_analyzer.presentation.sidebar import render_sidebar

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title=APP_NAME, page_icon=LOGO_EMOJI, layout="wide")
st.title(f"{LOGO_EMOJI} {APP_NAME}")
st.caption("Privacy-first, bring-your-own-key. Local scoring runs entirely offline; AI is opt-in.")
logger.info("app_start")

sidebar_state = render_sidebar()


def _review_seed(version: int, requirements: JobRequirements) -> dict[str, object]:
    """Form widget values showing the current requirements (joined back to lines)."""
    return {
        f"review_title_{version}": requirements.title,
        f"review_required_{version}": "\n".join(requirements.required_skills),
        f"review_preferred_{version}": "\n".join(requirements.preferred_skills),
        f"review_keywords_{version}": "\n".join(requirements.keywords),
        f"review_education_{version}": "\n".join(requirements.education),
        f"review_certifications_{version}": "\n".join(requirements.certifications),
        f"review_years_{version}": int(requirements.min_years_experience or 0),
    }


def _resume() -> ParsedResume | None:
    resume = st.session_state.get("resume")
    return resume if isinstance(resume, ParsedResume) else None


def _requirements() -> JobRequirements | None:
    requirements = st.session_state.get("requirements")
    return requirements if isinstance(requirements, JobRequirements) else None


def _ai_provider(config: ProviderConfig | None):
    """Create a provider (with its own HTTP client) from gadget state."""
    assert config is not None
    client = HttpJsonClient(timeout_seconds=config.timeout_seconds)
    return create_provider(config, http=client), client


def _save_results(
    ats,
    recs: list[str],
    feedback=None,
    label: str = "none",
    model: str = "",
) -> None:
    """Store one analysis outcome in session state (AI fields empty for local runs)."""
    st.session_state["ats"] = ats
    st.session_state["local_recs"] = recs
    st.session_state["ai_feedback"] = feedback
    st.session_state["ai_label"] = label
    st.session_state["ai_model"] = model


tab_upload, tab_jd, tab_review, tab_analyze, tab_results = st.tabs(
    ["1 · Resume", "2 · Job description", "3 · Review", "4 · Analyze", "5 · Results"]
)

# --- Step 1: resume -------------------------------------------------------------

with tab_upload:
    upload = st.file_uploader(
        "Resume document (PDF, DOCX or TXT · max 15 MB)",
        type=("pdf", "docx", "txt"),
        key="resume_uploader",
    )
    if upload is None:
        st.info("Upload a resume to start. The file is parsed in memory only — never stored.")
    else:
        data = upload.getvalue()
        if len(data) > MAX_UPLOAD_BYTES:
            st.error("The file exceeds the 15 MB limit.")
        else:
            try:
                resume = parse_resume(upload.name, data)
                st.session_state["resume"] = resume
                logger.info(
                    "upload type=%s size=%d pages=%d words=%d",
                    resume.file_format,
                    len(data),
                    resume.pages,
                    resume.word_count,
                )
                contacts = ", ".join([*resume.emails, *resume.phones[:1]]) or "no contacts detected"
                st.success(
                    f"Parsed {resume.word_count} words · {resume.pages} page(s) · {contacts}"
                )
                skill_chips = " · ".join(resume.skills[:18])
                if skill_chips:
                    st.caption(f"Detected skills: {skill_chips}")
                with st.expander("Review the extracted resume text", expanded=False):
                    st.text_area("Extracted text", value=resume.text, height=320, disabled=True)
            except AtsAnalyzerError as exc:
                st.session_state.pop("resume", None)
                st.error(str(exc))

# --- Step 2: job description -----------------------------------------------------

with tab_jd:
    jd_text = st.text_area(
        "Paste the full job description",
        height=380,
        key="jd_text",
        placeholder="Paste the posting here…",
    )
    if jd_text.strip():
        requirements = extract_job_requirements(jd_text)
        st.session_state["requirements"] = requirements
        logger.info(
            "jd_extracted required=%d preferred=%d keywords=%d",
            len(requirements.required_skills),
            len(requirements.preferred_skills),
            len(requirements.keywords),
        )
        st.success("Local extraction ready — review and edit it in the next tab.")
    elif isinstance(st.session_state.get("requirements"), JobRequirements):
        st.session_state.pop("requirements", None)

    if st.button("🤖 Improve extraction with AI", disabled=sidebar_state.is_local()):
        st.warning(AI_DISCLOSURE)
        try:
            config = sidebar_state.build_config()
            assert isinstance(config, ProviderConfig)
            provider, client = _ai_provider(config)
            with client, st.spinner(f"Asking {provider.describe()} to extract requirements…"):
                ai_requirements = extract_requirements_with_ai(provider, jd_text)
            logger.info(
                "ai_extract provider=%s model=%s title=%r required=%d",
                config.provider_id,
                config.model,
                ai_requirements.title[:60],
                len(ai_requirements.required_skills),
            )
            st.session_state["requirements"] = ai_requirements
            bumped = st.session_state.get("requirements_version", 0) + 1
            st.session_state["requirements_version"] = bumped
            st.success("AI extraction applied — review it in the Review tab.")
        except AtsAnalyzerError as exc:
            st.error(str(exc))
    if sidebar_state.is_local():
        st.caption("AI extraction needs a provider and API key; deterministic extraction is free.")

# --- Step 3: review --------------------------------------------------------------

with tab_review:
    resume = _resume()
    requirements = _requirements()
    if resume is None:
        st.info("Upload a resume first (tab 1).")
    if requirements is None:
        st.info("Paste a job description first (tab 2).")
    if requirements is not None:
        version = st.session_state.get("requirements_version", 0)
        if f"review_title_{version}" not in st.session_state:
            st.session_state.update(_review_seed(version, requirements))
        with st.form("requirements_form", border=True):
            st.text_input("Job title", key=f"review_title_{version}")
            st.text_area(
                "Required skills (one per line)", key=f"review_required_{version}", height=110
            )
            st.text_area(
                "Preferred skills (one per line)", key=f"review_preferred_{version}", height=90
            )
            st.text_area("Keywords (one per line)", key=f"review_keywords_{version}", height=110)
            st.text_area("Education (one per line)", key=f"review_education_{version}", height=70)
            st.text_area(
                "Certifications (one per line)", key=f"review_certifications_{version}", height=70
            )
            st.number_input(
                "Minimum years of experience",
                min_value=0,
                max_value=30,
                step=1,
                key=f"review_years_{version}",
            )
            if st.form_submit_button("Apply edits", type="primary"):

                def _lines(field: str) -> list[str]:
                    raw = st.session_state.get(f"review_{field}_{version}", "")
                    return [line.strip() for line in str(raw).split("\n") if line.strip()]

                minimum_years = int(st.session_state.get(f"review_years_{version}") or 0)
                requirements = JobRequirements(
                    title=str(st.session_state.get(f"review_title_{version}") or ""),
                    required_skills=_lines("required"),
                    preferred_skills=_lines("preferred"),
                    keywords=_lines("keywords"),
                    education=_lines("education"),
                    certifications=_lines("certifications"),
                    responsibilities=requirements.responsibilities,
                    min_years_experience=minimum_years or None,
                )
                st.session_state["requirements"] = requirements
                st.session_state["requirements_version"] = version + 1
                st.success("Edits applied — they now drive the scoring engine.")

# --- Step 4: analyze --------------------------------------------------------------

with tab_analyze:
    resume = _resume()
    requirements = _requirements()
    if resume is None or requirements is None:
        st.info("Complete steps 1–3 to enable analysis.")
    else:
        st.markdown(
            "**Local ATS analysis** — deterministic scoring, keyword matching and rule-based "
            "recommendations. No API key required; nothing leaves this machine."
        )
        if st.button("🧮 Run local ATS analysis", type="primary", use_container_width=True):
            with st.spinner("Scoring…"):
                local_ats = run_local_analysis(resume, requirements)
                recs = local_recommendations(resume, requirements, local_ats)
                _save_results(local_ats, recs)
            logger.info("local_analysis overall=%.1f", local_ats.overall)
            st.success("Done — open the Results tab.")

        st.divider()
        st.markdown("**AI-enhanced analysis** (optional)")
        if sidebar_state.is_local():
            st.caption("Select an AI provider in the sidebar to enable this.")
        else:
            st.warning(
                AI_DISCLOSURE
                + " Your API key is attached to the request from this machine. Storage and "
                "logging on the provider side are outside this app's control."
            )
            if st.button(
                f"⚡ Send to {sidebar_state.provider_id} and run AI feedback",
                type="secondary",
                use_container_width=True,
            ):
                try:
                    config = sidebar_state.build_config()
                    assert isinstance(config, ProviderConfig)
                    provider, client = _ai_provider(config)
                    with client, st.spinner(f"Analyzing with {provider.describe()}…"):
                        ai_ats, ai_feedback = run_ai_analysis(provider, resume, requirements)
                    recs = local_recommendations(resume, requirements, ai_ats)
                    _save_results(ai_ats, recs, ai_feedback, provider.describe(), config.model)
                    logger.info(
                        "ai_analysis provider=%s model=%s overall=%.1f",
                        config.provider_id,
                        config.model,
                        ai_ats.overall,
                    )
                except AtsAnalyzerError as exc:
                    st.error(str(exc))
                else:
                    st.success("Done — open the Results tab.")

# --- Step 5: results ---------------------------------------------------------------

with tab_results:
    resume = _resume()
    requirements = _requirements()
    ats = st.session_state.get("ats")
    if ats is None or resume is None or requirements is None:
        st.info("Run an analysis (tab 4) to see results here.")
    else:
        feedback = st.session_state.get("ai_feedback")
        report = build_report(
            resume=resume,
            requirements=requirements,
            ats=ats,
            local_recommendations=st.session_state.get("local_recs", []),
            ai_feedback=feedback,
            ai_provider_label=st.session_state.get("ai_label", "none"),
            ai_model=st.session_state.get("ai_model", ""),
        )
        render_report(report)
        if feedback is None:
            st.caption("Tip: run AI-enhanced analysis (tab 4) for bullet rewrites and gap notes.")
