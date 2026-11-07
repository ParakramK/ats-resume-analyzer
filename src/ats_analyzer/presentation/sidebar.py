"""Sidebar: provider selection, keys, connection test, privacy, reset.

Credentials entered here live only in Streamlit session memory. Nothing in
this module performs network I/O on render — the connection test runs strictly
on an explicit button click.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import streamlit as st

from ats_analyzer.application.analyzer_service import run_connection_test
from ats_analyzer.config import (
    CONNECTION_PROBE_MAX_RETRIES,
    CONNECTION_PROBE_TIMEOUT_SECONDS,
    MODEL_SUGGESTIONS,
    PRIVACY_NOTICE,
    PROVIDER_LABELS,
)
from ats_analyzer.domain.exceptions import AtsAnalyzerError
from ats_analyzer.domain.models import ProviderConfig
from ats_analyzer.infrastructure.http_client import HttpJsonClient
from ats_analyzer.infrastructure.providers.factory import create_provider

_logger = logging.getLogger(__name__)

_SESSION_KEYS_TO_CLEAR = (
    "resume",
    "jd_text",
    "requirements",
    "requirements_version",
    "ats",
    "ai_feedback",
    "ai_label",
    "local_recs",
    "resume_uploader",
)


@dataclass
class SidebarState:
    """What the user picked. Call :meth:`build_config` only when needed."""

    provider_id: str

    def is_local(self) -> bool:
        return self.provider_id == "local"

    def build_config(self) -> ProviderConfig | None:
        """Assemble an (unvalidated) ProviderConfig from widget state."""
        if self.is_local():
            return None
        api_key = st.session_state.get(f"key_{self.provider_id}") or None
        raw_base_url = st.session_state.get(f"base_url_{self.provider_id}") or ""
        config = ProviderConfig(
            provider_id=self.provider_id,
            model=str(st.session_state.get(f"model_{self.provider_id}") or ""),
            base_url=raw_base_url.strip() or None,
            api_key=api_key,
        )
        return config


def render_sidebar() -> SidebarState:
    with st.sidebar:
        st.subheader("⚙️ AI provider (BYOK)")
        provider_id = st.selectbox(
            "Provider",
            options=list(PROVIDER_LABELS),
            format_func=PROVIDER_LABELS.get,
            key="provider_choice",
        )
        state = SidebarState(provider_id=provider_id)
        if state.is_local():
            st.caption("Local analysis needs no API key. Select a provider to enable AI advice.")
        else:
            _render_provider_inputs(state)
            _render_connection_test(state)
        _render_privacy()
        _render_reset()
    return state


def _render_provider_inputs(state: SidebarState) -> None:
    st.text_input("API key", type="password", key=f"key_{state.provider_id}")
    st.caption("Held in session memory only — never stored, logged or sent anywhere else.")
    suggestions = MODEL_SUGGESTIONS.get(state.provider_id, ())
    st.text_input("Model name", key=f"model_{state.provider_id}")
    if suggestions:
        st.caption(
            "Suggestions: "
            + ", ".join(suggestions)
            + " — use a general chat model; reasoning-only ones may reply with empty text."
        )
    if state.provider_id == "compatible":
        st.text_input(
            "Base URL", key=f"base_url_{state.provider_id}", placeholder="http://localhost:11434/v1"
        )
        st.caption("HTTPS for remote endpoints; http:// only for localhost/private addresses.")
    elif state.provider_id == "mistral":
        st.text_input(
            "Base URL (optional)", key="base_url_mistral", placeholder="https://api.mistral.ai/v1"
        )
        st.caption(
            "Leave empty for the official Mistral API. HTTPS for remote endpoints; "
            "http:// only for localhost/private addresses."
        )


def _render_connection_test(state: SidebarState) -> None:
    if st.button("Test connection", use_container_width=True):
        try:
            config = state.build_config()
            assert config is not None  # local mode never reaches here
            probe_timeout = min(config.timeout_seconds, CONNECTION_PROBE_TIMEOUT_SECONDS)
            with HttpJsonClient(
                timeout_seconds=probe_timeout, max_retries=CONNECTION_PROBE_MAX_RETRIES
            ) as client:
                provider = create_provider(config, http=client)
                with st.spinner("Testing connection…"):
                    result = run_connection_test(provider)
            _logger.info("probe outcome=ok provider=%s model=%s", config.provider_id, config.model)
            st.success(result)
        except AtsAnalyzerError as exc:
            _logger.warning(
                "probe outcome=failed provider=%s model=%s reason=%r",
                state.provider_id,
                state.build_config().model if state.build_config() else "",
                exc,
            )
            st.error(str(exc))


def _render_privacy() -> None:
    with st.expander("Privacy & data handling", expanded=False):
        st.markdown(PRIVACY_NOTICE)


def _render_reset() -> None:
    if st.button("Reset session", use_container_width=True):
        for key in _SESSION_KEYS_TO_CLEAR:
            st.session_state.pop(key, None)
        st.rerun()
