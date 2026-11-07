"""Regression tests for SidebarState.build_config widget-state handling."""

from __future__ import annotations

import pytest
import streamlit as st

from ats_analyzer.presentation.sidebar import SidebarState


@pytest.fixture(autouse=True)
def _clean_session_state():
    yield
    for key in list(st.session_state):
        st.session_state.pop(key)


def test_build_config_does_not_invent_base_url_when_widget_absent() -> None:
    # Regression: str(None) used to produce the literal "None" as a base URL,
    # tripping the "custom base URL only for OpenAI-compatible" validation
    # even though the user never typed a base URL.
    for provider_id in ("openai", "gemini", "mistral"):
        state = SidebarState(provider_id=provider_id)
        config = state.build_config()
        assert config.model == ""
        assert config.base_url is None


def test_build_config_reads_typed_mistral_base_url_stripped() -> None:
    st.session_state["model_mistral"] = "mistral-small-2603"
    st.session_state["key_mistral"] = "k"
    st.session_state["base_url_mistral"] = " https://x.test/v1 "
    config = SidebarState(provider_id="mistral").build_config()
    assert config.model == "mistral-small-2603"
    assert config.api_key is not None and config.api_key.get_secret_value() == "k"
    assert config.base_url == "https://x.test/v1"


def test_build_config_whitespace_only_base_url_becomes_none() -> None:
    st.session_state["base_url_mistral"] = "   "
    config = SidebarState(provider_id="mistral").build_config()
    assert config.base_url is None
