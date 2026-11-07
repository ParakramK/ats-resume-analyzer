"""Application configuration: limits, environment overrides, and provider catalog.

Nothing in here is a secret. API keys are entered at runtime in the UI and
exist only in session memory — they are never read from files or environment.
"""

from __future__ import annotations

import os

APP_NAME = "ATS Resume Analyzer"
LOGO_EMOJI = "📄"

# --- Upload and document limits -------------------------------------------

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB per document
MAX_RESUME_PAGES = 40  # guard against pathological PDFs

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt")

# --- HTTP client defaults (developer-overridable via environment) ----------


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


DEFAULT_TIMEOUT_SECONDS = _env_float("ATS_HTTP_TIMEOUT", 30.0)
MAX_RETRIES = max(0, _env_int("ATS_HTTP_MAX_RETRIES", 2))  # 429/5xx/transient only

# The connection probe should answer quickly or report failure; it never retries.
CONNECTION_PROBE_TIMEOUT_SECONDS = 10.0
CONNECTION_PROBE_MAX_RETRIES = 0
MAX_RESPONSE_BYTES = max(1024, _env_int("ATS_HTTP_MAX_RESPONSE_BYTES", 2_000_000))

# --- AI provider catalog ---------------------------------------------------
# Model names are deliberately NOT hardcoded into requests: the UI collects a
# free-form model identifier per provider. The lists below are display-only
# suggestions shown under the model field, and change over time.

PROVIDER_LABELS: dict[str, str] = {
    "local": "Local only (no AI)",
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "mistral": "Mistral AI",
    "compatible": "OpenAI-compatible endpoint",
}

PROVIDER_REQUIRES_KEY = {"openai": True, "gemini": True, "mistral": True, "compatible": False}

MODEL_SUGGESTIONS: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-5.6", "gpt-5.4-mini", "gpt-5.4-nano"),
    "gemini": ("gemini-3.8-flash", "gemini-3.1-pro-preview"),
    "mistral": ("mistral-small-2603", "mistral-large-4"),
    "compatible": (),
}

# --- Privacy text (shared by sidebar and prompt boundaries) ----------------

PRIVACY_NOTICE = """
**Local-first by default.** Uploads are parsed in memory and are not written
to disk or a database. Deterministic ATS scoring, keyword matching, and report
generation all run locally with no API key.

**Cloud AI is opt-in and explicit.** When you run an *AI-enhanced* analysis or
ask an AI provider to extract job requirements, the resume text and job
description are sent over TLS to that provider's API using your own key.
Processing on a cloud provider is **not private in any absolute sense**: you
cannot control their logging, retention, storage, or human-review policies —
do not assume privacy guarantees from this app for cloud processing.

**Credentials.** API keys are typed into the sidebar, held only in server
session memory, never logged, never written to disk, and never persisted.
Closing the session discards them.

**No analytics, no tracking, no telemetry.**
""".strip()

AI_DISCLOSURE = (
    "This action sends your resume and job description content to the selected "
    "provider. Verify you trust the endpoint before continuing."
)
