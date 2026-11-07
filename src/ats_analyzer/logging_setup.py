"""Central logging configuration.

Privacy rules baked in: log events carry metadata only — host, model name,
status codes, durations, file type/size, skill counts. Never log API keys,
resume text, job description text, parsed contacts, or AI reply content.

Enable a rotating file (for debugging long sessions) with ATS_LOG_FILE;
control verbosity with ATS_LOG_LEVEL (default INFO).
"""

from __future__ import annotations

import logging
import logging.handlers
import os

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s :: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging() -> None:
    """Attach one console handler (+ optional rotating file) exactly once."""
    logger = logging.getLogger("ats_analyzer")
    if logger.handlers:  # Streamlit re-runs this script on every interaction
        return
    level = getattr(logging, (os.environ.get("ATS_LOG_LEVEL") or "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(console)
    path = os.environ.get("ATS_LOG_FILE")
    if path:
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=1_000_000, backupCount=2, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(file_handler)
