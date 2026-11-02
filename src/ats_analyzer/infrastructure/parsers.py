"""Resume document parsers: PDF, DOCX and TXT -> :class:`ParsedResume`.

- Everything happens in memory: PDFs are opened from a byte stream, DOCX from
  ``BytesIO``. No temporary files, no disk writes.
- File type is checked by extension *and* magic bytes; a deliberately renamed
  file (e.g. a PDF called ``.docx``) is rejected.
- Malformed, empty, and image-only (scanned) documents raise user-safe errors;
  scanned PDFs explain that OCR is required but not bundled with this app.
"""

from __future__ import annotations

import io
import re

import pymupdf
from docx import Document

from ats_analyzer.config import MAX_RESUME_PAGES, MAX_UPLOAD_BYTES
from ats_analyzer.domain.exceptions import ParserError, UploadError
from ats_analyzer.domain.models import ParsedResume
from ats_analyzer.domain.scoring_rules import SECTION_HEADERS, detect_skill_terms

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d()\s.-]{7,}\d")
_LINK_RE = re.compile(r"(?:https?://|www\.)[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")

_HEADING_RE = re.compile(r"^[#*_\-\s]{0,6}(?P<name>[A-Za-z][A-Za-z0-9 &/+-]{0,48})[#*_\-\s:]{0,6}$")


def parse_resume(filename: str, data: bytes) -> ParsedResume:
    """Parse an uploaded resume into structured sections, contacts and skills."""
    suffix = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    if not filename or suffix not in ("pdf", "docx", "txt"):
        raise UploadError("Unsupported file type. Upload a PDF, DOCX or TXT resume.")
    if not data:
        raise UploadError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("The file exceeds the 15 MB limit.")

    if suffix == "pdf" and not data.startswith(b"%PDF"):
        raise UploadError("This file is not a valid PDF document.")
    if suffix == "docx" and not data.startswith(b"PK\x03\x04"):
        raise UploadError("This file is not a valid DOCX document.")
    if suffix == "txt" and b"\x00" in data:
        raise UploadError("The file looks binary; upload a plain text resume.")

    try:
        text, pages = _extract_text(suffix, data)
    except (ParserError, UploadError):
        raise
    except Exception:  # boundary: parser libraries raise a zoo of error types
        raise ParserError(
            "The document could not be parsed; it may be corrupt or password-protected."
        ) from None

    return _build(text, filename, suffix, pages)


def _extract_text(file_format: str, data: bytes) -> tuple[str, int]:
    if file_format == "pdf":
        return _extract_pdf(data)
    if file_format == "docx":
        return _extract_docx(data)
    return _extract_txt(data)


def _extract_pdf(data: bytes) -> tuple[str, int]:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        pages = doc.page_count
        if pages == 0:
            raise ParserError("The PDF has no pages.")
        if pages > MAX_RESUME_PAGES:
            raise ParserError("The PDF is too long for analysis (more than 40 pages).")
        page_texts = [doc[i].get_text("text") for i in range(pages)]
    text = "\n".join(t for t in page_texts)
    if len(text.strip()) < 30:
        raise ParserError(
            "No selectable text found; this looks like a scanned/image-only PDF. "
            "OCR is required but not bundled with this app — use a text-based PDF "
            "or a DOCX/TXT export."
        )
    return text, pages


def _extract_docx(data: bytes) -> tuple[str, int]:
    document = Document(io.BytesIO(data))
    lines = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text for cell in row.cells))
    text = "\n".join(lines)
    if not text.strip():
        raise ParserError("The DOCX document has no readable text.")
    return text, 1


def _extract_txt(data: bytes) -> tuple[str, int]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")
    if not text.strip():
        raise ParserError("The text file is empty.")
    return text, 1


def _build(text: str, filename: str, file_format: str, pages: int) -> ParsedResume:
    cleaned = _clean(text)
    return ParsedResume(
        text=cleaned,
        filename=filename,
        file_format=file_format,  # type: ignore[arg-type]
        pages=pages,
        word_count=len(cleaned.split()),
        emails=_find_emails(cleaned),
        phones=_find_phones(cleaned),
        links=_find_links(cleaned),
        skills=_detect_skills(cleaned),
        sections=_split_sections(cleaned),
    )


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)  # control characters
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_emails(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for candidate in _EMAIL_RE.findall(text):
        seen.setdefault(candidate.lower(), None)
    return list(seen)


def _find_phones(text: str) -> list[str]:
    found: dict[str, None] = {}
    for match in _PHONE_RE.findall(text):
        digits = re.sub(r"\D", "", match)
        if 8 <= len(digits) <= 15:
            found.setdefault(match.strip(), None)
    return list(found)


def _find_links(text: str) -> list[str]:
    found: dict[str, None] = {}
    for candidate in _LINK_RE.findall(text):
        found.setdefault(candidate.rstrip(".,"), None)
    return list(found)[:12]


def _detect_skills(text: str) -> list[str]:
    return detect_skill_terms(text)


def _split_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in text.split("\n"):
        canonical = _canonical_heading(line)
        if canonical is not None:
            if canonical not in sections:  # first occurrence wins
                if current is not None:
                    sections[current] = "\n".join(buffer).strip()
                current, buffer = canonical, []
                continue
            buffer.append(line)
        else:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip()
    return sections


def _canonical_heading(line: str) -> str | None:
    match = _HEADING_RE.match(line.strip())
    if match is None:
        return None
    normalized = " ".join(match.group("name").lower().rstrip(":. ").split())
    for canonical, aliases in SECTION_HEADERS.items():
        if normalized in aliases:
            return canonical
    return None
