"""Tests for resume parsing: PDF, DOCX, TXT, failure paths, contact & skills."""

from __future__ import annotations

import io

import pytest
from docx import Document

from ats_analyzer.domain.exceptions import ParserError, UploadError
from ats_analyzer.infrastructure.parsers import parse_resume

# --- happy paths ----------------------------------------------------------


def test_parses_txt_with_contacts_sections_and_skills(sample_resume_text: str) -> None:
    resume = parse_resume("resume.txt", sample_resume_text.encode("utf-8"))
    assert resume.file_format == "txt"
    assert "jane.doe@example.com" in resume.emails
    assert any("555" in p for p in resume.phones)
    assert "summary" in resume.sections
    assert "experience" in resume.sections
    assert "education" in resume.sections
    assert "AWS" in resume.skills
    assert "PostgreSQL" in resume.skills
    assert resume.word_count > 80


def test_parses_pdf_in_memory(fake_pdf_text_bytes: bytes) -> None:
    resume = parse_resume("resume.pdf", fake_pdf_text_bytes)
    assert resume.file_format == "pdf"
    assert resume.pages == 1
    assert "jane.doe@example.com" in resume.emails
    assert "Python" in resume.skills
    assert "summary" in resume.sections


def test_parses_docx_in_memory(fake_docx_bytes: bytes) -> None:
    resume = parse_resume("resume.docx", fake_docx_bytes)
    assert resume.file_format == "docx"
    assert "jane.doe@example.com" in resume.emails
    assert "Kubernetes" in resume.skills


def test_skill_aliases_are_canonicalized_once(sample_resume_text: str) -> None:
    resume = parse_resume("resume.txt", sample_resume_text.encode())
    # resume text contains "CI/CD" once; must not duplicate
    assert resume.skills.count("CI/CD") == 1
    # "fastapi" written lowercase in a bullet should still map to lexicon FastAPI
    assert "FastAPI" in resume.skills


# --- upload failures -------------------------------------------------------


def test_rejects_unsupported_extension() -> None:
    with pytest.raises(UploadError, match="Unsupported file type"):
        parse_resume("resume.doc", b"x")


def test_rejects_empty_file() -> None:
    with pytest.raises(UploadError, match="empty"):
        parse_resume("resume.txt", b"")


def test_rejects_signature_mismatch(fake_pdf_text_bytes: bytes) -> None:
    with pytest.raises(UploadError, match="not a valid DOCX"):
        parse_resume("resume.docx", fake_pdf_text_bytes)


def test_rejects_binary_txt() -> None:
    with pytest.raises(UploadError, match="binary"):
        parse_resume("resume.txt", b"\x00\x01\x02binary")


# --- parse failures ---------------------------------------------------------


def test_corrupt_pdf_raises_parser_error() -> None:
    with pytest.raises(ParserError, match="could not be parsed"):
        parse_resume("resume.pdf", b"%PDF-1.4 broken bytes")


def test_image_only_pdf_explains_ocr(image_only_pdf_bytes: bytes) -> None:
    with pytest.raises(ParserError, match="OCR is required"):
        parse_resume("resume.pdf", image_only_pdf_bytes)


def test_empty_txt_raises(sample_resume_text: str) -> None:
    with pytest.raises(ParserError, match="empty"):
        parse_resume("resume.txt", b"   \n\n  ")


def test_empty_docx_raises() -> None:
    document = Document()
    target = io.BytesIO()
    document.save(target)
    with pytest.raises(ParserError, match="no readable text"):
        parse_resume("resume.docx", target.getvalue())
