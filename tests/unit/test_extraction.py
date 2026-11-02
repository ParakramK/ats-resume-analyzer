"""Tests for deterministic job-description extraction."""

from __future__ import annotations

from ats_analyzer.application.extraction import extract_job_requirements
from ats_analyzer.domain.models import JobRequirements


def test_extracts_title_sections_years_and_degrees(sample_jd_text: str) -> None:
    req = extract_job_requirements(sample_jd_text)
    assert req.title == "Senior Backend Engineer - Platform"
    assert req.min_years_experience == 5
    assert "Bachelor's degree" in req.education
    assert any("Python" in s for s in req.required_skills)
    assert any("Kubernetes" in s or "Docker" in s for s in req.required_skills)
    assert any("Kafka" in s for s in req.preferred_skills)
    assert len(req.responsibilities) >= 2


def test_lexicon_aliases_map_to_display_names() -> None:
    jd = "Backend Developer\nRequired:\n- k8s and docker\n- python\nPreferred:\n- psql\n"
    req = extract_job_requirements(jd)
    assert "Kubernetes" in req.required_skills
    assert "Docker" in req.required_skills
    assert "Python" in req.required_skills
    assert "PostgreSQL" in req.preferred_skills


def test_no_years_stays_none() -> None:
    jd = "Designer\nResponsibilities:\n- make things pretty\n"
    req = extract_job_requirements(jd)
    assert req.min_years_experience is None


def test_dedupes_repeated_skills() -> None:
    jd = "Data Engineer\nRequired:\n- Python\n- python\n- PYTHON\nPreferred:\n- Spark\n"
    req = extract_job_requirements(jd)
    assert req.required_skills.count("Python") == 1


def test_empty_jd_yields_empty_requirements() -> None:
    req = extract_job_requirements("   \n \n")
    assert req == JobRequirements()


def test_long_sentences_do_not_become_skills() -> None:
    jd = (
        "Program Manager\nRequired qualifications:\n"
        "- 5+ years of experience managing enterprise programs end to end\n"
        "- Strong communication skills across departments\n"
    )
    req = extract_job_requirements(jd)
    long_bullets = [r for r in req.required_skills if "experience managing" in r.lower()]
    assert long_bullets == []


def test_certifications_section_is_captured() -> None:
    jd = (
        "Cloud Consultant\nRequired:\n- Python\nCertifications:\n"
        "- AWS Certified Solutions Architect\n- Certified Kubernetes Administrator\n"
    )
    req = extract_job_requirements(jd)
    assert len(req.certifications) == 2
