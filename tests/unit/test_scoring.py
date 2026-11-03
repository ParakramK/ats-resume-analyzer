"""Deterministic scoring tests: matching, aliases, dedup, fairness, edge cases.

Scoring must be fully deterministic: run any test twice, get the same number.
"""

from __future__ import annotations

from ats_analyzer.application.extraction import extract_job_requirements
from ats_analyzer.application.scoring import (
    CATEGORY_WEIGHTS,
    _term_matches,
    score_resume,
)
from ats_analyzer.domain.models import JobRequirements
from ats_analyzer.infrastructure.parsers import parse_resume


def _resume(text: str):
    return parse_resume("resume.txt", text.encode("utf-8"))


def test_weights_sum_to_100() -> None:
    assert sum(CATEGORY_WEIGHTS.values()) == 100.0


def test_strong_match_scores_high(sample_resume_text, sample_jd_text) -> None:
    ats = score_resume(_resume(sample_resume_text), extract_job_requirements(sample_jd_text))
    assert 60 <= ats.overall <= 100
    assert {c.key for c in ats.categories} == set(CATEGORY_WEIGHTS)
    assert "Python" in ats.matched_required
    assert len(ats.strengths) >= 1


def test_deterministic_output(sample_resume_text, sample_jd_text) -> None:
    jd = extract_job_requirements(sample_jd_text)
    first = score_resume(_resume(sample_resume_text), jd)
    second = score_resume(_resume(sample_resume_text), jd)
    assert first == second
    assert first.overall == second.overall


def test_alias_matching_counts_once() -> None:
    jd = JobRequirements(required_skills=["JS", "React", "javascript"])
    resume = _resume("Alex\nalex@x.io\nSKILLS\nReact and JavaScript\n")
    ats = score_resume(resume, jd)
    # "JS" and "javascript" are aliases of the same skill and "React" too; the
    # deduped JD list contains two logical skills, both matched.
    assert ats.matched_required.count("JS") == 1 or ats.matched_required.count("JS") == 1
    assert "React" in ats.matched_required
    assert "javascript" in ats.matched_required
    # category score reflects the deduped 3-item list with all matched
    required = [c for c in ats.categories if c.key == "required_skills"][0]
    assert required.score == 100.0


def test_word_boundaries_prevent_false_hits() -> None:
    assert _term_matches("the spy program", "py") is False
    assert _term_matches("go from proposal to close", "go") is True  # boundary hit is fine
    assert _term_matches("handled billing pipelines", "py") is False


def test_missing_required_deducts_with_notes(sample_resume_text, sample_jd_text) -> None:
    jd = extract_job_requirements(sample_jd_text)
    jd.required_skills = ["Python", "PostgreSQL", "Rust", "Golang"]
    ats = score_resume(_resume(sample_resume_text), jd)
    required = [c for c in ats.categories if c.key == "required_skills"][0]
    assert required.score == 50.0
    assert sorted(ats.missing_required) == ["Golang", "Rust"]
    assert any("Deducted" in note for note in required.notes)


def test_unlisted_required_skills_are_excluded_not_penalized() -> None:
    resume = _resume(
        "Alex\nalex@x.io\n555-010-9876\nSUMMARY\nSolid generalist.\n"
        "EXPERIENCE\n- shipped things fast *\nEDUCATION\nB.S.\nSKILLS\nPython\n"
    )
    ats = score_resume(resume, JobRequirements(keywords=["python"]))
    required = [c for c in ats.categories if c.key == "required_skills"][0]
    assert required.applicable is False
    assert required.contribution == 0.0
    # overall over applicable weights only
    assert ats.overall >= 0 and ats.overall <= 100


def test_experience_meets_requirement(sample_resume_text, sample_jd_text) -> None:
    jd = extract_job_requirements(sample_jd_text)
    jd.min_years_experience = 5
    ats = score_resume(_resume(sample_resume_text), jd)
    experience = [c for c in ats.categories if c.key == "experience"][0]
    assert experience.score == 100.0
    assert any("meets" in note for note in experience.notes)


def test_experience_shortfall_is_proportional() -> None:
    resume = _resume("Alex\nalex@x.io\nSUMMARY\n3 years of experience.\nSKILLS\nPython\n")
    ats = score_resume(resume, JobRequirements(min_years_experience=10, keywords=["python"]))
    experience = [c for c in ats.categories if c.key == "experience"][0]
    assert experience.score == 30.0
    assert any("proportional" in note.lower() for note in experience.notes)


def test_experience_unstated_is_half_credit_with_note() -> None:
    resume = _resume("Alex\nalex@x.io\nSKILLS\nPython\nEXPERIENCE\nBuilt many APIs.\n")
    ats = score_resume(resume, JobRequirements(min_years_experience=5, keywords=["python"]))
    experience = [c for c in ats.categories if c.key == "experience"][0]
    assert experience.score == 50.0
    assert any("half credit" in note for note in experience.notes)


def test_experience_without_jd_requirement_excluded() -> None:
    resume = _resume("Alex\nalex@x.io\nSKILLS\nPython\n")
    ats = score_resume(resume, JobRequirements())
    experience = [c for c in ats.categories if c.key == "experience"][0]
    assert experience.applicable is False


def test_certification_match_counts(sample_resume_text, sample_jd_text) -> None:
    jd = extract_job_requirements(sample_jd_text)
    jd.certifications = [
        "AWS Solutions Architect"
    ]  # resume has "AWS Certified Solutions Architect"
    ats = score_resume(_resume(sample_resume_text), jd)
    education = [c for c in ats.categories if c.key == "education_certifications"][0]
    # education degree (Bachelor's) matched; certification matches by alias word set
    assert education.score == 100.0


def test_keyword_coverage_and_dedup() -> None:
    resume = _resume("Alex\nalex@x.io\nSUMMARY\nImproved latency by 40%.\nSKILLS\nPython\n")
    jd = JobRequirements(keywords=["latency", "latency", "python", "kafka"])
    ats = score_resume(resume, jd)
    keywords = [c for c in ats.categories if c.key == "keyword_coverage"][0]
    # 4 raw -> 3 unique: latency hit, python hit, kafka miss => 2/3
    assert keywords.score == round(2 / 3 * 100, 1)
    assert ats.missing_keywords == ["kafka"]


def test_structure_checks(sample_resume_text) -> None:
    good = _resume(sample_resume_text)
    ats = score_resume(good, JobRequirements())
    structure = [c for c in ats.categories if c.key == "structure"][0]
    assert structure.score == 100.0
    assert structure.notes and not any(str(n).startswith("No ") for n in structure.notes)


def test_structure_deducts_for_missing_contact(sample_resume_text: str) -> None:
    text = "\n".join(
        line
        for line in sample_resume_text.split("\n")
        if "example.com" not in line and "555" not in line
    )
    resume = _resume(text)
    ats = score_resume(resume, JobRequirements())
    structure = [c for c in ats.categories if c.key == "structure"][0]
    assert structure.score < 100.0
    assert any("No email or phone" in note for note in structure.notes)


def test_overall_is_weighted_average_of_applicable() -> None:
    resume = _resume(
        "Alex\nalex@x.io\n555-010-9876\nSUMMARY\nx\nEXPERIENCE\n- shipped *\n"
        "EDUCATION\nB.S.\nSKILLS\nPython\n"
    )
    requirements = JobRequirements(
        required_skills=["Python", "PostgreSQL"],
        preferred_skills=["Terraform"],
        keywords=["python", "docker"],
        min_years_experience=10,
    )
    ats = score_resume(resume, requirements)
    by_key = {c.key: c for c in ats.categories}
    applicable = [c for c in by_key.values() if c.applicable]
    total = sum(c.contribution for c in applicable) / sum(c.weight for c in applicable) * 100
    assert ats.overall == round(total, 1)


def test_result_is_serializable(sample_resume_text, sample_jd_text) -> None:
    ats = score_resume(_resume(sample_resume_text), extract_job_requirements(sample_jd_text))
    dumped = ats.model_dump_json()
    assert "overall" in dumped
    assert round(ats.overall, 1) == ats.overall
