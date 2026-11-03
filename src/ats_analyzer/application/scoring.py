"""Deterministic ATS scoring engine: pure, explainable, AI-free.

The score is a weighted keyword-and-structure heuristic. Nothing here imports
providers, LangChain, or the UI; identical inputs always produce identical
outputs. Categories that cannot be judged from the inputs are excluded from
the overall score instead of being guessed at.
"""

from __future__ import annotations

import re

from ats_analyzer.domain.models import AtsResult, CategoryScore, JobRequirements, ParsedResume
from ats_analyzer.domain.scoring_rules import (
    CATEGORY_LABELS,
    CATEGORY_WEIGHTS,
    DEGREE_TERMS,
    MAX_EXPERIENCE_YEARS,
    METHODOLOGY_LABEL,
    METHODOLOGY_NOTE,
    SKILL_ALIASES,
    STOPWORDS,
    STRUCTURE_CHECK_POINTS,
    STRUCTURE_CHECKS,
    YEARS_PATTERN,
)

# Reverse alias index: "k8s" -> "Kubernetes". Built once at import.
_REVERSE_ALIASES: dict[str, str] = {}
for _display, _aliases in SKILL_ALIASES.items():
    for _alias in (_display.lower(), *_aliases):
        _REVERSE_ALIASES.setdefault(_alias, _display)


def _normalized(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9+#./\- ]+", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned)


def _term_matches(haystack: str, term: str) -> bool:
    """Alias-aware, word-boundary match of a term inside normalized text."""
    candidates = {term.lower().strip()}
    display = _REVERSE_ALIASES.get(term.lower().strip())
    if display is not None:  # term is a known skill: extend with its aliases
        candidates.update({display.lower(), *SKILL_ALIASES[display]})
    normalized = " ".join(re.split(r"[^a-z0-9+#]+", term.lower().strip()))
    if normalized:
        candidates.add(normalized)
    if any(
        re.search(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)", haystack)
        for candidate in candidates
        if len(candidate) >= 2
    ):
        return True
    # Phrase fallback: multi-word terms (e.g. "AWS Solutions Architect" vs
    # "AWS Certified Solutions Architect") count as matched when every
    # significant token appears in the text, in any order.
    tokens = [
        token
        for token in re.split(r"[^a-z0-9+#]+", term.lower().strip())
        if len(token) >= 2 and token not in STOPWORDS
    ]
    if len(tokens) >= 2:
        return all(
            re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", haystack) for token in tokens
        )
    return False


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered


def score_resume(
    resume: ParsedResume,
    requirements: JobRequirements,
    weights: dict[str, float] | None = None,
) -> AtsResult:
    """Compute the full deterministic ATS result for a resume/JD pair."""
    weights = weights or CATEGORY_WEIGHTS
    haystack = _normalized(resume.text)
    categories = [
        _skills_category(
            resume, requirements.required_skills, haystack, "required_skills", weights
        ),
        _skills_category(
            resume, requirements.preferred_skills, haystack, "preferred_skills", weights
        ),
        _experience_category(haystack, requirements, weights),
        _education_category(haystack, requirements, weights),
        _keyword_category(haystack, requirements, weights),
        _structure_category(resume, weights),
    ]
    applicable = [category for category in categories if category.applicable]
    divisor = sum(category.weight for category in applicable)
    overall = (
        round(sum(category.contribution for category in applicable) / divisor * 100, 1)
        if divisor > 0
        else 0.0
    )
    return AtsResult(
        overall=overall,
        categories=categories,
        matched_required=_matched_names(requirements.required_skills, haystack),
        missing_required=_missing_names(requirements.required_skills, haystack),
        matched_preferred=_matched_names(requirements.preferred_skills, haystack),
        missing_preferred=_missing_names(requirements.preferred_skills, haystack),
        matched_keywords=_matched_keywords(haystack, requirements.keywords),
        missing_keywords=_missing_keywords(haystack, requirements.keywords),
        strengths=_strengths(categories, requirements, haystack),
        weaknesses=_weaknesses(categories, requirements, haystack),
        methodology=f"{METHODOLOGY_LABEL}. {METHODOLOGY_NOTE}",
    )


# --- category implementations ------------------------------------------------


def _matched_names(skills: list[str], haystack: str) -> list[str]:
    return [name for name in _unique(skills) if _term_matches(haystack, name)]


def _missing_names(skills: list[str], haystack: str) -> list[str]:
    return [name for name in _unique(skills) if not _term_matches(haystack, name)]


def _skills_category(
    resume: ParsedResume, skills: list[str], haystack: str, key: str, weights: dict[str, float]
) -> CategoryScore:
    unique = _unique(skills)
    if not unique:
        kind = "required" if key == "required_skills" else "preferred"
        return _category(
            key,
            weights[key],
            0.0,
            applicable=False,
            notes=[f"No {kind} skills listed; category did not influence the overall score."],
        )
    matched = [name for name in unique if _term_matches(haystack, name)]
    missing = [name for name in unique if name not in matched]
    score = round(len(matched) / len(unique) * 100, 1)
    notes: list[str] = [f"{len(matched)} of {len(unique)} matched."]
    if missing:
        notes.append("Deducted for terms not found in the resume text.")
    if missing and len(missing) <= 10:
        notes.append("Consider adding (only if you genuinely have it): " + ", ".join(missing))
    return _category(key, weights[key], score, notes=notes)


def _experience_category(
    haystack: str, requirements: JobRequirements, weights: dict[str, float]
) -> CategoryScore:
    required_years = requirements.min_years_experience
    if required_years is None:
        return _category(
            "experience",
            weights["experience"],
            0.0,
            applicable=False,
            notes=["The job description states no minimum years of experience; category excluded."],
        )
    stated = _stated_years(haystack)
    if stated is None:
        return _category(
            "experience",
            weights["experience"],
            50.0,
            notes=[
                "No explicit total years of experience found in the resume; half credit assumed "
                "until you state it plainly (e.g. '7 years of experience')."
            ],
        )
    if stated >= required_years:
        return _category(
            "experience",
            weights["experience"],
            100.0,
            notes=[f"Stated ~{stated} years meets the {required_years}+ years requirement."],
        )
    score = round(max(0.0, 100.0 * stated / required_years), 1)
    return _category(
        "experience",
        weights["experience"],
        score,
        notes=[
            f"Stated ~{stated} years vs the {required_years}+ years requirement "
            "(proportional credit)."
        ],
    )


def _stated_years(haystack: str) -> int | None:
    values = [int(match.group(1)) for match in re.finditer(YEARS_PATTERN, haystack, re.IGNORECASE)]
    plausible = [years for years in values if 0 < years <= MAX_EXPERIENCE_YEARS]
    return max(plausible) if plausible else None


def _education_category(
    haystack: str, requirements: JobRequirements, weights: dict[str, float]
) -> CategoryScore:
    checks: list[tuple[str, bool]] = []
    for display, pattern in DEGREE_TERMS:
        for requirement in requirements.education:
            if display.lower() in requirement.lower():
                checks.append((display, re.search(pattern, haystack) is not None))
    for requirement in requirements.certifications:
        term = requirement.strip()
        if term:
            checks.append((term, _term_matches(haystack, term)))
    if not checks:
        return _category(
            "education_certifications",
            weights["education_certifications"],
            0.0,
            applicable=False,
            notes=["No education or certification requirements found; category excluded."],
        )
    matched = [name for name, hit in checks if hit]
    score = round(len(matched) / len(checks) * 100, 1)
    notes = [f"{len(matched)} of {len(checks)} education/certification requirements matched."]
    missing = [name for name, hit in checks if not hit]
    if missing:
        notes.append("Not found in resume: " + ", ".join(missing))
    return _category(
        "education_certifications", weights["education_certifications"], score, notes=notes
    )


def _keyword_category(
    haystack: str, requirements: JobRequirements, weights: dict[str, float]
) -> CategoryScore:
    keywords = [keyword for keyword in _unique(requirements.keywords) if len(keyword.strip()) >= 2]
    if not keywords:
        return _category(
            "keyword_coverage",
            weights["keyword_coverage"],
            0.0,
            applicable=False,
            notes=["No keywords provided; category excluded."],
        )
    matched = _matched_keywords(haystack, keywords)
    missing = _missing_keywords(haystack, keywords)
    score = round(len(matched) / len(keywords) * 100, 1)
    notes = [f"{len(matched)} of {len(keywords)} keywords appear in the resume."]
    if missing and len(missing) <= 12:
        notes.append("Missing (add naturally, only where truthful): " + ", ".join(missing))
    return _category("keyword_coverage", weights["keyword_coverage"], score, notes=notes)


def _matched_keywords(haystack: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in _unique(keywords) if _term_matches(haystack, keyword)]


def _missing_keywords(haystack: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in _unique(keywords) if not _term_matches(haystack, keyword)]


def _structure_category(resume: ParsedResume, weights: dict[str, float]) -> CategoryScore:
    points = 0.0
    notes: list[str] = []
    contacts_ok = bool(resume.emails) or bool(resume.phones)
    points += STRUCTURE_CHECK_POINTS if contacts_ok else 0.0
    if not contacts_ok:
        notes.append("No email or phone found in contact area — ATS parsers need one.")
    else:
        notes.append(STRUCTURE_CHECKS[0])

    summary_ok = bool(resume.sections.get("summary"))
    points += STRUCTURE_CHECK_POINTS if summary_ok else 0.0
    if not summary_ok:
        notes.append("No summary/profile section detected.")

    core = [name for name in ("experience", "education", "skills") if name in resume.sections]
    points += STRUCTURE_CHECK_POINTS * len(core) / 3
    if len(core) < 3:
        missing_sections = ", ".join(
            name for name in ("experience", "education", "skills") if name not in core
        )
        notes.append(f"Sections not detected ({missing_sections}); headings may be non-standard.")

    experience_text = "\n".join(
        resume.sections.get(name, "") for name in ("experience", "projects")
    )
    bullets = [line for line in experience_text.split("\n") if _is_bullet(line)]
    if bullets:
        quantified = [line for line in bullets if re.search(r"\d", line)]
        points += STRUCTURE_CHECK_POINTS * len(quantified) / len(bullets)
        if len(quantified) < len(bullets):
            notes.append("Some accomplishment bullets have no measurable numbers.")
    else:
        notes.append("No bullet points detected in the experience section.")

    words = resume.word_count
    if 150 <= words <= 1800:
        points += STRUCTURE_CHECK_POINTS
    elif words < 150:
        points += STRUCTURE_CHECK_POINTS * max(0.0, words / 150)
        notes.append(f"Only ~{words} words; detail may be too thin for keyword matching.")
    else:
        notes.append(f"~{words} words is unusually long; some ATS parsers may truncate.")

    score = round(points / (STRUCTURE_CHECK_POINTS * len(STRUCTURE_CHECKS)) * 100, 1)
    return _category("structure", weights["structure"], score, notes=notes)


def _is_bullet(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and bool(re.match(r"^[-•\u2013\u2022*>\u25cf▪]+", stripped))


# --- assembly helpers ---------------------------------------------------------


def _category(
    key: str,
    weight: float,
    score: float,
    *,
    applicable: bool = True,
    notes: list[str] | None = None,
) -> CategoryScore:
    return CategoryScore(
        key=key,
        label=CATEGORY_LABELS[key],
        weight=weight,
        score=score,
        contribution=round(score * weight / 100, 2),
        applicable=applicable,
        notes=notes or [],
    )


def _strengths(
    categories: list[CategoryScore], requirements: JobRequirements, haystack: str
) -> list[str]:
    strengths: list[str] = []
    matched_required = _matched_names(requirements.required_skills, haystack)
    if len(matched_required) >= 3:
        strengths.append("Strong alignment on required skills: " + ", ".join(matched_required[:5]))
    elif matched_required:
        strengths.append("Matches required skills: " + ", ".join(matched_required))
    stated_years = _stated_years(haystack)
    if (
        requirements.min_years_experience
        and stated_years
        and stated_years >= requirements.min_years_experience
    ):
        strengths.append(f"Stated experience (~{stated_years} years) meets the requirement.")
    scores = {category.key: category.score for category in categories}
    if scores.get("keyword_coverage", 0) >= 70:
        strengths.append("Keyword coverage against this job description is solid.")
    if scores.get("structure", 0) >= 80:
        strengths.append("Clean structure that automated parsers read easily.")
    return strengths


def _weaknesses(
    categories: list[CategoryScore], requirements: JobRequirements, haystack: str
) -> list[str]:
    weaknesses: list[str] = []
    missing_required = _missing_names(requirements.required_skills, haystack)
    if missing_required:
        shown = ", ".join(missing_required[:8])
        weaknesses.append(f"Required skill terms not shown in the resume: {shown}.")
    missing_keywords = _missing_keywords(haystack, requirements.keywords)
    if len(missing_keywords) >= 3:
        weaknesses.append(
            f"{len(missing_keywords)} job keywords are absent; "
            "weave the truthful ones into achievements."
        )
    for category in categories:
        if category.applicable and category.score < 50:
            weaknesses.append(f"{category.label} scored low ({category.score:.0f}%).")
    return weaknesses
