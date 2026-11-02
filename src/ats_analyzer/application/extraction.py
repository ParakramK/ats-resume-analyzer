"""Deterministic job-description parsing — no AI involved.

Extracts a :class:`JobRequirements` draft that the user reviews and edits in
the UI before scoring. The parser is deliberately conservative: bullets it
cannot confidently classify stay unclassified rather than guessed, and the
keyword pool mixes lexicon skills with frequent significant terms.
"""

from __future__ import annotations

import re
from collections import Counter

from ats_analyzer.domain.models import JobRequirements
from ats_analyzer.domain.scoring_rules import (
    DEGREE_TERMS,
    MAX_SKILL_TERM_LENGTH,
    STOPWORDS,
    YEARS_PATTERN,
    detect_skill_terms,
)

_BULLET_RE = re.compile(r"^[-•*–>*+\u2022\u25aa\u25cf\u2043\u2219▪]+\s*|^\(?\d+[).:]\s+")
_HEADING_RE = re.compile(
    r"^[#*_\-\s]{0,6}(?P<name>[A-Za-z][A-Za-z0-9 '’/,+&-]{0,48})[#*_\-\s:]{0,6}$"
)
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z+#.]{3,}")

# JD section headings -> bucket. Order matters for longest-match first.
_JD_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "responsibilities",
        ("responsibilities", "duties", "what you'll do", "what you will do", "in this role"),
    ),
    (
        "required",
        (
            "required qualifications",
            "requirements",
            "required",
            "must have",
            "must-haves",
            "qualifications",
            "what you'll need",
            "what you will need",
            "basic qualifications",
            "minimum qualifications",
            "skills required",
        ),
    ),
    (
        "preferred",
        (
            "preferred qualifications",
            "preferred",
            "nice to have",
            "nice-to-have",
            "bonus",
            "bonus points",
            "preferred skills",
            "would be great",
            "desirable",
        ),
    ),
    ("education", ("education", "educational background", "qualifications and education")),
    ("certifications", ("certifications", "certificates", "licenses")),
    ("skills", ("skills", "technical skills", "skills and experience", "expertise")),
)

_SENTENCE_HINTS = ("years of", "experience with", "you will", "you will be", "word", "the ability")


def extract_job_requirements(text: str) -> JobRequirements:
    """Parse pasted job-description text into an editable requirements draft."""
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    title = _pick_title(lines)
    buckets = _route_lines(lines)

    return JobRequirements(
        title=title,
        responsibilities=_clean_unique(buckets["responsibilities"], allow_long=True),
        required_skills=_skills_from(buckets["required"]),
        preferred_skills=_skills_from(buckets["preferred"]),
        keywords=_derive_keywords(text, buckets),
        min_years_experience=_min_years(text),
        education=_education(buckets),
        certifications=_clean_unique(buckets["certifications"], allow_long=True),
    )


def _pick_title(lines: list[str]) -> str:
    for line in lines:
        lowered = line.lower()
        if not lowered or lowered.startswith(("job description", "about us", "about the company")):
            continue
        if len(line) <= 90 and not line.endswith((".", ":", "!")):
            return line.strip()
    return ""


def _route_lines(lines: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {name: [] for name, _ in _JD_SECTIONS}
    current: str | None = None
    for line in lines:
        if not line:
            continue
        canonical = _jd_heading(line)
        if canonical is not None:
            current = canonical
            continue
        if current is None:
            continue
        if any(
            line.lower().startswith(h) for h in ("about us", "about the", "our company", "why work")
        ):
            current = None
            continue
        cleaned = _BULLET_RE.sub("", line).strip()
        if cleaned and len(cleaned) <= 200:
            buckets[current].append(cleaned)
    return buckets


def _jd_heading(line: str) -> str | None:
    match = _HEADING_RE.match(line)
    if match is None:
        return None
    normalized = " ".join(match.group("name").lower().rstrip(":. ").split())
    for canonical, aliases in _JD_SECTIONS:
        for alias in aliases:
            if normalized == alias:
                return canonical
            # tolerate headings like "required qualifications you'll bring"
            if normalized.startswith(alias) and len(normalized) - len(alias) <= 24:
                return canonical
    return None


def _skills_from(items: list[str]) -> list[str]:
    """Lexicon skills inside a bullet win; untouched short tech terms stay verbatim."""
    skills: list[str] = []
    for item in items:
        matched = detect_skill_terms(item)
        if matched:
            skills.extend(matched)
            continue
        lowered = item.lower().rstrip(".")
        if (
            len(item) <= MAX_SKILL_TERM_LENGTH
            and lowered.count(" ") <= 3
            and not _sentence_like(item)
        ):
            skills.append(item.rstrip(" .,;"))
    return _clean_unique(skills, allow_long=False)


def _sentence_like(item: str) -> bool:
    lowered = item.lower()
    return any(hint in lowered for hint in _SENTENCE_HINTS)


def _clean_unique(items: list[str], *, allow_long: bool) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        candidate = item if allow_long else item[:MAX_SKILL_TERM_LENGTH]
        key = " ".join(candidate.lower().split())
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _min_years(text: str) -> int | None:
    years = [int(match.group(1)) for match in re.finditer(YEARS_PATTERN, text, re.IGNORECASE)]
    if not years:
        return None
    plausible = [years_count for years_count in years if 0 < years_count <= 30]
    return max(plausible) if plausible else None


def _education(buckets: dict[str, list[str]]) -> list[str]:
    found: list[str] = []
    context = buckets.get("education", []) or buckets.get("required", [])
    for display, pattern in DEGREE_TERMS:
        for line in context:
            if re.search(pattern, line, re.IGNORECASE):
                found.append(display)
                break
    return _dedupe(found)


def _derive_keywords(text: str, buckets: dict[str, list[str]]) -> list[str]:
    """Lexicon skills first, then frequent significant words as fallback."""
    keywords: list[str] = _clean_unique(
        [
            kw
            for bucket in ("skills", "required", "preferred")
            for kw in _skills_from(buckets[bucket])
        ],
        allow_long=False,
    )
    words = Counter(
        token.rstrip(".")
        for token in _TOKEN_RE.findall(text.lower())
        if token.rstrip(".") not in STOPWORDS
    )
    frequent = [word for word, count in words.most_common(18) if count >= 2]
    return _dedupe([*keywords, *frequent])[:24]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
