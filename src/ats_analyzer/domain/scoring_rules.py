"""Deterministic scoring vocabulary.

Pure data and constants: category weights, matching aliases, structure checks.
No I/O, no AI, no UI. The scoring engine (:mod:`ats_analyzer.application.scoring`)
implements the math; this module is its single source of truth for rules.
"""

from __future__ import annotations

import re

METHODOLOGY_LABEL = "Weighted heuristic model v1"

# Initial heuristic weights (percent). Configurable by editing this mapping.
CATEGORY_WEIGHTS: dict[str, float] = {
    "required_skills": 30.0,
    "preferred_skills": 15.0,
    "experience": 20.0,
    "education_certifications": 10.0,
    "keyword_coverage": 15.0,
    "structure": 10.0,
}

CATEGORY_LABELS: dict[str, str] = {
    "required_skills": "Required skills match",
    "preferred_skills": "Preferred skills match",
    "experience": "Experience relevance",
    "education_certifications": "Education & certifications",
    "keyword_coverage": "Keyword coverage",
    "structure": "Resume structure & readability",
}

METHODOLOGY_NOTE = (
    "Scores come from a transparent weighted keyword-and-structure heuristic "
    "(required skills 30%, preferred skills 15%, experience 20%, "
    "education/certifications 10%, keyword coverage 15%, structure 10%). "
    "Categories that cannot be judged from the resume alone are excluded from "
    "the overall score instead of being guessed. These weights are initial "
    "heuristics, not industry-standard ATS weights."
)

NOT_A_PREDICTION_NOTE = (
    "This is an estimated resume-to-job-description compatibility score. "
    "It is not a prediction of ATS acceptance, recruiter screening, or "
    "interview success."
)

# --- Structure checks (5 checks x 2 points = structure weight of 10) ------

STRUCTURE_CHECK_POINTS = 2.0

STRUCTURE_CHECKS: tuple[str, ...] = (
    "Contact information (email and phone)",
    "Professional summary or profile section",
    "Core sections present (experience, education, skills)",
    "Action-driven, quantified bullet points",
    "Healthy resume length (150-1,800 words)",
)

# --- Skill matching -------------------------------------------------------
# Display name -> lowercase aliases. Matching checks the display name and
# every alias with word boundaries, so "JS" matches "JavaScript" once even
# if both spellings appear (results are deduplicated by display name).

SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": (
        "py",
        "python3",
    ),
    "JavaScript": ("js", "es6", "ecmascript", "javascript"),
    "TypeScript": ("ts",),
    "Java": ("java",),
    "C#": ("c#", "csharp", "c sharp"),
    "C++": ("c++", "cpp"),
    "Go": ("golang",),
    "Rust": ("rust",),
    "React": ("react", "reactjs", "react.js"),
    "Vue.js": ("vue", "vuejs", "vue.js"),
    "Angular": ("angularjs", "angular"),
    "Node.js": ("nodejs", "node.js", "node"),
    "Django": ("django",),
    "Flask": ("flask",),
    "FastAPI": ("fastapi",),
    "Spring": ("spring", "springboot", "spring boot"),
    "SQL": ("sql",),
    "PostgreSQL": ("postgresql", "postgres", "psql"),
    "MySQL": ("mysql",),
    "MongoDB": ("mongodb", "mongo"),
    "Redis": ("redis",),
    "NoSQL": ("nosql",),
    "Docker": ("docker", "containerization"),
    "Kubernetes": ("kubernetes", "k8s"),
    "AWS": ("aws", "amazon web services"),
    "Azure": ("azure", "microsoft azure"),
    "GCP": ("gcp", "google cloud", "google cloud platform"),
    "Terraform": ("terraform",),
    "CI/CD": ("ci/cd", "cicd", "ci cd"),
    "Git": ("git",),
    "Linux": ("linux", "unix"),
    "Bash": ("bash", "shell scripting"),
    "REST APIs": ("rest", "restful", "rest api", "rest apis", "restful api"),
    "GraphQL": ("graphql",),
    "gRPC": ("grpc",),
    "Kafka": ("kafka",),
    "RabbitMQ": ("rabbitmq",),
    "Spark": ("spark", "apache spark"),
    "Airflow": ("airflow", "apache airflow"),
    "dbt": ("dbt",),
    "Snowflake": ("snowflake",),
    "Machine Learning": ("machine learning", "ml"),
    "Deep Learning": ("deep learning", "neural networks"),
    "NLP": ("natural language processing", "nlp"),
    "Computer Vision": ("computer vision",),
    "LLMs": ("large language models", "llm", "llms"),
    "Pandas": ("pandas",),
    "NumPy": ("numpy",),
    "PyTorch": ("pytorch",),
    "TensorFlow": ("tensorflow", "tf"),
    "Scikit-learn": ("scikit-learn", "sklearn"),
    "HTML": ("html", "html5"),
    "CSS": ("css", "css3", "scss"),
    "Tailwind CSS": ("tailwind", "tailwindcss"),
    "Figma": ("figma",),
}

# --- Document section headings --------------------------------------------

SECTION_HEADERS: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "professional summary", "profile", "objective", "about me", "about"),
    "experience": (
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
        "career history",
        "industry experience",
    ),
    "education": ("education", "academics", "academic background", "qualifications"),
    "skills": ("skills", "technical skills", "skills & tools", "technologies", "core competencies"),
    "certifications": ("certifications", "certificates", "licenses", "licenses & certifications"),
    "projects": ("projects", "personal projects", "selected projects", "key projects"),
}

# Matching falls back to these canonical headings in priority order for UI hints.
CANONICAL_SECTIONS: tuple[str, ...] = ("experience", "education", "skills")

# --- Language used by structure / recommendation heuristics ----------------

ACTION_VERBS: frozenset[str] = frozenset(
    {
        "led",
        "built",
        "designed",
        "implemented",
        "improved",
        "reduced",
        "increased",
        "launched",
        "migrated",
        "automated",
        "delivered",
        "owned",
        "architected",
        "optimized",
        "created",
        "developed",
        "drove",
        "scaled",
        "refactored",
        "introduced",
        "mentored",
        "directed",
        "coordinated",
        "established",
        "spearheaded",
        "negotiated",
        "analyzed",
        "shipped",
        "presented",
        "streamlined",
    }
)

# English stopwords used when deriving fallback keywords from a job description.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "must",
        "new",
        "not",
        "of",
        "on",
        "or",
        "our",
        "per",
        "plus",
        "should",
        "so",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "up",
        "us",
        "was",
        "we",
        "were",
        "will",
        "with",
        "would",
        "you",
        "your",
        "who",
        "what",
        "when",
        "where",
        "why",
        "all",
        "also",
        "more",
        "most",
        "other",
        "some",
        "work",
        "working",
        "works",
        "must-have",
        "nice",
        "years",
        "year",
        "experience",
        "role",
        "team",
        "teams",
        "strong",
        "able",
        "etc",
        "using",
        "use",
        "within",
        "about",
        "across",
        "while",
    }
)

YEARS_PATTERN = r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b"

# Display name -> detection regex pattern (case-insensitive).
DEGREE_TERMS: tuple[tuple[str, str], ...] = (
    ("Associate degree", r"associate'?s?\s+(?:degree|of\b)|a\.?a\.?s\b"),
    ("Bachelor's degree", r"bachelor'?s?\b|b\.?s\.?c?\.?\b|beng\b|b\.?tech\b|b\.?sc\b"),
    ("Master's degree", r"master'?s?\b|m\.?s\.?c?\.?\b|m\.?sc\b|m\.?eng\b|m\.?tech\b"),
    ("MBA", r"\bmba\b"),
    ("PhD", r"ph\.?d\b|doctorate|doctoral"),
)

MAX_SKILL_TERM_LENGTH = 40  # sanity cap for extracted requirement terms
MAX_EXPERIENCE_YEARS = 45  # ignore absurd "35+ years" parser artifacts beyond this

# --- Shared skill-term detection -------------------------------------------
# Used by both document parsers and job-description extraction so matching is
# alias-aware, word-bounded and canonicalized in exactly one place.

_NON_WORD_RE = re.compile(r"[^a-z0-9+#./\- ]+")


def detect_skill_terms(text: str) -> list[str]:
    """Return canonical lexicon skill names present in ``text``.

    Aliases map to display names ("k8s" -> "Kubernetes"); matching uses word
    boundaries ("spy" never matches "py"); each display name appears at most
    once, so "JS" and "JavaScript" both in the text count as one skill.
    """
    haystack = re.sub(r"\s+", " ", _NON_WORD_RE.sub(" ", text.lower()))
    matched: list[str] = []
    for display in sorted(SKILL_ALIASES):
        for term in (display.lower(), *SKILL_ALIASES[display]):
            if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", haystack):
                matched.append(display)
                break
    return matched
