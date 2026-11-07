# ATS Resume Analyzer

A privacy-first ATS resume analyzer with deterministic local scoring and optional AI feedback using your own API key.

Upload a resume (PDF, DOCX, or TXT) and paste a job description. The application scores the resume against the job requirements on your machine, explains every deduction in plain language, and produces recommendations plus JSON/PDF reports. AI feedback from OpenAI, Google Gemini, Mistral, or any OpenAI-compatible endpoint is optional, requires explicit consent, and is the only feature that ever contacts the network.

## Overview

1. **Upload** — the resume is parsed in memory. Files are never written to disk or a database.
2. **Job description** — requirements are extracted conservatively. Anything the parser cannot confidently classify stays unclassified rather than guessed. You review and edit the extracted requirements before scoring.
3. **Analyze** — the deterministic engine scores six categories and generates recommendations. This step needs no API key.
4. **Optional AI feedback** — after you explicitly consent, resume and job description text is sent over TLS to a provider you configured with your own key. Feedback is appended to the local score; the score itself never depends on it.

## Getting started

Requirements: Python 3.11 or later (3.12 recommended) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run streamlit run app.py
```

Open http://localhost:8501. Local analysis works immediately with no configuration.

To enable AI feedback: select a provider in the sidebar, enter your API key and a model name, run **Test connection** (sends a fixed probe sentence only — never resume content), then run **AI-enhanced analysis**.

### Running fully offline

With [Ollama](https://ollama.com) installed and running locally:

```bash
ollama pull llama3.1
```

In the sidebar, select **OpenAI-compatible endpoint**, set the base URL to `http://localhost:11434/v1`, set the model to `llama3.1`, and leave the key empty.

## Supported AI providers

| Provider | Key required | Endpoint | Suggested models |
| --- | --- | --- | --- |
| OpenAI | Yes | OpenAI REST API | `gpt-5.6`, `gpt-5.4-mini` |
| Google Gemini | Yes | Gemini REST API | `gemini-3.8-flash`, `gemini-3.5-flash-lite` |
| Mistral AI | Yes | Mistral REST API, or a custom base URL (HTTPS for remote) | `mistral-small-2603`, `mistral-large-4` |
| OpenAI-compatible | No | Any `https://` URL, or `http://` on localhost/private IPs | depends on the service |
| Local only (no AI) | — | none | — |

Model names are free-text; the suggestions in the UI are display-only. Prefer general-purpose chat models: reasoning-only models can answer with empty or thinking-only text and will not analyze.

## Scoring methodology

Each category is scored from 0 to 100 and weighted into an overall score (Weighted heuristic model v1):

| Category | Weight |
| --- | --- |
| Required skills match | 30 |
| Experience relevance | 20 |
| Keyword coverage | 15 |
| Preferred skills match | 15 |
| Education & certifications | 10 |
| Resume structure & readability | 10 |

- Skill matching is alias-aware (for example, "REST APIs" matches "RESTful API").
- The structure category runs five checks worth 2 points each: contact details, summary, core sections, quantified bullets, and a 150–1,800 word length band.
- Categories the engine cannot judge are excluded from the overall score instead of being guessed. If the job description states no education requirements, that category contributes nothing and is marked not applicable.
- Every deduction carries a human-readable note that appears in the on-screen results and in both export formats.

The weights are starting heuristics, not industry standards, and the score is not a prediction of how any specific ATS will parse or rank a resume. This caveat is stated in the exported reports. To change weights, edit `CATEGORY_WEIGHTS` in `src/ats_analyzer/domain/scoring_rules.py`.

## Privacy

- **Local by default.** Deterministic scoring, keyword matching, and report generation run entirely on your machine with no API key.
- **Explicit consent before any cloud call.** The application shows this disclosure before the first AI request: *"This action sends your resume and job description content to the selected provider. Verify you trust the endpoint before continuing."*
- **Your cloud calls are yours.** Content sent to a provider is subject to that provider's logging, retention, and human-review policies, which this application cannot control or verify.
- **Session-only credentials.** API keys are entered in the UI, held in server session memory, and discarded when the session ends. The application does not read keys from `.env` files or the environment, and never writes or logs them.
- **No telemetry.** No analytics and no tracking; Streamlit usage statistics are explicitly disabled in `.streamlit/config.toml`.

## Security controls

- Remote endpoints must use HTTPS. Plain `http://` is accepted only for localhost and private IP addresses.
- Credentials never appear in URLs.
- HTTP redirects are never followed.
- Transient failures (429, 5xx, connection errors) are retried a bounded number of times, honoring the `Retry-After` header.
- Response bodies are size-capped (default 2 MB) before parsing.
- Error messages shown to users are pre-composed and user-safe; raw provider responses are never rendered.
- The connection test transmits only a fixed probe sentence, never resume or job description content.

## Configuration

Optional environment overrides for HTTP client behavior (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `ATS_HTTP_TIMEOUT` | 30 | Request timeout in seconds |
| `ATS_HTTP_MAX_RETRIES` | 2 | Retry attempts for transient failures |
| `ATS_HTTP_MAX_RESPONSE_BYTES` | 2000000 | Maximum response body size in bytes |

Uploads are limited to 15 MB per document.

## Project structure

```
app.py                     Streamlit entry point
src/ats_analyzer/
  domain/                  Pydantic models, provider protocol, scoring rules
  application/             scoring, JD extraction, recommendations, report builders
  ai/                      prompt templates and LangChain Core chains
  infrastructure/          HTTPX JSON client and one provider per REST dialect
  presentation/            sidebar and dashboard rendering
tests/                     unit and integration tests (mock transport only)
.github/workflows/ci.yml   lint, type-check, and test pipeline
```

Orchestration uses LangChain Core only. All provider calls are plain HTTPX; no `openai`, `google-genai`, or `mistralai` SDK is installed. No agents, RAG, embeddings, or databases.

## Development

```bash
uv run ruff check app.py src tests
uv run ruff format app.py src tests
uv run mypy
uv run pytest
```

- Tests run entirely on fake providers and in-memory fixtures. No test sends a real request or reads a credential.
- Pre-commit hooks (ruff lint and format): `uvx pre-commit install`.
- Continuous integration runs ruff, mypy, and pytest on push and pull requests.

## Known limitations

- No OCR. A scanned or image-only resume is detected and rejected with an explanatory message instead of being transcoded.
- Endpoint validation inspects the literal URL only. Hostnames that resolve to internal IP addresses are not additionally blocked.
- The provider adapter does not support token streaming or tool calls. JSON repair and validation handle most model response quirks.
- The scoring engine is a heuristic, not an ATS simulator. See the methodology note above.

## License

Released under the [MIT License](LICENSE).
