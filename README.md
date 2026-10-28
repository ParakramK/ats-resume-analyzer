# ats-resume-analyzer

Privacy-first ATS (Applicant Tracking System) resume analyzer with deterministic local scoring and optional bring-your-own-key AI feedback.

- **No account, no upload, no telemetry** — everything runs on your machine until you explicitly invoke a cloud AI provider.
- **Your key, your provider** — OpenAI, Gemini, Mistral, or any OpenAI-compatible endpoint.
- **Transparent scoring** — a deterministic, explainable engine; the AI never generates your score.

Run it:

```bash
uv sync --all-groups
uv run streamlit run app.py
```

Full documentation is being written and will cover architecture, privacy trade-offs, scoring methodology, and provider configuration.
