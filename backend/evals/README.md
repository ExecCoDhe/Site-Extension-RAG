# Eval fixtures and datasets

Committed fixtures (`fixtures/`) and golden datasets (`datasets/`) support offline retrieval and QA evaluation without network calls.

## E1 — Fixtures and golden datasets

- **Fixtures:** per-site `pages.json` and committed `embeddings.json`, plus shared `query_embeddings.json`.
- **Datasets:** `datasets/retrieval.jsonl` (retrieval golden cases) and `datasets/qa.jsonl` (QA golden cases).
- **Loader:** `evals/loader.py` provides `build_site_chunks`, `load_doc_embeddings`, `FixtureQueryEmbeddingClient`, and `ephemeral_workspace` for tests.

Regenerate embeddings manually (paid Gemini calls):

```bash
cd backend
uv run python -m evals.generate_fixture_embeddings
```

## E2 — LangSmith evaluation

E2 adds deterministic offline evaluators and a manual-only LangSmith `evaluate()` runner.

### Offline CI gate (no network, no secrets)

```bash
cd backend
uv run pytest tests/test_evals_offline.py
```

The offline suite asserts these retrieval floors over the committed golden dataset (cosine + BM25, fixture query embeddings):

| Metric | Floor |
|--------|-------|
| `hit_rate` | ≥ 0.90 |
| `recall_at_k` | ≥ 0.80 |
| `mrr` | ≥ 0.60 |
| `decomposition_accuracy` | ≥ 0.80 |

`run_offline_retrieval_eval()` in `evals/langsmith_eval.py` computes the same aggregates for reuse (e.g. E3 scorecard).

### Manual online flow (paid + secrets — never run in CI)

Required environment (via `backend/.env` or shell):

- `LANGSMITH_API_KEY`
- `GEMINI_API_KEY`
- Optional: `LANGSMITH_PROJECT`, `LANGSMITH_TRACING`

**1. Upload datasets** (idempotent; use `--recreate` for a clean rebuild):

```bash
cd backend
uv run python -m evals.upload_datasets
uv run python -m evals.upload_datasets --recreate
```

Creates/syncs:

- `aura-retrieval-golden` — retrieval evaluators
- `aura-qa-golden` — faithfulness + correctness evaluators

**2. Run LangSmith experiments:**

```bash
cd backend
uv run python -m evals.langsmith_eval
```

Produces two experiments (`aura-rag-retrieval`, `aura-rag-qa`) with retrieval, faithfulness, and correctness scores. Experiment URLs are printed to stdout and feed E3's scorecard.

Online QA thresholds (reported in experiments, not CI-gated in E2):

- `faithful_rate` ≥ 0.90
- `avg_faithfulness` ≥ 0.80
- `correctness` ≥ 0.70
