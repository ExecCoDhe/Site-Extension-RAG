from pathlib import Path

from app.config import Settings

FIXED_RUN_ID = "eval-fixed-run-v1"
EVAL_WORKSPACE_ID = "default"

_DEFAULTS = Settings()

EVAL_SETTINGS_OVERRIDES: dict[str, int] = {
    "top_k": 5,
    "rerank_limit": 5,
    "retrieval_candidate_limit": 24,
    "parent_context_limit": 4,
    "decomposition_max_subqueries": 3,
    "child_chunk_token_budget": _DEFAULTS.child_chunk_token_budget,
    "child_chunk_token_overlap": _DEFAULTS.child_chunk_token_overlap,
}

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


def eval_settings(
    *,
    sqlite_path: str | Path,
    qdrant_path: str | Path,
    chunking_version: str,
    embedding_model: str,
) -> Settings:
    return Settings(
        sqlite_path=str(sqlite_path),
        qdrant_path=str(qdrant_path),
        chunking_version=chunking_version,
        gemini_embedding_model=embedding_model,
        **EVAL_SETTINGS_OVERRIDES,
    )
