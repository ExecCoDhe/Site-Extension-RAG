from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# `env_file=".env"` is relative to the process CWD, so a run from the repo root
# would read the wrong (or no) file while you edit `backend/.env`. Always load
# this package's backend root `.env`.
BACKEND_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_chat_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_CHAT_MODEL")
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001",
        alias="GEMINI_EMBEDDING_MODEL",
    )

    host: str = "127.0.0.1"
    port: int = 8000
    cors_allow_origins: list[str] = ["*"]

    sqlite_path: str = str(Path(__file__).resolve().parent.parent / ".local" / "workspace.sqlite3")
    qdrant_path: str = str(Path(__file__).resolve().parent.parent / ".local" / "qdrant")
    vector_store_backend: str = "local"

    crawl_timeout_seconds: int = 60
    max_crawl_pages: int = 20
    crawl_user_agent: str = (
        "Mozilla/5.0 (compatible; proj1-local-rag/0.1; "
        "+http://127.0.0.1:8000)"
    )
    top_k: int = 5
    retrieval_candidate_limit: int = 24
    rerank_limit: int = 8
    parent_context_limit: int = 4
    decomposition_max_subqueries: int = 3
    chunk_size_chars: int = 2_000
    chunk_overlap_chars: int = 300
    child_chunk_token_budget: int = 220
    child_chunk_token_overlap: int = 40
    chunking_version: str = "dom-heading-token-v2"


@lru_cache
def get_settings() -> Settings:
    return Settings()
