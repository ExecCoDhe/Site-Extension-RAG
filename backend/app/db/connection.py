import os
import sqlite3
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings

_pg_pool = None

_VALID_TABLES = frozenset({
    "eval_run",
    "eval_case",
    "chat_trace",
    "session_memory",
    "embedding_record",
    "child_chunk",
    "parent_section",
    "page_version",
    "ingest_run",
    "workspace",
})


class PostgresConnection:
    def __init__(self, connection, *, _owned_by_pool: bool = False) -> None:
        self._connection = connection
        self._owned_by_pool = _owned_by_pool

    def __enter__(self) -> "PostgresConnection":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        try:
            if exc_type is None:
                self._connection.commit()
            else:
                self._connection.rollback()
        finally:
            if self._owned_by_pool:
                _pg_pool.putconn(self._connection)
            else:
                self._connection.close()

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        return self._connection.execute(_postgres_sql(sql), parameters)

    def executescript(self, script: str) -> None:
        for statement in _split_sql_statements(script):
            self.execute(statement)


def _get_pg_pool(database_url: str):
    """Return a lazily-created connection pool singleton."""
    global _pg_pool
    if _pg_pool is None:
        import logging

        import psycopg
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        logger = logging.getLogger(__name__)

        # Verify connectivity with a direct connection first so we get
        # a clear error (wrong password, unreachable host, bad socket)
        # instead of a generic 30-second pool timeout.
        try:
            logger.info("Testing database connection: %s", database_url.split("@")[0] + "@...")
            test_conn = psycopg.connect(database_url, connect_timeout=10, row_factory=dict_row)
            test_conn.close()
            logger.info("Database connection test passed.")
        except Exception:
            logger.exception("Database connection test FAILED.")
            raise

        _pg_pool = ConnectionPool(
            database_url,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row, "connect_timeout": 10},
            open=True,
        )
    return _pg_pool


def get_connection(settings: Settings | None = None) -> sqlite3.Connection | PostgresConnection:
    settings = settings or get_settings()
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        pool = _get_pg_pool(database_url)
        conn = pool.getconn(timeout=15)
        return PostgresConnection(conn, _owned_by_pool=True)

    path = Path(settings.sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def close_pool() -> None:
    """Shut down the Postgres connection pool, if one was created."""
    global _pg_pool
    if _pg_pool is not None:
        _pg_pool.close()
        _pg_pool = None


def initialize_database(settings: Settings | None = None) -> None:
    with get_connection(settings) as connection:
        connection.executescript(SCHEMA)


def validate_table_name(table: str) -> str:
    """Validate that a table name is in the known allowlist."""
    if table not in _VALID_TABLES:
        raise ValueError(f"Unknown table name: {table!r}")
    return table


def _postgres_sql(sql: str) -> str:
    translated = sql
    stripped = translated.lstrip()
    if stripped.startswith("INSERT OR REPLACE INTO embedding_record"):
        translated = translated.replace("INSERT OR REPLACE INTO", "INSERT INTO", 1)
        translated += """
        ON CONFLICT (chunk_id, embedding_version) DO UPDATE SET
          vector_json = EXCLUDED.vector_json,
          created_at = EXCLUDED.created_at
        """
    elif stripped.startswith("INSERT OR REPLACE INTO session_memory"):
        translated = translated.replace("INSERT OR REPLACE INTO", "INSERT INTO", 1)
        translated += """
        ON CONFLICT (session_id) DO UPDATE SET
          workspace_id = EXCLUDED.workspace_id,
          summary_json = EXCLUDED.summary_json,
          updated_at = EXCLUDED.updated_at
        """

    return translated.replace("?", "%s")


def _split_sql_statements(script: str) -> list[str]:
    """Split a SQL script into individual statements.

    This is a simple splitter that respects single-quoted string literals
    containing semicolons. It is NOT a full SQL parser, but handles the
    common case of CREATE TABLE statements with default values.
    """
    statements: list[str] = []
    current: list[str] = []
    in_string = False

    for char in script:
        if char == "'" and not in_string:
            in_string = True
            current.append(char)
        elif char == "'" and in_string:
            in_string = False
            current.append(char)
        elif char == ";" and not in_string:
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(char)

    # Handle trailing statement without semicolon
    stmt = "".join(current).strip()
    if stmt:
        statements.append(stmt)

    return statements


SCHEMA = """
CREATE TABLE IF NOT EXISTS workspace (
  workspace_id TEXT PRIMARY KEY,
  hostname TEXT,
  registrable_domain TEXT,
  include_subdomains TEXT NOT NULL DEFAULT '[]',
  state TEXT NOT NULL,
  active_run_id TEXT,
  active_chunking_version TEXT,
  active_embedding_version TEXT,
  last_synced_at TEXT,
  page_count INTEGER NOT NULL DEFAULT 0,
  chunk_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_run (
  run_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  state TEXT NOT NULL,
  seed_url TEXT NOT NULL,
  hostname TEXT NOT NULL,
  registrable_domain TEXT NOT NULL,
  page_count INTEGER NOT NULL DEFAULT 0,
  chunk_count INTEGER NOT NULL DEFAULT 0,
  fetched_count INTEGER NOT NULL DEFAULT 0,
  skipped_count INTEGER NOT NULL DEFAULT 0,
  rendered_fallback_count INTEGER NOT NULL DEFAULT 0,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  error_code TEXT,
  error_message TEXT,
  FOREIGN KEY(workspace_id) REFERENCES workspace(workspace_id)
);

CREATE TABLE IF NOT EXISTS page_version (
  page_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  canonical_url TEXT NOT NULL,
  discovered_url TEXT NOT NULL,
  title TEXT NOT NULL,
  acquisition_method TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  quality_score REAL NOT NULL,
  quality_signals TEXT NOT NULL,
  boilerplate_removed TEXT NOT NULL,
  clean_text TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1,
  is_stale INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  FOREIGN KEY(workspace_id) REFERENCES workspace(workspace_id),
  FOREIGN KEY(run_id) REFERENCES ingest_run(run_id)
);

CREATE INDEX IF NOT EXISTS idx_page_workspace_active
ON page_version(workspace_id, is_active, is_stale);

CREATE TABLE IF NOT EXISTS parent_section (
  section_id TEXT PRIMARY KEY,
  page_id TEXT NOT NULL,
  workspace_id TEXT NOT NULL,
  heading_path TEXT NOT NULL,
  section_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  start_offset INTEGER NOT NULL,
  end_offset INTEGER NOT NULL,
  FOREIGN KEY(page_id) REFERENCES page_version(page_id)
);

CREATE TABLE IF NOT EXISTS child_chunk (
  chunk_id TEXT PRIMARY KEY,
  section_id TEXT NOT NULL,
  page_id TEXT NOT NULL,
  workspace_id TEXT NOT NULL,
  chunking_version TEXT NOT NULL,
  title TEXT NOT NULL,
  url TEXT NOT NULL,
  heading_path TEXT NOT NULL,
  text TEXT NOT NULL,
  token_start INTEGER NOT NULL,
  token_end INTEGER NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1,
  FOREIGN KEY(section_id) REFERENCES parent_section(section_id),
  FOREIGN KEY(page_id) REFERENCES page_version(page_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_workspace_active
ON child_chunk(workspace_id, is_active, chunking_version);

CREATE TABLE IF NOT EXISTS embedding_record (
  chunk_id TEXT NOT NULL,
  embedding_version TEXT NOT NULL,
  vector_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(chunk_id, embedding_version),
  FOREIGN KEY(chunk_id) REFERENCES child_chunk(chunk_id)
);

CREATE TABLE IF NOT EXISTS session_memory (
  session_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_trace (
  trace_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  question TEXT NOT NULL,
  rewritten_question TEXT NOT NULL,
  decomposition_json TEXT NOT NULL,
  candidates_json TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  groundedness TEXT NOT NULL,
  latency_ms INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_case (
  eval_case_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  question TEXT NOT NULL,
  expected_evidence_json TEXT NOT NULL,
  should_rewrite INTEGER NOT NULL DEFAULT 0,
  should_decompose INTEGER NOT NULL DEFAULT 0,
  expected_answer TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_run (
  eval_run_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  config_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""
