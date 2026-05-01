import sqlite3
from pathlib import Path

from app.config import Settings, get_settings


def get_connection(settings: Settings | None = None) -> sqlite3.Connection:
    settings = settings or get_settings()
    path = Path(settings.sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(settings: Settings | None = None) -> None:
    with get_connection(settings) as connection:
        connection.executescript(SCHEMA)


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
