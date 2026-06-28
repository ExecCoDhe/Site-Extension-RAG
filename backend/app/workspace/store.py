import json
from datetime import UTC, datetime
from uuid import uuid4

from app.api.errors import ErrorCode
from app.db import get_connection, initialize_database, validate_table_name
from app.workspace.models import (
    ChildChunkRecord,
    IngestRunSummary,
    PageVersionRecord,
    ParentSectionRecord,
    RunState,
    WorkspaceState,
    WorkspaceSummary,
)

DEFAULT_WORKSPACE_ID = "default"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class WorkspaceStore:
    def __init__(self, workspace_id: str = DEFAULT_WORKSPACE_ID) -> None:
        self.workspace_id = workspace_id
        self._initialized = False
        self._initializing = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize database tables and default workspace row.

        Uses an ``_initializing`` guard to prevent infinite recursion when
        public methods (``ensure_workspace`` → ``get_workspace``) call back
        into ``_ensure_initialized``.
        """
        if self._initialized or self._initializing:
            return
        self._initializing = True
        try:
            initialize_database()
            self.ensure_workspace()
            self._initialized = True
        finally:
            self._initializing = False

    def reset(self) -> None:
        self._ensure_initialized()
        with get_connection() as connection:
            for table in [
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
            ]:
                # validate_table_name guards against SQL injection even though
                # the list above is hardcoded — defence in depth.
                connection.execute(f"DELETE FROM {validate_table_name(table)}")
        self.ensure_workspace()

    def ensure_workspace(self) -> WorkspaceSummary:
        self._ensure_initialized()
        existing = self.get_workspace()
        if existing is not None:
            return existing

        now = utc_now()
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO workspace (
                  workspace_id, state, include_subdomains, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (self.workspace_id, WorkspaceState.IDLE.value, "[]", now, now),
            )
        workspace = self.get_workspace()
        assert workspace is not None
        return workspace

    def get_workspace(self) -> WorkspaceSummary | None:
        self._ensure_initialized()
        with get_connection() as connection:
            row = connection.execute(
                "SELECT * FROM workspace WHERE workspace_id = ?",
                (self.workspace_id,),
            ).fetchone()
        if row is None:
            return None
        return WorkspaceSummary(
            workspace_id=row["workspace_id"],
            state=WorkspaceState(row["state"]),
            hostname=row["hostname"],
            registrable_domain=row["registrable_domain"],
            included_subdomains=json.loads(row["include_subdomains"]),
            active_run_id=row["active_run_id"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            last_synced_at=row["last_synced_at"],
            active_chunking_version=row["active_chunking_version"],
            active_embedding_version=row["active_embedding_version"],
        )

    def active_ingest_run(self) -> IngestRunSummary | None:
        self._ensure_initialized()
        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM ingest_run
                WHERE workspace_id = ? AND state = ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (self.workspace_id, RunState.INGESTING.value),
            ).fetchone()
        return _run_from_row(row) if row else None

    def recover_interrupted_ingest_runs(self) -> int:
        """Mark ingest runs that could not survive a backend restart as retryable errors."""
        self._ensure_initialized()
        now = utc_now()
        with get_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE ingest_run
                SET state = ?, completed_at = ?, error_code = ?, error_message = ?
                WHERE workspace_id = ? AND state = ?
                """,
                (
                    RunState.ERROR.value,
                    now,
                    "INGEST_INTERRUPTED",
                    "Previous ingest was interrupted before it could finish. Start ingest again.",
                    self.workspace_id,
                    RunState.INGESTING.value,
                ),
            )
            recovered_count = cursor.rowcount
            if recovered_count:
                connection.execute(
                    """
                    UPDATE workspace
                    SET state = ?, updated_at = ?
                    WHERE workspace_id = ? AND state = ?
                    """,
                    (
                        WorkspaceState.ERROR.value,
                        now,
                        self.workspace_id,
                        WorkspaceState.INGESTING.value,
                    ),
                )
        return recovered_count

    def start_ingest_run(
        self,
        *,
        seed_url: str,
        hostname: str,
        registrable_domain: str,
        included_subdomains: list[str],
        chunking_version: str,
        embedding_version: str,
    ) -> IngestRunSummary | None:
        self._ensure_initialized()
        if self.active_ingest_run() is not None:
            return None

        run_id = f"run_{uuid4().hex}"
        now = utc_now()
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO ingest_run (
                  run_id, workspace_id, state, seed_url, hostname, registrable_domain,
                  started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    self.workspace_id,
                    RunState.INGESTING.value,
                    seed_url,
                    hostname,
                    registrable_domain,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE workspace
                SET state = ?, hostname = ?, registrable_domain = ?, include_subdomains = ?,
                    active_run_id = ?, active_chunking_version = ?,
                    active_embedding_version = ?, updated_at = ?
                WHERE workspace_id = ?
                """,
                (
                    WorkspaceState.INGESTING.value,
                    hostname,
                    registrable_domain,
                    json.dumps(included_subdomains),
                    run_id,
                    chunking_version,
                    embedding_version,
                    now,
                    self.workspace_id,
                ),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> IngestRunSummary | None:
        self._ensure_initialized()
        normalized_run_id = self.normalize_run_id(run_id)
        with get_connection() as connection:
            row = connection.execute(
                "SELECT * FROM ingest_run WHERE run_id = ?",
                (normalized_run_id,),
            ).fetchone()
        return _run_from_row(row) if row else None

    def normalize_run_id(self, run_id: str) -> str:
        if run_id.startswith(("run_", "job_")):
            return run_id.replace("job_", "run_", 1)
        with get_connection() as connection:
            row = connection.execute(
                "SELECT run_id FROM ingest_run WHERE run_id = ?",
                (f"run_{run_id}",),
            ).fetchone()
        if row is not None:
            return f"run_{run_id}"
        return run_id

    def replace_active_content(
        self,
        *,
        run_id: str,
        pages: list[PageVersionRecord],
        sections: list[ParentSectionRecord],
        chunks: list[ChildChunkRecord],
        embeddings: dict[str, list[float]],
        embedding_version: str,
        rendered_fallback_count: int,
        skipped_count: int,
        retained_canonical_urls: list[str] | None = None,
    ) -> None:
        self._ensure_initialized()
        now = utc_now()
        retained_canonical_urls = retained_canonical_urls or []
        with get_connection() as connection:
            if retained_canonical_urls:
                placeholders = ", ".join("?" for _url in retained_canonical_urls)
                connection.execute(
                    f"""
                    UPDATE page_version
                    SET is_active = 0, is_stale = 1
                    WHERE workspace_id = ? AND is_active = 1
                      AND canonical_url NOT IN ({placeholders})
                    """,
                    (self.workspace_id, *retained_canonical_urls),
                )
            else:
                connection.execute(
                    """
                    UPDATE page_version
                    SET is_active = 0, is_stale = 1
                    WHERE workspace_id = ? AND is_active = 1
                    """,
                    (self.workspace_id,),
                )
            connection.execute(
                """
                UPDATE child_chunk
                SET is_active = 0
                WHERE workspace_id = ? AND is_active = 1
                  AND page_id NOT IN (
                    SELECT page_id FROM page_version
                    WHERE workspace_id = ? AND is_active = 1
                  )
                """,
                (self.workspace_id, self.workspace_id),
            )

            for page in pages:
                connection.execute(
                    """
                    INSERT INTO page_version (
                      page_id, workspace_id, run_id, canonical_url, discovered_url,
                      title, acquisition_method, content_hash, quality_score,
                      quality_signals, boilerplate_removed, clean_text, is_active,
                      is_stale, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        page.page_id,
                        page.workspace_id,
                        page.run_id,
                        page.canonical_url,
                        page.discovered_url,
                        page.title,
                        page.acquisition_method.value,
                        page.content_hash,
                        page.quality_score,
                        json.dumps(page.quality_signals),
                        json.dumps(page.boilerplate_removed),
                        page.clean_text,
                        int(page.is_active),
                        int(page.is_stale),
                        now,
                    ),
                )

            for section in sections:
                connection.execute(
                    """
                    INSERT INTO parent_section (
                      section_id, page_id, workspace_id, heading_path, section_index,
                      text, start_offset, end_offset
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        section.section_id,
                        section.page_id,
                        section.workspace_id,
                        json.dumps(section.heading_path),
                        section.section_index,
                        section.text,
                        section.start_offset,
                        section.end_offset,
                    ),
                )

            for chunk in chunks:
                connection.execute(
                    """
                    INSERT INTO child_chunk (
                      chunk_id, section_id, page_id, workspace_id, chunking_version,
                      title, url, heading_path, text, token_start, token_end, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.section_id,
                        chunk.page_id,
                        chunk.workspace_id,
                        chunk.chunking_version,
                        chunk.title,
                        chunk.url,
                        json.dumps(chunk.heading_path),
                        chunk.text,
                        chunk.token_start,
                        chunk.token_end,
                        int(chunk.is_active),
                    ),
                )
                vector = embeddings.get(chunk.chunk_id)
                if vector is not None:
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO embedding_record (
                          chunk_id, embedding_version, vector_json, created_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (chunk.chunk_id, embedding_version, json.dumps(vector), now),
                    )

            active_counts = connection.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM page_version WHERE workspace_id = ? AND is_active = 1) AS page_count,
                  (SELECT COUNT(*) FROM child_chunk WHERE workspace_id = ? AND is_active = 1) AS chunk_count
                """,
                (self.workspace_id, self.workspace_id),
            ).fetchone()
            page_count = int(active_counts["page_count"])
            chunk_count = int(active_counts["chunk_count"])

            connection.execute(
                """
                UPDATE ingest_run
                SET state = ?, page_count = ?, chunk_count = ?, fetched_count = ?,
                    skipped_count = ?, rendered_fallback_count = ?, completed_at = ?
                WHERE run_id = ?
                """,
                (
                    RunState.READY.value,
                    page_count,
                    chunk_count,
                    len(pages) + skipped_count,
                    skipped_count,
                    rendered_fallback_count,
                    now,
                    run_id,
                ),
            )
            connection.execute(
                """
                UPDATE workspace
                SET state = ?, page_count = ?, chunk_count = ?, last_synced_at = ?,
                    updated_at = ?
                WHERE workspace_id = ?
                """,
                (
                    WorkspaceState.READY.value,
                    page_count,
                    chunk_count,
                    now,
                    now,
                    self.workspace_id,
                ),
            )

    def active_page_hashes(self) -> dict[str, str]:
        self._ensure_initialized()
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT canonical_url, content_hash
                FROM page_version
                WHERE workspace_id = ? AND is_active = 1
                """,
                (self.workspace_id,),
            ).fetchall()
        return {row["canonical_url"]: row["content_hash"] for row in rows}

    def fail_run(
        self,
        run_id: str,
        *,
        code: ErrorCode,
        message: str,
    ) -> None:
        self._ensure_initialized()
        now = utc_now()
        with get_connection() as connection:
            connection.execute(
                """
                UPDATE ingest_run
                SET state = ?, completed_at = ?, error_code = ?, error_message = ?
                WHERE run_id = ?
                """,
                (RunState.ERROR.value, now, code, message, run_id),
            )
            connection.execute(
                """
                UPDATE workspace
                SET state = ?, updated_at = ?
                WHERE workspace_id = ?
                """,
                (WorkspaceState.ERROR.value, now, self.workspace_id),
            )

    def active_chunks(self) -> list[ChildChunkRecord]:
        self._ensure_initialized()
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM child_chunk
                WHERE workspace_id = ? AND is_active = 1
                ORDER BY chunk_id
                """,
                (self.workspace_id,),
            ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def embeddings(self, embedding_version: str) -> dict[str, list[float]]:
        self._ensure_initialized()
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, vector_json FROM embedding_record
                WHERE embedding_version = ?
                """,
                (embedding_version,),
            ).fetchall()
        return {row["chunk_id"]: json.loads(row["vector_json"]) for row in rows}

    def session_memory(self, session_id: str | None) -> dict[str, object]:
        if not session_id:
            return {}
        self._ensure_initialized()
        with get_connection() as connection:
            row = connection.execute(
                "SELECT summary_json FROM session_memory WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return json.loads(row["summary_json"]) if row else {}

    def update_session_memory(
        self,
        *,
        session_id: str | None,
        question: str,
        answer: str,
        citations: list[dict[str, object]],
    ) -> None:
        if not session_id:
            return
        self._ensure_initialized()
        topic = citations[0].get("title") if citations else None
        summary = {
            "last_question": question,
            "last_answer": answer[:500],
            "last_topic": topic,
            "last_citation_count": len(citations),
        }
        with get_connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO session_memory (
                  session_id, workspace_id, summary_json, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                (session_id, self.workspace_id, json.dumps(summary), utc_now()),
            )

    def save_chat_trace(
        self,
        *,
        question: str,
        rewritten_question: str,
        decomposition: dict[str, object],
        candidates: list[dict[str, object]],
        evidence: list[dict[str, object]],
        groundedness: str,
        latency_ms: int,
    ) -> str:
        self._ensure_initialized()
        trace_id = f"trace_{uuid4().hex}"
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO chat_trace (
                  trace_id, workspace_id, question, rewritten_question,
                  decomposition_json, candidates_json, evidence_json,
                  groundedness, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    self.workspace_id,
                    question,
                    rewritten_question,
                    json.dumps(decomposition),
                    json.dumps(candidates),
                    json.dumps(evidence),
                    groundedness,
                    latency_ms,
                    utc_now(),
                ),
            )
        return trace_id


def _run_from_row(row) -> IngestRunSummary:
    error = None
    if row["error_code"]:
        error = {
            "code": row["error_code"],
            "message": row["error_message"],
            "details": None,
            "retryable": True,
        }
    return IngestRunSummary(
        run_id=row["run_id"],
        workspace_id=row["workspace_id"],
        state=RunState(row["state"]),
        seed_url=row["seed_url"],
        hostname=row["hostname"],
        registrable_domain=row["registrable_domain"],
        page_count=row["page_count"],
        chunk_count=row["chunk_count"],
        fetched_count=row["fetched_count"],
        skipped_count=row["skipped_count"],
        rendered_fallback_count=row["rendered_fallback_count"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        error=error,
    )


def _chunk_from_row(row) -> ChildChunkRecord:
    return ChildChunkRecord(
        chunk_id=row["chunk_id"],
        section_id=row["section_id"],
        page_id=row["page_id"],
        workspace_id=row["workspace_id"],
        chunking_version=row["chunking_version"],
        title=row["title"],
        url=row["url"],
        heading_path=json.loads(row["heading_path"]),
        text=row["text"],
        token_start=row["token_start"],
        token_end=row["token_end"],
        is_active=bool(row["is_active"]),
    )


workspace_store = WorkspaceStore()
