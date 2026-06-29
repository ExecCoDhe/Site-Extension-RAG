import gc
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from app.config import Settings
from app.config import get_settings as app_get_settings
from app.crawl.models import PageRecord
from app.retrieval.vector_store import QdrantVectorStore
from app.workspace.builder import build_workspace_records
from app.workspace.models import ChildChunkRecord, PageVersionRecord, ParentSectionRecord
from app.workspace.store import WorkspaceStore
from evals.config import EVAL_WORKSPACE_ID, FIXED_RUN_ID, FIXTURES_DIR, eval_settings


def load_manifest() -> dict:
    manifest_path = FIXTURES_DIR / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def get_site_entry(site: str) -> dict:
    manifest = load_manifest()
    for entry in manifest["sites"]:
        if entry["site"] == site:
            return entry
    raise KeyError(f"unknown eval site {site!r}")


def load_pages(site: str) -> list[PageRecord]:
    pages_path = FIXTURES_DIR / site / "pages.json"
    raw_pages = json.loads(pages_path.read_text(encoding="utf-8"))
    return [PageRecord.model_validate(page) for page in raw_pages]


def build_site_chunks(
    site: str,
) -> tuple[list[PageVersionRecord], list[ParentSectionRecord], list[ChildChunkRecord]]:
    manifest = load_manifest()
    get_site_entry(site)
    pages = load_pages(site)
    fixed_run_id = manifest["fixed_run_id"]
    if fixed_run_id != FIXED_RUN_ID:
        raise ValueError(
            f"manifest fixed_run_id {fixed_run_id!r} does not match eval constant {FIXED_RUN_ID!r}"
        )
    settings = eval_settings(
        sqlite_path=":memory:",
        qdrant_path=":memory:",
        chunking_version=manifest["chunking_version"],
        embedding_model=manifest["embedding_model"],
    )
    return build_workspace_records(
        workspace_id=EVAL_WORKSPACE_ID,
        run_id=fixed_run_id,
        pages=pages,
        settings=settings,
    )


def load_doc_embeddings(site: str) -> dict[str, list[float]]:
    embeddings_path = FIXTURES_DIR / site / "embeddings.json"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"missing committed embeddings for site {site!r}: {embeddings_path}")
    return json.loads(embeddings_path.read_text(encoding="utf-8"))


def load_query_embeddings() -> dict[str, list[float]]:
    embeddings_path = FIXTURES_DIR / "query_embeddings.json"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"missing committed query embeddings: {embeddings_path}")
    return json.loads(embeddings_path.read_text(encoding="utf-8"))


class FixtureQueryEmbeddingClient:
    """Lookup committed subquery vectors; never calls the network."""

    def __init__(self, query_embeddings: dict[str, list[float]] | None = None) -> None:
        self._query_embeddings = query_embeddings or load_query_embeddings()

    def embed_query(self, text: str) -> list[float]:
        try:
            return list(self._query_embeddings[text])
        except KeyError as error:
            raise KeyError(f"no committed query embedding for subquery {text!r}") from error

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("document embeddings are pre-committed; embed_documents is not supported")


@dataclass
class EphemeralWorkspace:
    settings: Settings
    store: WorkspaceStore
    chunks: list[ChildChunkRecord]
    embeddings: dict[str, list[float]]
    query_client: FixtureQueryEmbeddingClient
    workspace_id: str
    run_id: str


@contextmanager
def ephemeral_workspace(site: str) -> Iterator[EphemeralWorkspace]:
    manifest = load_manifest()
    site_entry = get_site_entry(site)
    tmp = Path(tempfile.mkdtemp())
    settings = eval_settings(
        sqlite_path=tmp / "ws.sqlite3",
        qdrant_path=str(tmp / "qdrant"),
        chunking_version=manifest["chunking_version"],
        embedding_model=manifest["embedding_model"],
    )
    patches = [
        patch("app.db.connection.get_settings", return_value=settings),
        patch("app.config.get_settings", return_value=settings),
        patch.dict(os.environ, {"DATABASE_URL": ""}, clear=False),
    ]
    for item in patches:
        item.start()
    try:
        store = WorkspaceStore(workspace_id=EVAL_WORKSPACE_ID)
        store._initialized = False
        store.reset()

        pages, sections, chunks = build_site_chunks(site)
        run = store.start_ingest_run(
            seed_url=site_entry["seed_url"],
            hostname=site_entry["hostname"],
            registrable_domain=site_entry["registrable_domain"],
            included_subdomains=[site_entry["hostname"]],
            chunking_version=manifest["chunking_version"],
            embedding_version=manifest["embedding_model"],
        )
        if run is None:
            raise RuntimeError(f"failed to start ingest run for site {site!r}")

        for page in pages:
            page.run_id = run.run_id
        if pages and not all(page.run_id == run.run_id for page in pages):
            raise RuntimeError("page_version.run_id must match the active ingest run")

        embeddings = load_doc_embeddings(site)
        store.replace_active_content(
            run_id=run.run_id,
            pages=pages,
            sections=sections,
            chunks=chunks,
            embeddings=embeddings,
            embedding_version=manifest["embedding_model"],
            rendered_fallback_count=0,
            skipped_count=0,
        )
        QdrantVectorStore(path=settings.qdrant_path).upsert_chunks(
            collection_name=f"workspace_{EVAL_WORKSPACE_ID}",
            chunks=chunks,
            embeddings=embeddings,
        )

        yield EphemeralWorkspace(
            settings=settings,
            store=store,
            chunks=chunks,
            embeddings=embeddings,
            query_client=FixtureQueryEmbeddingClient(),
            workspace_id=EVAL_WORKSPACE_ID,
            run_id=run.run_id,
        )
    finally:
        for item in patches:
            item.stop()
        gc.collect()
        try:
            shutil.rmtree(tmp)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
        app_get_settings.cache_clear()
