from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.evals import run_retrieval_eval
from app.retrieval import RetrievalPipeline
from app.retrieval.vector_store import LangChainQdrantDenseSearchProvider
from evals.config import EVAL_SETTINGS_OVERRIDES, EVAL_WORKSPACE_ID, FIXTURES_DIR, eval_settings
from evals.loader import (
    build_site_chunks,
    ephemeral_workspace,
    load_doc_embeddings,
    load_manifest,
    load_pages,
    load_query_embeddings,
)
from evals.schema import load_qa_dataset, load_retrieval_dataset

SITES = ["acme_docs", "nimbus_api", "orchard_co"]
EXPECTED_CHUNK_COUNTS = {
    "acme_docs": 7,
    "nimbus_api": 8,
    "orchard_co": 7,
}


def _eval_settings() -> Settings:
    manifest = load_manifest()
    return eval_settings(
        sqlite_path=":memory:",
        qdrant_path=":memory:",
        chunking_version=manifest["chunking_version"],
        embedding_model=manifest["embedding_model"],
    )


def _subqueries_for_question(question: str) -> list[str]:
    settings = _eval_settings()

    class _NoopEmbeddingClient:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise NotImplementedError

        def embed_query(self, text: str) -> list[float]:
            return [0.0]

    plan = RetrievalPipeline(
        settings=settings,
        embedding_client=_NoopEmbeddingClient(),
        chunks=[],
        embeddings={},
    )._query_plan(question)
    return plan.subqueries


@pytest.mark.parametrize("site", SITES)
def test_build_site_chunks_is_deterministic(site: str) -> None:
    first_ids = {chunk.chunk_id for chunk in build_site_chunks(site)[2]}
    second_ids = {chunk.chunk_id for chunk in build_site_chunks(site)[2]}
    assert first_ids == second_ids


@pytest.mark.parametrize("site", SITES)
def test_committed_doc_embeddings_match_chunk_ids(site: str) -> None:
    chunk_ids = {chunk.chunk_id for chunk in build_site_chunks(site)[2]}
    embedding_ids = set(load_doc_embeddings(site).keys())
    assert embedding_ids == chunk_ids


def test_manifest_matches_settings_defaults() -> None:
    manifest = load_manifest()
    defaults = Settings()
    assert manifest["chunking_version"] == defaults.chunking_version
    assert manifest["embedding_model"] == defaults.gemini_embedding_model
    assert manifest["embedding_dim"] == 3072
    assert manifest["fixed_run_id"] == "eval-fixed-run-v1"


@pytest.mark.parametrize("site", SITES)
def test_each_site_has_expected_chunk_count(site: str) -> None:
    chunks = build_site_chunks(site)[2]
    assert len(chunks) == EXPECTED_CHUNK_COUNTS[site]


@pytest.mark.parametrize("site", SITES)
def test_pages_json_has_required_fields(site: str) -> None:
    for page in load_pages(site):
        assert page.url
        assert page.canonical_url
        assert page.content_hash
        assert page.title
        assert page.heading_paths
        assert page.clean_text


@pytest.mark.parametrize("site", SITES)
def test_each_site_has_multi_chunk_section(site: str) -> None:
    chunks = build_site_chunks(site)[2]
    section_counts: dict[str, int] = {}
    for chunk in chunks:
        section_counts[chunk.section_id] = section_counts.get(chunk.section_id, 0) + 1
    assert any(count >= 2 for count in section_counts.values())


@pytest.mark.parametrize("site", SITES)
def test_doc_embedding_vectors_are_3072(site: str) -> None:
    manifest = load_manifest()
    for vector in load_doc_embeddings(site).values():
        assert len(vector) == manifest["embedding_dim"]


def test_query_embedding_vectors_are_3072() -> None:
    manifest = load_manifest()
    for vector in load_query_embeddings().values():
        assert len(vector) == manifest["embedding_dim"]


def test_retrieval_dataset_counts_and_schema() -> None:
    cases = load_retrieval_dataset()
    assert len(cases) >= 10
    assert sum(case.should_decompose for case in cases) >= 3


def test_qa_dataset_counts_and_schema() -> None:
    cases = load_qa_dataset()
    assert len(cases) >= 10
    assert any(case.expected_groundedness == "not_grounded" for case in cases)


@pytest.mark.parametrize("site", SITES)
def test_retrieval_expected_chunk_ids_exist(site: str) -> None:
    chunk_ids = {chunk.chunk_id for chunk in build_site_chunks(site)[2]}
    page_urls = {page.url for page in load_pages(site)}
    for case in load_retrieval_dataset():
        if case.site != site:
            continue
        for chunk_id in case.expected_chunk_ids + case.equivalent_chunk_ids:
            assert chunk_id in chunk_ids
        for url in case.expected_urls:
            assert url in page_urls


@pytest.mark.parametrize("site", SITES)
def test_qa_expected_chunk_ids_and_urls_exist(site: str) -> None:
    chunk_ids = {chunk.chunk_id for chunk in build_site_chunks(site)[2]}
    page_urls = {page.url for page in load_pages(site)}
    for case in load_qa_dataset():
        if case.site != site:
            continue
        for chunk_id in case.expected_chunk_ids:
            assert chunk_id in chunk_ids
        for url in case.expected_urls:
            assert url in page_urls


def test_query_embeddings_cover_dataset_subqueries() -> None:
    query_embeddings = load_query_embeddings()
    for case in load_retrieval_dataset():
        for subquery in _subqueries_for_question(case.question):
            assert subquery in query_embeddings
    for case in load_qa_dataset():
        for subquery in _subqueries_for_question(case.question):
            assert subquery in query_embeddings


def test_should_decompose_cases_really_decompose() -> None:
    settings = _eval_settings()

    class _NoopEmbeddingClient:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise NotImplementedError

        def embed_query(self, text: str) -> list[float]:
            return [0.0]

    pipeline = RetrievalPipeline(
        settings=settings,
        embedding_client=_NoopEmbeddingClient(),
        chunks=[],
        embeddings={},
    )
    for case in load_retrieval_dataset():
        if not case.should_decompose:
            continue
        plan = pipeline._query_plan(case.question)
        assert plan.decomposed is True


@pytest.mark.parametrize("site", SITES)
def test_ephemeral_workspace_seeds_sqlite_and_qdrant(site: str) -> None:
    default_sqlite = Path(get_settings().sqlite_path)
    default_qdrant = Path(get_settings().qdrant_path)
    default_sqlite_mtime = default_sqlite.stat().st_mtime if default_sqlite.exists() else None
    default_qdrant_exists = default_qdrant.exists()

    with ephemeral_workspace(site) as handle:
        assert len(handle.store.active_chunks()) == len(handle.chunks)
        from qdrant_client import QdrantClient

        client = QdrantClient(path=handle.settings.qdrant_path)
        try:
            collection = f"workspace_{EVAL_WORKSPACE_ID}"
            assert client.collection_exists(collection)
            info = client.get_collection(collection)
            assert info.points_count == len(handle.chunks)
        finally:
            client.close()

    if default_sqlite_mtime is not None:
        assert default_sqlite.stat().st_mtime == default_sqlite_mtime
    assert default_qdrant.exists() == default_qdrant_exists


def test_ephemeral_workspace_removes_temp_directory() -> None:
    temp_dir: Path | None = None
    with ephemeral_workspace("acme_docs") as handle:
        temp_dir = Path(handle.settings.sqlite_path).parent
        assert temp_dir.exists()

    assert temp_dir is not None
    assert not temp_dir.exists()


def test_ephemeral_workspace_uses_sqlite_when_database_url_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/evals")
    temp_dir: Path | None = None
    with ephemeral_workspace("acme_docs") as handle:
        temp_dir = Path(handle.settings.sqlite_path).parent
        assert Path(handle.settings.sqlite_path).exists()
        assert len(handle.store.active_chunks()) == len(handle.chunks)

    assert temp_dir is not None
    assert not temp_dir.exists()


def test_ephemeral_workspace_qdrant_dense_round_trip() -> None:
    case = next(case for case in load_retrieval_dataset() if case.site == "acme_docs" and case.expected_chunk_ids)
    subquery = _subqueries_for_question(case.question)[0]
    query_vector = load_query_embeddings()[subquery]
    expected_chunk_id = case.expected_chunk_ids[0]

    with ephemeral_workspace(case.site) as handle:
        provider = LangChainQdrantDenseSearchProvider(
            path=handle.settings.qdrant_path,
            collection_name=f"workspace_{EVAL_WORKSPACE_ID}",
        )
        scores = provider.search_scores(query_vector, limit=5)
        assert scores is not None
        assert expected_chunk_id in scores


@pytest.mark.parametrize(
    ("site", "question"),
    [
        ("acme_docs", "How do I set up the Acme desktop agent?"),
        ("nimbus_api", "How do I authenticate to the Nimbus API?"),
    ],
)
def test_offline_retrieval_eval_membership(site: str, question: str) -> None:
    case = next(case for case in load_retrieval_dataset() if case.site == site and case.question == question)
    with ephemeral_workspace(site) as handle:
        metrics = run_retrieval_eval(
            cases=[case.to_eval_case()],
            settings=handle.settings,
            embedding_client=handle.query_client,
            chunks=handle.chunks,
            embeddings=handle.embeddings,
        )
    result = metrics["results"][0]
    if case.expected_chunk_ids:
        assert any(chunk_id in result["retrieved_chunk_ids"] for chunk_id in case.expected_chunk_ids)
    if case.expected_urls:
        assert any(url in result["retrieved_urls"] for url in case.expected_urls)


def test_eval_settings_override_rerank_limit() -> None:
    manifest = load_manifest()
    settings = eval_settings(
        sqlite_path=":memory:",
        qdrant_path=":memory:",
        chunking_version=manifest["chunking_version"],
        embedding_model=manifest["embedding_model"],
    )
    assert settings.rerank_limit == EVAL_SETTINGS_OVERRIDES["rerank_limit"]
    assert settings.top_k == EVAL_SETTINGS_OVERRIDES["top_k"]
    assert settings.child_chunk_token_budget == EVAL_SETTINGS_OVERRIDES["child_chunk_token_budget"]
    assert settings.child_chunk_token_overlap == EVAL_SETTINGS_OVERRIDES["child_chunk_token_overlap"]


def test_fixtures_manifest_lists_all_sites() -> None:
    manifest = load_manifest()
    fixture_dirs = {path.name for path in FIXTURES_DIR.iterdir() if path.is_dir()}
    listed = {entry["site"] for entry in manifest["sites"]}
    assert listed <= fixture_dirs
