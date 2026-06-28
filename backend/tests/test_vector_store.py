from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from app.retrieval.vector_store import (
    LangChainQdrantDenseSearchProvider,
    QdrantVectorStore,
)
from app.workspace.models import ChildChunkRecord

COLLECTION = "workspace_ws-test"


def _make_chunk(*, suffix: str) -> ChildChunkRecord:
    chunk_id = str(uuid5(NAMESPACE_URL, f"test-chunk:{suffix}"))
    return ChildChunkRecord(
        chunk_id=chunk_id,
        section_id=f"section-{suffix}",
        page_id=f"page-{suffix}",
        workspace_id="ws-test",
        chunking_version="test",
        title=f"Title {suffix}",
        url=f"https://example.com/{suffix}",
        heading_path=["H1", f"H2-{suffix}"],
        text=f"body text {suffix}",
        token_start=0,
        token_end=10,
    )


def test_upsert_then_dense_search_returns_seeded_chunk(tmp_path) -> None:
    chunk_a = _make_chunk(suffix="a")
    chunk_b = _make_chunk(suffix="b")
    embeddings = {
        chunk_a.chunk_id: [1.0, 0.0, 0.0],
        chunk_b.chunk_id: [0.0, 1.0, 0.0],
    }

    QdrantVectorStore(path=str(tmp_path)).upsert_chunks(
        collection_name=COLLECTION,
        chunks=[chunk_a, chunk_b],
        embeddings=embeddings,
    )

    scores = LangChainQdrantDenseSearchProvider(
        path=str(tmp_path),
        collection_name=COLLECTION,
    ).search_scores([1.0, 0.0, 0.0], limit=5)

    assert scores is not None
    assert chunk_a.chunk_id in scores
    assert chunk_b.chunk_id in scores
    assert isinstance(scores[chunk_a.chunk_id], float)
    assert scores[chunk_a.chunk_id] > scores[chunk_b.chunk_id]


def test_missing_collection_returns_none(tmp_path) -> None:
    scores = LangChainQdrantDenseSearchProvider(
        path=str(tmp_path),
        collection_name="workspace_missing",
    ).search_scores([1.0, 0.0, 0.0], limit=5)

    assert scores is None


def test_upsert_payload_round_trips_full_fields(tmp_path) -> None:
    chunk = _make_chunk(suffix="payload")
    embeddings = {chunk.chunk_id: [1.0, 0.0, 0.0]}

    QdrantVectorStore(path=str(tmp_path)).upsert_chunks(
        collection_name=COLLECTION,
        chunks=[chunk],
        embeddings=embeddings,
    )

    client = QdrantClient(path=str(tmp_path))
    try:
        points = client.retrieve(
            collection_name=COLLECTION,
            ids=[chunk.chunk_id],
            with_payload=True,
        )
    finally:
        client.close()

    assert len(points) == 1
    payload = points[0].payload or {}
    assert payload["chunk_id"] == chunk.chunk_id
    assert payload["section_id"] == chunk.section_id
    assert payload["page_id"] == chunk.page_id
    assert payload["workspace_id"] == chunk.workspace_id
    assert payload["url"] == chunk.url
    assert payload["title"] == chunk.title
    assert payload["heading_path"] == chunk.heading_path
    assert payload["token_start"] == chunk.token_start
    assert payload["token_end"] == chunk.token_end


def test_upsert_creates_collection_with_discovered_dimension(tmp_path) -> None:
    chunk = _make_chunk(suffix="dim")
    embeddings = {chunk.chunk_id: [1.0, 0.0, 0.0]}

    QdrantVectorStore(path=str(tmp_path)).upsert_chunks(
        collection_name=COLLECTION,
        chunks=[chunk],
        embeddings=embeddings,
    )

    client = QdrantClient(path=str(tmp_path))
    try:
        info = client.get_collection(COLLECTION)
    finally:
        client.close()

    assert info.config.params.vectors.size == 3
    assert info.config.params.vectors.distance == Distance.COSINE


def test_langchain_dense_search_exception_returns_none(tmp_path, monkeypatch) -> None:
    chunk = _make_chunk(suffix="err")
    embeddings = {chunk.chunk_id: [1.0, 0.0, 0.0]}
    QdrantVectorStore(path=str(tmp_path)).upsert_chunks(
        collection_name=COLLECTION,
        chunks=[chunk],
        embeddings=embeddings,
    )

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated qdrant failure")

    monkeypatch.setattr(
        "langchain_qdrant.QdrantVectorStore.similarity_search_with_score_by_vector",
        _raise,
    )

    scores = LangChainQdrantDenseSearchProvider(
        path=str(tmp_path),
        collection_name=COLLECTION,
    ).search_scores([1.0, 0.0, 0.0], limit=5)

    assert scores is None


def test_writer_closed_before_reader_no_deadlock(tmp_path) -> None:
    chunk = _make_chunk(suffix="lock")
    embeddings = {chunk.chunk_id: [1.0, 0.0, 0.0]}

    QdrantVectorStore(path=str(tmp_path)).upsert_chunks(
        collection_name=COLLECTION,
        chunks=[chunk],
        embeddings=embeddings,
    )

    scores = LangChainQdrantDenseSearchProvider(
        path=str(tmp_path),
        collection_name=COLLECTION,
    ).search_scores([1.0, 0.0, 0.0], limit=5)

    assert scores is not None
    assert chunk.chunk_id in scores
