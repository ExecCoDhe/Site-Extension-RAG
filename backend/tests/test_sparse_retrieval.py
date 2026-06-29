from app.config import Settings
from app.retrieval.service import BM25SparseSearchProvider, RetrievalPipeline
from app.workspace.models import ChildChunkRecord


class FakeEmbeddingClient:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _chunk(*, chunk_id: str, text: str, title: str = "Title") -> ChildChunkRecord:
    return ChildChunkRecord(
        chunk_id=chunk_id,
        section_id=f"section_{chunk_id}",
        page_id="page_1",
        workspace_id="default",
        chunking_version="test",
        title=title,
        url=f"https://example.com/{chunk_id}",
        heading_path=[title],
        text=text,
        token_start=0,
        token_end=3,
    )


def test_bm25_provider_ranks_matching_chunk_higher_and_bounds_scores() -> None:
    chunks = [
        _chunk(chunk_id="chunk_1", text="alpha beta gamma", title="Alpha"),
        _chunk(chunk_id="chunk_2", text="delta epsilon zeta", title="Delta"),
        _chunk(chunk_id="chunk_3", text="theta iota kappa", title="Theta"),
    ]
    provider = BM25SparseSearchProvider(chunks=chunks)

    scores = provider.search_scores("gamma", limit=5)
    assert scores is not None
    assert scores["chunk_1"] == 1.0
    assert scores.get("chunk_2", 0.0) < scores["chunk_1"]
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_bm25_provider_returns_none_for_empty_corpus() -> None:
    provider = BM25SparseSearchProvider(chunks=[])

    assert provider.search_scores("anything", limit=5) is None


def test_bm25_provider_returns_empty_dict_for_no_match_query() -> None:
    chunks = [_chunk(chunk_id="chunk_1", text="alpha beta", title="Alpha")]
    provider = BM25SparseSearchProvider(chunks=chunks)

    scores = provider.search_scores("xyzzy", limit=5)
    assert scores == {}


def test_retrieval_pipeline_with_bm25_preserves_evidence_contract() -> None:
    chunks = [
        _chunk(chunk_id="chunk_1", text="alpha beta detail", title="Alpha"),
        _chunk(chunk_id="chunk_2", text="gamma delta detail", title="Gamma"),
        _chunk(chunk_id="chunk_3", text="theta iota kappa", title="Theta"),
    ]
    sparse_provider = BM25SparseSearchProvider(chunks=chunks)

    result = RetrievalPipeline(
        settings=Settings(retrieval_candidate_limit=5, rerank_limit=3, parent_context_limit=1),
        embedding_client=FakeEmbeddingClient(),
        chunks=chunks,
        embeddings={
            "chunk_1": [1.0, 0.0],
            "chunk_2": [0.0, 1.0],
            "chunk_3": [0.5, 0.5],
        },
        sparse_search_provider=sparse_provider,
    ).retrieve("gamma")

    assert result.evidence
    assert any(item.sparse_score > 0 for item in result.evidence)
    for item in result.evidence:
        assert item.dense_score is not None
        assert item.sparse_score is not None
        assert item.rerank_score is not None
        assert 0.0 <= item.sparse_score <= 1.0

    assert result.trace["candidate_count"] >= 1
    assert result.trace["deduped_count"] >= 1
    assert result.trace["evidence_count"] == len(result.evidence)
