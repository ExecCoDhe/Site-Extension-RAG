from app.config import Settings
from app.retrieval import RetrievalPipeline
from app.workspace.models import ChildChunkRecord


class FakeEmbeddingClient:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeDenseSearchProvider:
    def search_scores(self, query_embedding: list[float], *, limit: int) -> dict[str, float]:
        return {"chunk_2": 0.99}


def test_retrieval_assembles_bounded_parent_context_for_evidence() -> None:
    chunks = [
        ChildChunkRecord(
            chunk_id="chunk_1",
            section_id="section_1",
            page_id="page_1",
            workspace_id="default",
            chunking_version="test",
            title="Alpha",
            url="https://example.com/a",
            heading_path=["Alpha"],
            text="alpha first detail",
            token_start=0,
            token_end=3,
        ),
        ChildChunkRecord(
            chunk_id="chunk_2",
            section_id="section_1",
            page_id="page_1",
            workspace_id="default",
            chunking_version="test",
            title="Alpha",
            url="https://example.com/a",
            heading_path=["Alpha"],
            text="beta second detail",
            token_start=3,
            token_end=6,
        ),
    ]

    result = RetrievalPipeline(
        settings=Settings(retrieval_candidate_limit=5, rerank_limit=3, parent_context_limit=1),
        embedding_client=FakeEmbeddingClient(),
        chunks=chunks,
        embeddings={
            "chunk_1": [1.0, 0.0],
            "chunk_2": [1.0, 0.0],
        },
    ).retrieve("alpha beta")

    assert result.trace["parent_context_count"] == 1
    assert len(result.evidence) == 1
    assert result.evidence[0].parent_context_id == "section_1"
    assert "alpha first detail" in result.evidence[0].nearby_context
    assert "beta second detail" in result.evidence[0].nearby_context
    assert result.parent_contexts[0].evidence_ids == ["evidence_1"]


def test_retrieval_prefers_dense_search_provider_scores() -> None:
    chunks = [
        ChildChunkRecord(
            chunk_id="chunk_1",
            section_id="section_1",
            page_id="page_1",
            workspace_id="default",
            chunking_version="test",
            title="Alpha",
            url="https://example.com/a",
            heading_path=["Alpha"],
            text="alpha",
            token_start=0,
            token_end=1,
        ),
        ChildChunkRecord(
            chunk_id="chunk_2",
            section_id="section_2",
            page_id="page_1",
            workspace_id="default",
            chunking_version="test",
            title="Omega",
            url="https://example.com/o",
            heading_path=["Omega"],
            text="omega",
            token_start=0,
            token_end=1,
        ),
    ]

    result = RetrievalPipeline(
        settings=Settings(retrieval_candidate_limit=5, rerank_limit=3),
        embedding_client=FakeEmbeddingClient(),
        chunks=chunks,
        embeddings={},
        dense_search_provider=FakeDenseSearchProvider(),
    ).retrieve("question")

    assert result.evidence[0].chunk_id == "chunk_2"
    assert result.evidence[0].dense_score == 0.99
