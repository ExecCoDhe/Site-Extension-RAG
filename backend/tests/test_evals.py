from app.config import Settings
from app.evals import EvalCase, run_retrieval_eval
from app.workspace.models import ChildChunkRecord


class FakeEmbeddingClient:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_retrieval_eval_scores_expected_evidence_and_decomposition() -> None:
    chunk = ChildChunkRecord(
        chunk_id="chunk_1",
        section_id="section_1",
        page_id="page_1",
        workspace_id="default",
        chunking_version="test",
        title="Alpha",
        url="https://example.com/a",
        heading_path=["Alpha"],
        text="alpha beta",
        token_start=0,
        token_end=2,
    )

    metrics = run_retrieval_eval(
        cases=[
            EvalCase(
                question="Compare alpha and beta",
                expected_chunk_ids=["chunk_1"],
                expected_urls=["https://example.com/a"],
                should_decompose=True,
            )
        ],
        settings=Settings(retrieval_candidate_limit=5, rerank_limit=3),
        embedding_client=FakeEmbeddingClient(),
        chunks=[chunk],
        embeddings={"chunk_1": [1.0, 0.0]},
    )

    assert metrics["hit_rate"] == 1.0
    assert metrics["decomposition_accuracy"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0


def test_retrieval_eval_gives_graded_credit_for_equivalent_evidence() -> None:
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
            text="alpha beta",
            token_start=0,
            token_end=2,
        ),
        ChildChunkRecord(
            chunk_id="chunk_2",
            section_id="section_2",
            page_id="page_1",
            workspace_id="default",
            chunking_version="test",
            title="Gamma",
            url="https://example.com/g",
            heading_path=["Gamma"],
            text="gamma delta",
            token_start=0,
            token_end=2,
        ),
    ]

    metrics = run_retrieval_eval(
        cases=[
            EvalCase(
                question="gamma",
                expected_chunk_ids=["missing_chunk"],
                equivalent_chunk_ids=["chunk_2"],
            )
        ],
        settings=Settings(retrieval_candidate_limit=5, rerank_limit=3),
        embedding_client=FakeEmbeddingClient(),
        chunks=chunks,
        embeddings={
            "chunk_1": [1.0, 0.0],
            "chunk_2": [1.0, 0.0],
        },
    )

    assert metrics["hit_rate"] == 1.0
    assert metrics["exact_chunk_hit_rate"] == 0.0
    assert metrics["equivalent_chunk_hit_rate"] == 1.0
    assert metrics["recall_at_k"] == 0.0
    assert metrics["graded_recall"] == 0.5
    assert metrics["mrr"] == 1.0
