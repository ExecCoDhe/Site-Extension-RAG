from app.config import Settings
from app.index import VectorIndex
from app.jobs.models import ChunkRecord, IngestJob, JobState
from app.rag.generation import GeneratedAnswer
from app.rag.service import NOT_FOUND_ANSWER, answer_question


class FakeEmbeddingClient:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class GroundedGenerationClient:
    def generate_answer(self, *, question, hits):
        return GeneratedAnswer(
            answer="Alpha is described in the indexed page.",
            grounded=True,
            supporting_chunk_ids=[hits[0].chunk.chunk_id],
        )


class UnsupportedGenerationClient:
    def generate_answer(self, *, question, hits):
        return GeneratedAnswer(
            answer="Unsupported model answer.",
            grounded=True,
            supporting_chunk_ids=["not-retrieved"],
        )


def ready_job() -> IngestJob:
    chunk = ChunkRecord(
        chunk_id="job_1:0000:0000",
        url="https://example.com/a",
        title="A",
        text="alpha content",
    )
    return IngestJob(
        job_id="job_1",
        state=JobState.READY,
        hostname="example.com",
        page_count=1,
        chunks=[chunk],
        vector_index=VectorIndex.from_embeddings([chunk], [[1.0, 0.0]]),
    )


def test_answer_question_returns_validated_citations() -> None:
    response = answer_question(
        job=ready_job(),
        question="What is alpha?",
        settings=Settings(top_k=5),
        embedding_client=FakeEmbeddingClient(),
        generation_client=GroundedGenerationClient(),
    )

    assert response.grounded is True
    assert response.answer == "Alpha is described in the indexed page."
    assert response.citations[0].chunk_id == "job_1:0000:0000"
    assert response.citations[0].url == "https://example.com/a"


def test_answer_question_downgrades_unsupported_generation() -> None:
    response = answer_question(
        job=ready_job(),
        question="What is alpha?",
        settings=Settings(top_k=5),
        embedding_client=FakeEmbeddingClient(),
        generation_client=UnsupportedGenerationClient(),
    )

    assert response.grounded is False
    assert response.answer == NOT_FOUND_ANSWER
    assert response.citations == []
