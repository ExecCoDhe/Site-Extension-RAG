import pytest

from app.config import Settings
from app.index import VectorIndex, build_index
from app.index.embeddings import (
    GOOGLE_EMBEDDING_BATCH_SIZE,
    GoogleEmbeddingClient,
    _configuration_error_message,
)
from app.jobs import job_manager
from app.jobs.models import ChunkRecord, PageRecord


class FakeEmbeddingClient:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]


def test_vector_index_returns_nearest_chunk() -> None:
    chunks = [
        ChunkRecord(chunk_id="c1", url="https://example.com/a", title="A", text="alpha"),
        ChunkRecord(chunk_id="c2", url="https://example.com/b", title="B", text="omega"),
    ]
    index = VectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]])

    hits = index.search([1.0, 0.0], top_k=1)

    assert hits[0].chunk.chunk_id == "c1"
    assert hits[0].score == pytest.approx(1.0)


def test_build_index_chunks_pages_and_indexes_vectors() -> None:
    settings = Settings(chunk_size_chars=100, chunk_overlap_chars=10)

    chunks, index = build_index(
        job_id="job_1",
        pages=[
            PageRecord(url="https://example.com/a", title="A", clean_text="alpha text"),
            PageRecord(url="https://example.com/b", title="B", clean_text="omega text"),
        ],
        settings=settings,
        embedding_client=FakeEmbeddingClient(),
    )

    assert len(chunks) == 2
    assert index.search([0.0, 1.0], top_k=1)[0].chunk.title == "B"


def test_embedding_error_message_identifies_invalid_api_key() -> None:
    error = Exception("400 INVALID_ARGUMENT: API_KEY_INVALID API Key not found.")

    assert _configuration_error_message(error) == (
        "GEMINI_API_KEY is invalid or not enabled for the Gemini API."
    )


def test_google_embedding_client_batches_document_embeddings(monkeypatch) -> None:
    calls = []

    class FakeEmbedding:
        def __init__(self, value: int) -> None:
            self.values = [float(value)]

    class FakeResponse:
        def __init__(self, count: int) -> None:
            self.embeddings = [FakeEmbedding(index) for index in range(count)]

    class FakeModels:
        def embed_content(self, *, model, contents, config) -> FakeResponse:
            calls.append(list(contents))
            return FakeResponse(len(contents))

    class FakeClient:
        def __init__(self, *, api_key) -> None:
            self.models = FakeModels()

    monkeypatch.setattr("app.index.embeddings.genai.Client", FakeClient)
    texts = [f"text {index}" for index in range(GOOGLE_EMBEDDING_BATCH_SIZE + 1)]

    embeddings = GoogleEmbeddingClient(api_key="fake", model="fake").embed_documents(texts)

    assert calls == [
        texts[:GOOGLE_EMBEDDING_BATCH_SIZE],
        texts[GOOGLE_EMBEDDING_BATCH_SIZE:],
    ]
    assert len(embeddings) == len(texts)


def test_mark_ready_stores_chunks_and_index() -> None:
    job = job_manager.create_ingest_job("example.com")
    assert job is not None
    chunks = [ChunkRecord(chunk_id="c1", url="https://example.com", title="Example", text="alpha")]
    index = VectorIndex.from_embeddings(chunks, [[1.0, 0.0]])

    job_manager.mark_ready(job.job_id, chunks=chunks, vector_index=index)

    ready_job = job_manager.get(job.job_id)
    assert ready_job is not None
    assert ready_job.state == "ready"
    assert ready_job.completed_at is not None
    assert ready_job.chunks == chunks
    assert ready_job.vector_index is index
