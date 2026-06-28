from fastapi.testclient import TestClient

from app.main import app
from app.rag.generation import GeneratedAnswer
from app.workspace import workspace_store
from app.workspace.models import (
    AcquisitionMethod,
    ChildChunkRecord,
    PageVersionRecord,
    ParentSectionRecord,
)


class FakeEmbeddingClient:
    def __init__(self, *, api_key=None, model="fake", **kwargs) -> None:
        pass

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeGenerationClient:
    def __init__(self, *, api_key=None, model="fake", **kwargs) -> None:
        pass

    def generate_answer(self, *, question, hits):
        return GeneratedAnswer(
            answer="Alpha is described in the indexed page.",
            grounded=True,
            supporting_chunk_ids=[hits[0].chunk.chunk_id],
        )

    def generate_answer_from_evidence(self, *, question, evidence):
        return GeneratedAnswer(
            answer="Alpha is described in the indexed page.",
            grounded=True,
            groundedness="grounded",
            claims=[
                {
                    "text": "Alpha is described in the indexed page.",
                    "supporting_evidence_ids": [evidence[0].evidence_id],
                    "supported": True,
                }
            ],
            supporting_evidence_ids=[evidence[0].evidence_id],
        )


class UnsupportedEvidenceGenerationClient(FakeGenerationClient):
    def generate_answer_from_evidence(self, *, question, evidence):
        return GeneratedAnswer(
            answer="Unsupported answer.",
            grounded=True,
            groundedness="grounded",
            claims=[
                {
                    "text": "Unsupported answer.",
                    "supporting_evidence_ids": [],
                    "supported": False,
                }
            ],
            supporting_evidence_ids=[],
        )


def make_ready_workspace() -> str:
    run = workspace_store.start_ingest_run(
        seed_url="https://example.com/a",
        hostname="example.com",
        registrable_domain="example.com",
        included_subdomains=["example.com"],
        chunking_version="test",
        embedding_version="fake",
    )
    assert run is not None
    page = PageVersionRecord(
        page_id="page_1",
        workspace_id=workspace_store.workspace_id,
        run_id=run.run_id,
        canonical_url="https://example.com/a",
        discovered_url="https://example.com/a",
        title="A",
        acquisition_method=AcquisitionMethod.HTML,
        content_hash="hash",
        quality_score=1.0,
        quality_signals={},
        boilerplate_removed=[],
        clean_text="alpha content",
    )
    section = ParentSectionRecord(
        section_id="section_1",
        page_id="page_1",
        workspace_id=workspace_store.workspace_id,
        heading_path=["A"],
        section_index=0,
        text="alpha content",
        start_offset=0,
        end_offset=13,
    )
    chunk = ChildChunkRecord(
        chunk_id="chunk_1",
        section_id="section_1",
        page_id="page_1",
        workspace_id=workspace_store.workspace_id,
        chunking_version="test",
        title="A",
        url="https://example.com/a",
        heading_path=["A"],
        text="alpha content",
        token_start=0,
        token_end=2,
    )
    workspace_store.replace_active_content(
        run_id=run.run_id,
        pages=[page],
        sections=[section],
        chunks=[chunk],
        embeddings={"chunk_1": [1.0, 0.0]},
        embedding_version="fake",
        rendered_fallback_count=0,
        skipped_count=0,
    )
    return run.run_id


def test_chat_before_ready_returns_error() -> None:
    client = TestClient(app)

    response = client.post("/chat", json={"question": "What is alpha?"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CHAT_BEFORE_READY"


def test_chat_returns_grounded_answer_with_citations(monkeypatch) -> None:
    monkeypatch.setattr("app.api.chat.LangChainEmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr("app.api.chat.LangChainGenerationClient", FakeGenerationClient)
    make_ready_workspace()
    client = TestClient(app)

    response = client.post("/chat", json={"question": "What is alpha?", "session_id": "test"})

    assert response.status_code == 200
    body = response.json()
    assert body["groundedness"] == "grounded"
    assert body["answer"] == "Alpha is described in the indexed page."
    assert body["citations"][0]["url"] == "https://example.com/a"
    assert body["citations"][0]["chunk_id"] == "chunk_1"
    assert body["citations"][0]["snippet"] == "alpha content"
    assert body["trace_id"].startswith("trace_")


def test_chat_missing_google_configuration_returns_error() -> None:
    make_ready_workspace()
    client = TestClient(app)

    response = client.post("/chat", json={"question": "What is alpha?"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MISSING_API_KEY"


def test_chat_downgrades_unsupported_claims(monkeypatch) -> None:
    monkeypatch.setattr("app.api.chat.LangChainEmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr("app.api.chat.LangChainGenerationClient", UnsupportedEvidenceGenerationClient)
    make_ready_workspace()
    client = TestClient(app)

    response = client.post("/chat", json={"question": "What is alpha?"})

    assert response.status_code == 200
    body = response.json()
    assert body["groundedness"] == "not_grounded"
    assert body["answer"] == "The indexed site content does not contain enough information to answer that."
    assert body["citations"] == []
