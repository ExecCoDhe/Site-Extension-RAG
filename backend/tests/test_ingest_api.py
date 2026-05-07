from fastapi.testclient import TestClient

from app.jobs import job_manager
from app.jobs.models import PageRecord
from app.main import app
from app.workspace import workspace_store
from app.workspace.models import RunState


async def noop_ingest_job(job_id: str, url: str) -> None:
    return None


class FakeEmbeddingClient:
    embedded_texts: list[str] = []
    calls: list[list[str]] = []

    def __init__(self, *, api_key=None, model="fake", **kwargs) -> None:
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.__class__.embedded_texts = texts
        self.__class__.calls.append(texts)
        return [[float(index), 0.0] for index, _text in enumerate(texts, start=1)]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class NoopVectorStore:
    def __init__(self, *, path: str) -> None:
        pass

    def upsert_chunks(self, *, collection_name, chunks, embeddings) -> None:
        return None


def test_start_ingest_returns_job_and_status(monkeypatch) -> None:
    monkeypatch.setattr("app.api.ingest.is_public_http_url", lambda url: True)
    monkeypatch.setattr("app.api.ingest.run_ingest_job", noop_ingest_job)
    client = TestClient(app)

    response = client.post("/ingest", json={"url": "https://example.com/page"})

    assert response.status_code == 202
    body = response.json()
    assert body["state"] == "ingesting"
    assert body["hostname"] == "example.com"
    assert body["page_count"] == 0
    assert body["workspace_id"] == "default"

    status = client.get(f"/ingest/{body['job_id']}/status")
    assert status.status_code == 200
    assert status.json()["job_id"] == body["job_id"]

    workspace = client.get("/workspace/status")
    assert workspace.status_code == 200
    assert workspace.json()["state"] == "ingesting"


def test_active_job_returns_error_envelope(monkeypatch) -> None:
    monkeypatch.setattr("app.api.ingest.is_public_http_url", lambda url: True)
    monkeypatch.setattr("app.api.ingest.run_ingest_job", noop_ingest_job)
    client = TestClient(app)

    first = client.post("/ingest", json={"url": "https://example.com/page"})
    second = client.post("/ingest", json={"url": "https://example.com/other"})

    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "ACTIVE_JOB"
    assert second.json()["error"]["details"]["hostname"] == "example.com"


def test_recover_interrupted_ingest_allows_new_ingest(monkeypatch) -> None:
    monkeypatch.setattr("app.api.ingest.is_public_http_url", lambda url: True)
    monkeypatch.setattr("app.api.ingest.run_ingest_job", noop_ingest_job)
    client = TestClient(app)

    interrupted_run = workspace_store.start_ingest_run(
        seed_url="https://example.com/page",
        hostname="example.com",
        registrable_domain="example.com",
        included_subdomains=["example.com"],
        chunking_version="test",
        embedding_version="fake",
    )
    assert interrupted_run is not None
    blocked = client.post("/ingest", json={"url": "https://example.com/other"})
    assert blocked.status_code == 409

    recovered_count = workspace_store.recover_interrupted_ingest_runs()
    recovered_run = workspace_store.get_run(interrupted_run.run_id)

    assert recovered_count == 1
    assert recovered_run is not None
    assert recovered_run.state == RunState.ERROR
    assert recovered_run.error is not None
    assert recovered_run.error["code"] == "INGEST_INTERRUPTED"

    response = client.post("/ingest", json={"url": "https://example.com/other"})

    assert response.status_code == 202
    assert response.json()["state"] == "ingesting"


def test_unknown_status_returns_chat_before_ready() -> None:
    client = TestClient(app)

    response = client.get("/ingest/missing/status")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CHAT_BEFORE_READY"


def test_status_accepts_job_id_without_prefix(monkeypatch) -> None:
    monkeypatch.setattr("app.api.ingest.is_public_http_url", lambda url: True)
    monkeypatch.setattr("app.api.ingest.run_ingest_job", noop_ingest_job)
    client = TestClient(app)

    response = client.post("/ingest", json={"url": "https://example.com/page"})
    job_id = response.json()["job_id"]

    status = client.get(f"/ingest/{job_id.removeprefix('job_')}/status")

    assert status.status_code == 200
    assert status.json()["job_id"] == job_id


def test_private_seed_url_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr("app.api.ingest.is_public_http_url", lambda url: False)
    client = TestClient(app)

    response = client.post("/ingest", json={"url": "http://127.0.0.1:9999"})

    assert response.status_code == 422
    assert job_manager.active_ingest_job() is None


def test_run_ingest_embeds_persisted_child_chunks(monkeypatch) -> None:
    from app.api.ingest import run_ingest_job
    from app.config import Settings
    from app.crawl.crawler import CrawlResult

    async def fake_crawl_site(*args, **kwargs) -> CrawlResult:
        return CrawlResult(
            pages=[
                PageRecord(
                    url="https://example.com/a",
                    canonical_url="https://example.com/a",
                    title="A",
                    clean_text="one two three four five six seven",
                    content_hash="hash",
                    heading_paths=[["A"]],
                )
            ]
        )

    settings = Settings(
        gemini_api_key="fake",
        gemini_embedding_model="fake-embedding",
        child_chunk_token_budget=3,
        child_chunk_token_overlap=0,
        chunking_version="test-child",
    )
    run = workspace_store.start_ingest_run(
        seed_url="https://example.com/a",
        hostname="example.com",
        registrable_domain="example.com",
        included_subdomains=["example.com"],
        chunking_version=settings.chunking_version,
        embedding_version=settings.gemini_embedding_model,
    )
    assert run is not None
    FakeEmbeddingClient.calls = []

    monkeypatch.setattr("app.api.ingest.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.ingest.crawl_site", fake_crawl_site)
    monkeypatch.setattr("app.api.ingest.GoogleEmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr("app.api.ingest.QdrantVectorStore", NoopVectorStore)

    import anyio

    anyio.run(run_ingest_job, run.run_id, "https://example.com/a")

    stored_run = workspace_store.get_run(run.run_id)
    assert stored_run is not None
    assert stored_run.state == RunState.READY

    active_chunks = sorted(workspace_store.active_chunks(), key=lambda chunk: chunk.token_start)
    assert [chunk.text for chunk in active_chunks] == [
        "one two three",
        "four five six",
        "seven",
    ]
    assert FakeEmbeddingClient.embedded_texts == [chunk.text for chunk in active_chunks]
    assert workspace_store.embeddings(settings.gemini_embedding_model) == {
        active_chunks[0].chunk_id: [1.0, 0.0],
        active_chunks[1].chunk_id: [2.0, 0.0],
        active_chunks[2].chunk_id: [3.0, 0.0],
    }


def test_run_ingest_skips_unchanged_pages_on_resync(monkeypatch) -> None:
    from app.api.ingest import run_ingest_job
    from app.config import Settings
    from app.crawl.crawler import CrawlResult

    async def fake_crawl_site(*args, **kwargs) -> CrawlResult:
        return CrawlResult(
            pages=[
                PageRecord(
                    url="https://example.com/a",
                    canonical_url="https://example.com/a",
                    title="A",
                    clean_text="one two three four five six seven",
                    content_hash="same-hash",
                    heading_paths=[["A"]],
                )
            ]
        )

    settings = Settings(
        gemini_api_key="fake",
        gemini_embedding_model="fake-embedding",
        child_chunk_token_budget=3,
        child_chunk_token_overlap=0,
        chunking_version="test-child",
    )
    monkeypatch.setattr("app.api.ingest.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.ingest.crawl_site", fake_crawl_site)
    monkeypatch.setattr("app.api.ingest.GoogleEmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr("app.api.ingest.QdrantVectorStore", NoopVectorStore)

    import anyio

    first_run = workspace_store.start_ingest_run(
        seed_url="https://example.com/a",
        hostname="example.com",
        registrable_domain="example.com",
        included_subdomains=["example.com"],
        chunking_version=settings.chunking_version,
        embedding_version=settings.gemini_embedding_model,
    )
    assert first_run is not None
    FakeEmbeddingClient.calls = []
    anyio.run(run_ingest_job, first_run.run_id, "https://example.com/a")
    initial_chunk_ids = {chunk.chunk_id for chunk in workspace_store.active_chunks()}

    second_run = workspace_store.start_ingest_run(
        seed_url="https://example.com/a",
        hostname="example.com",
        registrable_domain="example.com",
        included_subdomains=["example.com"],
        chunking_version=settings.chunking_version,
        embedding_version=settings.gemini_embedding_model,
    )
    assert second_run is not None
    anyio.run(run_ingest_job, second_run.run_id, "https://example.com/a")

    stored_second_run = workspace_store.get_run(second_run.run_id)
    assert stored_second_run is not None
    assert stored_second_run.state == RunState.READY
    assert stored_second_run.skipped_count == 1
    assert FakeEmbeddingClient.calls == [["one two three", "four five six", "seven"]]
    assert {chunk.chunk_id for chunk in workspace_store.active_chunks()} == initial_chunk_ids
