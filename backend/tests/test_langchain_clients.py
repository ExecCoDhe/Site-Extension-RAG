import json
from typing import Any

import pytest

from app.evals.faithfulness import evaluate_faithfulness
from app.index.embeddings import (
    GOOGLE_EMBEDDING_BATCH_SIZE,
    LangChainEmbeddingClient,
    MissingGoogleConfiguration,
)
from app.index.vector_index import SearchHit
from app.jobs.models import ChunkRecord
from app.rag.generation import (
    NOT_FOUND_ANSWER,
    GeneratedAnswerSchema,
    LangChainGenerationClient,
    _ClaimSchema,
)
from app.retrieval.models import EvidenceSnippet


class RecordingEmbeddings:
    def __init__(self) -> None:
        self.document_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.document_vectors = [[0.1, 0.2], [0.3, 0.4]]
        self.query_vector = [0.5, 0.6]
        self.raise_on_documents: Exception | None = None
        self.raise_on_query: Exception | None = None

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.document_calls.append({"texts": texts, **kwargs})
        if self.raise_on_documents is not None:
            raise self.raise_on_documents
        return self.document_vectors[: len(texts)]

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        self.query_calls.append({"text": text, **kwargs})
        if self.raise_on_query is not None:
            raise self.raise_on_query
        return self.query_vector


class FakeStructuredRunnable:
    def __init__(self, result: object | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.result


class FakeChatMessage:
    def __init__(self, content: object) -> None:
        self.content = content


class FakeChatModel:
    def __init__(
        self,
        *,
        structured: FakeStructuredRunnable | None = None,
        raw_result: object = "",
        raw_error: Exception | None = None,
    ) -> None:
        self.structured = structured or FakeStructuredRunnable()
        self.raw_result = raw_result
        self.raw_error = raw_error
        self.structured_schema: type | None = None
        self.structured_method: str | None = None
        self.raw_prompts: list[str] = []

    def with_structured_output(self, schema: type, *, method: str) -> FakeStructuredRunnable:
        self.structured_schema = schema
        self.structured_method = method
        return self.structured

    def invoke(self, prompt: str) -> FakeChatMessage:
        self.raw_prompts.append(prompt)
        if self.raw_error is not None:
            raise self.raw_error
        return FakeChatMessage(self.raw_result)


def evidence() -> list[EvidenceSnippet]:
    return [
        EvidenceSnippet(
            evidence_id="evidence_1",
            chunk_id="chunk_1",
            section_id="section_1",
            parent_context_id="section_1",
            url="https://example.com/a",
            title="A",
            heading_path=["A"],
            snippet="alpha content",
            nearby_context="alpha content with surrounding context",
            dense_score=1.0,
            sparse_score=1.0,
            rerank_score=1.0,
        )
    ]


@pytest.fixture
def recording_embeddings(monkeypatch: pytest.MonkeyPatch) -> RecordingEmbeddings:
    fake = RecordingEmbeddings()
    monkeypatch.setattr("app.index.embeddings.GoogleGenerativeAIEmbeddings", lambda **_: fake)
    return fake


@pytest.fixture
def fake_chat(monkeypatch: pytest.MonkeyPatch) -> FakeChatModel:
    fake = FakeChatModel()
    monkeypatch.setattr("app.rag.generation.ChatGoogleGenerativeAI", lambda **_: fake)
    return fake


def test_langchain_embedding_client_missing_api_key() -> None:
    with pytest.raises(MissingGoogleConfiguration, match="GEMINI_API_KEY is not configured"):
        LangChainEmbeddingClient(api_key=None, model="gemini-embedding-001")


def test_langchain_embedding_client_pins_developer_api(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_gge(**kwargs: object) -> RecordingEmbeddings:
        captured.update(kwargs)
        return RecordingEmbeddings()

    monkeypatch.setattr("app.index.embeddings.GoogleGenerativeAIEmbeddings", fake_gge)
    LangChainEmbeddingClient(api_key="fake-key", model="gemini-embedding-001")
    assert captured.get("vertexai") is False


def test_langchain_embedding_client_embed_documents(
    recording_embeddings: RecordingEmbeddings,
) -> None:
    client = LangChainEmbeddingClient(api_key="fake-key", model="gemini-embedding-001")
    vectors = client.embed_documents(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert len(recording_embeddings.document_calls) == 1
    call = recording_embeddings.document_calls[0]
    assert call["texts"] == ["a", "b"]
    assert call["task_type"] == "RETRIEVAL_DOCUMENT"
    assert call["batch_size"] == GOOGLE_EMBEDDING_BATCH_SIZE
    assert "output_dimensionality" not in call


def test_langchain_embedding_client_embed_query(recording_embeddings: RecordingEmbeddings) -> None:
    client = LangChainEmbeddingClient(api_key="fake-key", model="gemini-embedding-001")
    vector = client.embed_query("q")

    assert vector == [0.5, 0.6]
    assert len(recording_embeddings.query_calls) == 1
    call = recording_embeddings.query_calls[0]
    assert call["text"] == "q"
    assert call["task_type"] == "RETRIEVAL_QUERY"
    assert "output_dimensionality" not in call


def test_langchain_embedding_client_empty_documents_skips_api(
    recording_embeddings: RecordingEmbeddings,
) -> None:
    client = LangChainEmbeddingClient(api_key="fake-key", model="gemini-embedding-001")
    assert client.embed_documents([]) == []
    assert recording_embeddings.document_calls == []


def test_langchain_embedding_client_maps_api_key_invalid_error(
    recording_embeddings: RecordingEmbeddings,
) -> None:
    recording_embeddings.raise_on_documents = Exception("API_KEY_INVALID: bad key")
    client = LangChainEmbeddingClient(api_key="fake-key", model="gemini-embedding-001")

    with pytest.raises(
        MissingGoogleConfiguration,
        match="GEMINI_API_KEY is invalid or not enabled for the Gemini API",
    ):
        client.embed_documents(["a"])


def test_langchain_embedding_client_maps_generic_error(
    recording_embeddings: RecordingEmbeddings,
) -> None:
    recording_embeddings.raise_on_query = Exception("network down")
    client = LangChainEmbeddingClient(api_key="fake-key", model="gemini-embedding-001")

    with pytest.raises(MissingGoogleConfiguration, match="Google embedding configuration is unusable"):
        client.embed_query("q")


def test_langchain_generation_client_missing_api_key() -> None:
    with pytest.raises(MissingGoogleConfiguration, match="GEMINI_API_KEY is not configured"):
        LangChainGenerationClient(api_key=None, model="gemini-2.0-flash")


def test_langchain_generation_client_structured_success(fake_chat: FakeChatModel) -> None:
    fake_chat.structured = FakeStructuredRunnable(
        result=GeneratedAnswerSchema(
            reasoning="Because alpha is in evidence.",
            answer="Alpha is supported.",
            groundedness="grounded",
            claims=[
                _ClaimSchema(
                    text="Alpha is supported.",
                    supporting_evidence_ids=["evidence_1"],
                    supported=True,
                )
            ],
            supporting_evidence_ids=["evidence_1"],
        )
    )

    client = LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")
    answer = client.generate_answer_from_evidence(
        question="What is alpha?",
        evidence=evidence(),
    )

    assert answer.groundedness == "grounded"
    assert answer.supporting_evidence_ids == ["evidence_1"]
    assert answer.claims[0]["text"] == "Alpha is supported."
    assert fake_chat.structured_schema is GeneratedAnswerSchema
    assert fake_chat.structured_method == "json_schema"


def test_langchain_generation_client_pins_developer_api(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_chat(**kwargs: object) -> FakeChatModel:
        captured.update(kwargs)
        return FakeChatModel()

    monkeypatch.setattr("app.rag.generation.ChatGoogleGenerativeAI", fake_chat)
    LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")
    assert captured.get("vertexai") is False


def test_langchain_generation_client_structured_empty_falls_back_to_raw(fake_chat: FakeChatModel) -> None:
    fake_chat.structured = FakeStructuredRunnable(result=GeneratedAnswerSchema())
    fake_chat.raw_result = json.dumps(
        {
            "answer": "Alpha is supported.",
            "groundedness": "grounded",
            "claims": [
                {
                    "text": "Alpha is supported.",
                    "supporting_evidence_ids": ["evidence_1"],
                    "supported": True,
                }
            ],
            "supporting_evidence_ids": ["evidence_1"],
        }
    )

    client = LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")
    answer = client.generate_answer_from_evidence(
        question="What is alpha?",
        evidence=evidence(),
    )

    assert answer.groundedness == "grounded"
    assert answer.supporting_evidence_ids == ["evidence_1"]
    assert fake_chat.raw_prompts


def test_langchain_generation_client_both_empty_returns_not_found(fake_chat: FakeChatModel) -> None:
    fake_chat.structured = FakeStructuredRunnable(result=GeneratedAnswerSchema())
    fake_chat.raw_result = "{}"

    client = LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")
    answer = client.generate_answer_from_evidence(
        question="What is alpha?",
        evidence=evidence(),
    )

    assert answer.groundedness == "not_grounded"
    assert answer.answer == NOT_FOUND_ANSWER


def test_langchain_generation_client_partial_grounding(fake_chat: FakeChatModel) -> None:
    fake_chat.structured = FakeStructuredRunnable(
        result=GeneratedAnswerSchema(
            answer="Mixed support.",
            groundedness="grounded",
            claims=[
                _ClaimSchema(
                    text="Alpha is supported.",
                    supporting_evidence_ids=["evidence_1"],
                    supported=True,
                ),
                _ClaimSchema(
                    text="Omega is unsupported.",
                    supporting_evidence_ids=[],
                    supported=False,
                ),
            ],
            supporting_evidence_ids=["evidence_1"],
        )
    )

    client = LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")
    answer = client.generate_answer_from_evidence(
        question="What is alpha?",
        evidence=evidence(),
    )

    assert answer.groundedness == "partially_grounded"


def test_langchain_generation_client_structured_validation_error_falls_back_to_raw(
    fake_chat: FakeChatModel,
) -> None:
    fake_chat.structured = FakeStructuredRunnable(error=ValueError("structured schema parse failed"))
    fake_chat.raw_result = json.dumps(
        {
            "answer": "Alpha is supported.",
            "groundedness": "grounded",
            "claims": [
                {
                    "text": "Alpha is supported.",
                    "supporting_evidence_ids": ["evidence_1"],
                    "supported": True,
                }
            ],
            "supporting_evidence_ids": ["evidence_1"],
        }
    )

    client = LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")
    answer = client.generate_answer_from_evidence(
        question="What is alpha?",
        evidence=evidence(),
    )

    assert answer.groundedness == "grounded"
    assert fake_chat.raw_prompts


def test_langchain_generation_client_reasoning_only_skips_raw_fallback(fake_chat: FakeChatModel) -> None:
    fake_chat.structured = FakeStructuredRunnable(
        result=GeneratedAnswerSchema(
            reasoning="Evidence is insufficient for a definitive answer.",
            groundedness="not_grounded",
        )
    )

    client = LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")
    answer = client.generate_answer_from_evidence(
        question="What is alpha?",
        evidence=evidence(),
    )

    assert answer.groundedness == "not_grounded"
    assert fake_chat.raw_prompts == []


def test_langchain_generation_client_structured_transport_error_skips_raw(
    fake_chat: FakeChatModel,
) -> None:
    fake_chat.structured = FakeStructuredRunnable(error=RuntimeError("structured failed"))

    client = LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")
    with pytest.raises(MissingGoogleConfiguration, match="Google chat configuration is unusable"):
        client.generate_answer_from_evidence(
            question="What is alpha?",
            evidence=evidence(),
        )

    assert fake_chat.raw_prompts == []


def test_langchain_generation_client_generate_answer(fake_chat: FakeChatModel) -> None:
    fake_chat.raw_result = json.dumps(
        {
            "answer": "Alpha is described in the indexed page.",
            "reasoning": "Chunk supports alpha.",
            "grounded": True,
            "supporting_chunk_ids": ["chunk_1"],
        }
    )
    chunk = ChunkRecord(
        chunk_id="chunk_1",
        url="https://example.com/a",
        title="A",
        text="alpha content",
    )
    client = LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")
    answer = client.generate_answer(
        question="What is alpha?",
        hits=[SearchHit(chunk=chunk, score=1.0)],
    )

    assert answer.grounded is True
    assert answer.supporting_chunk_ids == ["chunk_1"]


def test_langchain_generation_client_raw_failure_raises_missing_configuration(
    fake_chat: FakeChatModel,
) -> None:
    fake_chat.structured = FakeStructuredRunnable(result=GeneratedAnswerSchema())
    fake_chat.raw_error = RuntimeError("raw failed")

    client = LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")
    with pytest.raises(MissingGoogleConfiguration, match="Google chat configuration is unusable"):
        client.generate_answer_from_evidence(
            question="What is alpha?",
            evidence=evidence(),
        )


def test_faithfulness_handles_malformed_score(fake_chat: FakeChatModel) -> None:
    fake_chat.raw_result = json.dumps(
        {
            "faithful": False,
            "score": "not-a-number",
            "reasoning": "Malformed score.",
            "unsupported_claims": [],
        }
    )
    client = LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")

    result = evaluate_faithfulness(
        question="What is alpha?",
        answer="Alpha is supported.",
        evidence_snippets=["alpha content"],
        generation_client=client,
    )

    assert result.score == 0.0
    assert result.reasoning == "Failed to evaluate: generation error."


def test_langchain_generation_client_generate_raw_string_content(fake_chat: FakeChatModel) -> None:
    fake_chat.raw_result = "plain text"
    client = LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")
    assert client.generate_raw("x") == "plain text"


def test_langchain_generation_client_generate_raw_content_blocks(fake_chat: FakeChatModel) -> None:
    fake_chat.raw_result = [{"text": "hello "}, {"text": "world"}]
    client = LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")
    assert client.generate_raw("x") == "hello world"


def test_faithfulness_clamps_score_with_langchain_client(fake_chat: FakeChatModel) -> None:
    fake_chat.raw_result = json.dumps(
        {
            "faithful": True,
            "score": 2.0,
            "reasoning": "Fully supported.",
            "unsupported_claims": [],
        }
    )
    client = LangChainGenerationClient(api_key="fake-key", model="gemini-2.0-flash")

    result = evaluate_faithfulness(
        question="What is alpha?",
        answer="Alpha is supported.",
        evidence_snippets=["alpha content"],
        generation_client=client,
    )

    assert result.score == 1.0
