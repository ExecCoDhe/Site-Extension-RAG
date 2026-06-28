"""Characterization contract tests for Milestone B.

These tests lock public retrieval and /chat response shapes before the
Milestone C LangChain rewrite. They assert field names, enum values, error
codes, and set membership — not exact ranking positions or score values,
which are allowed to drift when BM25/Qdrant/LangChain internals change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.rag.generation import NOT_FOUND_ANSWER as GENERATION_NOT_FOUND_ANSWER
from app.rag.generation import GeneratedAnswer
from app.rag.service import NOT_FOUND_ANSWER
from app.retrieval import RetrievalPipeline
from app.retrieval.models import QueryPlan, RetrievalResult
from app.workspace import Groundedness, workspace_store
from app.workspace.models import (
    AcquisitionMethod,
    ChildChunkRecord,
    PageVersionRecord,
    ParentSectionRecord,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "characterization_corpus.json"

RETRIEVAL_RESULT_KEYS = {"query_plan", "candidates", "evidence", "parent_contexts", "trace"}
QUERY_PLAN_KEYS = {"original_question", "rewritten_question", "decomposed", "subqueries"}
EVIDENCE_SNIPPET_KEYS = {
    "evidence_id",
    "chunk_id",
    "section_id",
    "parent_context_id",
    "url",
    "title",
    "heading_path",
    "snippet",
    "nearby_context",
    "dense_score",
    "sparse_score",
    "rerank_score",
}
PARENT_CONTEXT_KEYS = {
    "parent_context_id",
    "section_id",
    "url",
    "title",
    "heading_path",
    "text",
    "evidence_ids",
    "rerank_score",
}
CANDIDATE_SCORE_KEYS = {"dense_score", "sparse_score", "rerank_score"}
RETRIEVAL_CANDIDATE_KEYS = {
    "chunk",
    "dense_score",
    "sparse_score",
    "fused_score",
    "rerank_score",
    "subquery_provenance",
}
CLAIM_KEYS = {"text", "supporting_evidence_ids", "supported"}
CHAT_RESPONSE_KEYS = {
    "answer",
    "grounded",
    "groundedness",
    "citations",
    "evidence",
    "claims",
    "trace_id",
    "langsmith_run_id",
}
CITATION_KEYS = {
    "url",
    "title",
    "chunk_id",
    "score",
    "section",
    "snippet",
    "nearby_context",
    "evidence_id",
    "parent_context_id",
    "dense_score",
    "sparse_score",
    "rerank_score",
}
RETRIEVAL_TRACE_KEYS = {"query_plan", "candidates", "parent_contexts", "trace"}
ERROR_ENVELOPE_KEYS = {"code", "message", "details", "retryable"}


# ---------------------------------------------------------------------------
# Drift-tolerant assertion helpers (U5)
# ---------------------------------------------------------------------------


def assert_keys(obj: dict[str, Any], expected: set[str], *, label: str = "object") -> None:
    """Require exact key set — catches renamed/removed public fields."""
    assert set(obj.keys()) == expected, f"{label} keys mismatch: {set(obj.keys())} != {expected}"


def assert_score_fields_present(obj: dict[str, Any], fields: set[str]) -> None:
    """Scores must exist and be numeric; exact values are intentionally not locked."""
    for field in fields:
        assert field in obj, f"missing score field {field!r}"
        assert isinstance(obj[field], (int, float)), f"{field!r} must be numeric"


def assert_expected_membership(
    items: list[dict[str, Any]],
    *,
    key: str,
    expected: set[str],
    top_k: int | None = None,
) -> None:
    """Expected IDs must appear somewhere in top-k, regardless of rank order."""
    pool = items[:top_k] if top_k is not None else items
    found = {item[key] for item in pool if key in item}
    assert expected <= found, f"expected {expected} within top-{top_k or 'all'}, found {found}"


def assert_trace_id_format(trace_id: str | None) -> None:
    assert trace_id is not None
    assert trace_id.startswith("trace_")


def build_retrieval_trace(result: RetrievalResult) -> dict[str, object]:
    """Mirror answer_workspace_question internal trace assembly."""
    return {
        "query_plan": result.query_plan.model_dump(),
        "candidates": [
            {
                "chunk_id": candidate.chunk.chunk_id,
                "dense_score": candidate.dense_score,
                "sparse_score": candidate.sparse_score,
                "fused_score": candidate.fused_score,
                "rerank_score": candidate.rerank_score,
                "subquery_provenance": candidate.subquery_provenance,
            }
            for candidate in result.candidates
        ],
        "parent_contexts": [parent.model_dump() for parent in result.parent_contexts],
        "trace": result.trace,
    }


# ---------------------------------------------------------------------------
# Fixture corpus loaders (U1)
# ---------------------------------------------------------------------------


def load_characterization_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def build_chunks_from_fixture(fixture: dict[str, Any], workspace_id: str | None = None) -> list[ChildChunkRecord]:
    workspace_id = workspace_id or workspace_store.workspace_id
    chunking_version = fixture["chunking_version"]
    return [
        ChildChunkRecord(
            chunk_id=chunk["chunk_id"],
            section_id=chunk["section_id"],
            page_id=chunk["page_id"],
            workspace_id=workspace_id,
            chunking_version=chunking_version,
            title=chunk["title"],
            url=chunk["url"],
            heading_path=chunk["heading_path"],
            text=chunk["text"],
            token_start=chunk["token_start"],
            token_end=chunk["token_end"],
        )
        for chunk in fixture["chunks"]
    ]


def build_embeddings_from_fixture(fixture: dict[str, Any]) -> dict[str, list[float]]:
    return {chunk_id: list(vector) for chunk_id, vector in fixture["embeddings"].items()}


class FixtureEmbeddingClient:
    """Offline embedding client driven by fixture query_embeddings."""

    def __init__(self, fixture: dict[str, Any] | None = None, **kwargs) -> None:
        fixture = fixture or load_characterization_fixture()
        self._query_embeddings = fixture["query_embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        if "alpha" in lowered:
            return list(self._query_embeddings["alpha"])
        if "beta" in lowered:
            return list(self._query_embeddings["beta"])
        if "pricing" in lowered or "plans" in lowered:
            return list(self._query_embeddings["pricing"])
        return list(self._query_embeddings["default"])


def seed_characterization_workspace(fixture: dict[str, Any] | None = None) -> str:
    fixture = fixture or load_characterization_fixture()
    workspace_id = workspace_store.workspace_id
    run = workspace_store.start_ingest_run(
        seed_url=fixture["pages"][0]["canonical_url"],
        hostname="example.com",
        registrable_domain="example.com",
        included_subdomains=["example.com"],
        chunking_version=fixture["chunking_version"],
        embedding_version=fixture["embedding_version"],
    )
    assert run is not None

    pages = [
        PageVersionRecord(
            page_id=page["page_id"],
            workspace_id=workspace_id,
            run_id=run.run_id,
            canonical_url=page["canonical_url"],
            discovered_url=page["discovered_url"],
            title=page["title"],
            acquisition_method=AcquisitionMethod.HTML,
            content_hash=f"hash_{page['page_id']}",
            quality_score=1.0,
            quality_signals={},
            boilerplate_removed=[],
            clean_text=page["clean_text"],
        )
        for page in fixture["pages"]
    ]
    sections = [
        ParentSectionRecord(
            section_id=section["section_id"],
            page_id=section["page_id"],
            workspace_id=workspace_id,
            heading_path=section["heading_path"],
            section_index=section["section_index"],
            text=section["text"],
            start_offset=section["start_offset"],
            end_offset=section["end_offset"],
        )
        for section in fixture["sections"]
    ]
    chunks = build_chunks_from_fixture(fixture)
    embeddings = build_embeddings_from_fixture(fixture)

    workspace_store.replace_active_content(
        run_id=run.run_id,
        pages=pages,
        sections=sections,
        chunks=chunks,
        embeddings=embeddings,
        embedding_version=fixture["embedding_version"],
        rendered_fallback_count=0,
        skipped_count=0,
    )
    return run.run_id


def retrieval_settings() -> Settings:
    return Settings(retrieval_candidate_limit=5, rerank_limit=3, parent_context_limit=2)


def run_retrieval(question: str, fixture: dict[str, Any] | None = None) -> RetrievalResult:
    fixture = fixture or load_characterization_fixture()
    return RetrievalPipeline(
        settings=retrieval_settings(),
        embedding_client=FixtureEmbeddingClient(fixture),
        chunks=build_chunks_from_fixture(fixture),
        embeddings=build_embeddings_from_fixture(fixture),
    ).retrieve(question)


# ---------------------------------------------------------------------------
# Fake generation clients for /chat contract tests (U3)
# ---------------------------------------------------------------------------


class GroundedGenerationClient:
    def __init__(self, *, api_key=None, model="fake", **kwargs) -> None:
        pass

    def generate_answer_from_evidence(self, *, question, evidence):
        first = evidence[0]
        return GeneratedAnswer(
            answer="Alpha setup is documented in the indexed pages.",
            grounded=True,
            groundedness="grounded",
            claims=[
                {
                    "text": "Alpha setup is documented in the indexed pages.",
                    "supporting_evidence_ids": [first.evidence_id],
                    "supported": True,
                }
            ],
            supporting_evidence_ids=[first.evidence_id],
        )


class PartiallyGroundedGenerationClient:
    def __init__(self, *, api_key=None, model="fake", **kwargs) -> None:
        pass

    def generate_answer_from_evidence(self, *, question, evidence):
        first = evidence[0]
        return GeneratedAnswer(
            answer="Partial answer with mixed claims.",
            grounded=True,
            groundedness="partially_grounded",
            claims=[
                {
                    "text": "Supported claim about alpha setup.",
                    "supporting_evidence_ids": [first.evidence_id],
                    "supported": True,
                },
                {
                    "text": "Unsupported claim about unrelated topic.",
                    "supporting_evidence_ids": [],
                    "supported": False,
                },
            ],
            supporting_evidence_ids=[first.evidence_id],
        )


class NotGroundedGenerationClient:
    def __init__(self, *, api_key=None, model="fake", **kwargs) -> None:
        pass

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


# ---------------------------------------------------------------------------
# U1 — Fixture corpus validation
# ---------------------------------------------------------------------------


def test_fixture_loads_valid_chunks_and_embeddings() -> None:
    fixture = load_characterization_fixture()
    chunks = build_chunks_from_fixture(fixture)
    embeddings = build_embeddings_from_fixture(fixture)

    assert 3 <= len(chunks) <= 5
    assert len({chunk.section_id for chunk in chunks}) >= 2
    assert len({chunk.url for chunk in chunks}) >= 2
    assert set(embeddings) == {chunk.chunk_id for chunk in chunks}


def test_fixture_has_same_section_pair_for_parent_context() -> None:
    fixture = load_characterization_fixture()
    chunks = build_chunks_from_fixture(fixture)
    by_section: dict[str, list[str]] = {}
    for chunk in chunks:
        by_section.setdefault(chunk.section_id, []).append(chunk.chunk_id)
    assert any(len(ids) >= 2 for ids in by_section.values())


def test_fixture_has_compound_and_no_match_questions() -> None:
    fixture = load_characterization_fixture()
    assert fixture["questions"]["compound"]["expect_decomposed"] is True
    assert fixture["questions"]["no_match"]["expected_chunk_ids"] == []


# ---------------------------------------------------------------------------
# U2 — RetrievalResult and retrieval trace shape
# ---------------------------------------------------------------------------


def test_retrieval_result_top_level_shape() -> None:
    fixture = load_characterization_fixture()
    question = fixture["questions"]["happy_path"]["text"]
    result = run_retrieval(question, fixture)
    dumped = result.model_dump()

    assert_keys(dumped, RETRIEVAL_RESULT_KEYS, label="RetrievalResult")


def test_retrieval_nested_model_shapes() -> None:
    fixture = load_characterization_fixture()
    result = run_retrieval(fixture["questions"]["happy_path"]["text"], fixture)

    assert_keys(result.query_plan.model_dump(), QUERY_PLAN_KEYS, label="QueryPlan")
    assert result.candidates, "expected non-empty candidates on happy path"
    candidate_dump = result.candidates[0].model_dump()
    assert_keys(candidate_dump, RETRIEVAL_CANDIDATE_KEYS, label="RetrievalCandidate")
    assert_score_fields_present(candidate_dump, CANDIDATE_SCORE_KEYS | {"fused_score"})
    assert isinstance(candidate_dump["subquery_provenance"], list)

    assert result.evidence, "expected non-empty evidence on happy path"
    assert_keys(result.evidence[0].model_dump(), EVIDENCE_SNIPPET_KEYS, label="EvidenceSnippet")
    assert_score_fields_present(result.evidence[0].model_dump(), CANDIDATE_SCORE_KEYS)

    assert result.parent_contexts, "expected parent contexts on happy path"
    assert_keys(result.parent_contexts[0].model_dump(), PARENT_CONTEXT_KEYS, label="ParentContext")


def test_retrieval_happy_path_membership() -> None:
    fixture = load_characterization_fixture()
    q = fixture["questions"]["happy_path"]
    result = run_retrieval(q["text"], fixture)
    top_k = retrieval_settings().rerank_limit
    expected = set(q["expected_chunk_ids"])

    assert result.evidence
    evidence_ids = {item.chunk_id for item in result.evidence[:top_k]}
    candidate_ids = {candidate.chunk.chunk_id for candidate in result.candidates[:top_k]}
    found_ids = evidence_ids | candidate_ids
    # Same-section dedupe keeps the strongest chunk; either alpha chunk satisfies membership.
    assert expected & found_ids, f"expected overlap with {expected}, found {found_ids}"
    found_urls = {item.url for item in result.evidence}
    assert set(q["expected_urls"]) <= found_urls


def test_retrieval_compound_question_decomposes() -> None:
    fixture = load_characterization_fixture()
    q = fixture["questions"]["compound"]
    result = run_retrieval(q["text"], fixture)

    assert result.query_plan.decomposed is True
    assert 1 < len(result.query_plan.subqueries) <= Settings().decomposition_max_subqueries
    evidence_ids = {item.chunk_id for item in result.evidence}
    assert set(q["expected_chunk_ids"]) & evidence_ids


def test_retrieval_parent_context_fields() -> None:
    fixture = load_characterization_fixture()
    result = run_retrieval(fixture["questions"]["happy_path"]["text"], fixture)

    evidence = result.evidence[0]
    assert evidence.parent_context_id
    assert evidence.nearby_context
    matching = [
        parent
        for parent in result.parent_contexts
        if parent.parent_context_id == evidence.parent_context_id
    ]
    assert matching
    assert matching[0].evidence_ids


def test_retrieval_dedupes_same_section_evidence() -> None:
    fixture = load_characterization_fixture()
    result = run_retrieval(fixture["questions"]["happy_path"]["text"], fixture)

    section_ids = [item.section_id for item in result.evidence]
    assert len(section_ids) == len(set(section_ids))
    # Returned candidates are post-dedupe; trace keeps pre-dedupe candidate_count.
    assert result.trace["candidate_count"] >= result.trace["deduped_count"]
    assert result.trace["deduped_count"] >= result.trace["evidence_count"]
    assert result.trace["candidate_count"] >= 2
    assert result.trace["deduped_count"] == 1


def test_retrieval_no_match_preserves_shape() -> None:
    fixture = load_characterization_fixture()
    result = run_retrieval(fixture["questions"]["no_match"]["text"], fixture)

    assert_keys(result.model_dump(), RETRIEVAL_RESULT_KEYS, label="RetrievalResult")
    assert result.evidence == []
    assert isinstance(result.query_plan, QueryPlan)


def test_retrieval_trace_compatible_with_workspace_rag() -> None:
    fixture = load_characterization_fixture()
    result = run_retrieval(fixture["questions"]["happy_path"]["text"], fixture)
    trace = build_retrieval_trace(result)

    assert_keys(trace, RETRIEVAL_TRACE_KEYS, label="retrieval_trace")
    assert_keys(trace["query_plan"], QUERY_PLAN_KEYS, label="retrieval_trace.query_plan")
    assert trace["candidates"]
    for candidate in trace["candidates"]:
        assert "chunk_id" in candidate
        assert_score_fields_present(candidate, CANDIDATE_SCORE_KEYS | {"fused_score"})


# ---------------------------------------------------------------------------
# U3 — /chat public success contract
# ---------------------------------------------------------------------------


@pytest.fixture
def chat_client(monkeypatch):
    monkeypatch.setattr("app.api.chat.LangChainEmbeddingClient", FixtureEmbeddingClient)
    return TestClient(app)


def test_chat_grounded_response_shape(chat_client, monkeypatch) -> None:
    fixture = load_characterization_fixture()
    monkeypatch.setattr("app.api.chat.LangChainGenerationClient", GroundedGenerationClient)
    seed_characterization_workspace(fixture)

    response = chat_client.post(
        "/chat",
        json={"question": fixture["questions"]["happy_path"]["text"], "session_id": "char"},
    )

    assert response.status_code == 200
    body = response.json()
    assert_keys(body, CHAT_RESPONSE_KEYS, label="ChatResponse")
    assert "retrieval_trace" not in body
    assert body["groundedness"] == "grounded"
    assert body["grounded"] is True
    assert body["citations"]
    assert_keys(body["citations"][0], CITATION_KEYS, label="Citation")
    citation = body["citations"][0]
    assert citation["evidence_id"]
    assert citation["parent_context_id"]
    assert citation["snippet"]
    assert citation["nearby_context"] is not None
    assert_score_fields_present(citation, {"score", "dense_score", "sparse_score", "rerank_score"})
    assert body["evidence"]
    assert_keys(body["evidence"][0], EVIDENCE_SNIPPET_KEYS, label="ChatResponse.evidence")
    assert_trace_id_format(body["trace_id"])


def test_chat_partially_grounded_response(chat_client, monkeypatch) -> None:
    fixture = load_characterization_fixture()
    monkeypatch.setattr("app.api.chat.LangChainGenerationClient", PartiallyGroundedGenerationClient)
    seed_characterization_workspace(fixture)

    response = chat_client.post(
        "/chat",
        json={"question": fixture["questions"]["happy_path"]["text"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["groundedness"] == "partially_grounded"
    assert body["grounded"] is False
    assert len(body["claims"]) == 2
    for claim in body["claims"]:
        assert_keys(claim, CLAIM_KEYS, label="ChatResponse.claim")
    assert body["claims"][0]["supported"] is True
    assert body["claims"][1]["supported"] is False
    assert body["citations"]
    assert body["citations"][0]["evidence_id"] == body["claims"][0]["supporting_evidence_ids"][0]
    assert "retrieval_trace" not in body


def test_chat_not_grounded_response(chat_client, monkeypatch) -> None:
    fixture = load_characterization_fixture()
    monkeypatch.setattr("app.api.chat.LangChainGenerationClient", NotGroundedGenerationClient)
    seed_characterization_workspace(fixture)

    response = chat_client.post(
        "/chat",
        json={"question": fixture["questions"]["happy_path"]["text"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == NOT_FOUND_ANSWER
    assert body["grounded"] is False
    assert body["groundedness"] == "not_grounded"
    assert body["citations"] == []
    assert body["evidence"] == []
    assert body["claims"]
    assert "retrieval_trace" not in body


def test_chat_trace_id_and_langsmith_fields(chat_client, monkeypatch) -> None:
    fixture = load_characterization_fixture()
    monkeypatch.setattr("app.api.chat.LangChainGenerationClient", GroundedGenerationClient)
    seed_characterization_workspace(fixture)

    response = chat_client.post(
        "/chat",
        json={"question": fixture["questions"]["happy_path"]["text"]},
    )

    body = response.json()
    assert_trace_id_format(body["trace_id"])
    assert body["langsmith_run_id"] is None or isinstance(body["langsmith_run_id"], str)


# ---------------------------------------------------------------------------
# U4 — Error codes, enums, and fallback constants
# ---------------------------------------------------------------------------


def test_groundedness_enum_values() -> None:
    assert {item.value for item in Groundedness} == {
        "grounded",
        "partially_grounded",
        "not_grounded",
    }


def test_not_found_answer_constant_matches_generation() -> None:
    assert NOT_FOUND_ANSWER == GENERATION_NOT_FOUND_ANSWER
    assert NOT_FOUND_ANSWER == (
        "The indexed site content does not contain enough information to answer that."
    )


def test_chat_before_ready_error_envelope() -> None:
    client = TestClient(app)
    response = client.post("/chat", json={"question": "What is alpha?"})

    assert response.status_code == 409
    body = response.json()
    assert "error" in body
    assert_keys(body["error"], ERROR_ENVELOPE_KEYS, label="error")
    assert body["error"]["code"] == "CHAT_BEFORE_READY"
    assert body["error"]["retryable"] is True


def test_chat_missing_api_key_error_envelope() -> None:
    seed_characterization_workspace()
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"question": load_characterization_fixture()["questions"]["happy_path"]["text"]},
    )

    assert response.status_code == 503
    body = response.json()
    assert_keys(body["error"], ERROR_ENVELOPE_KEYS, label="error")
    assert body["error"]["code"] == "MISSING_API_KEY"
    assert body["error"]["retryable"] is True


# ---------------------------------------------------------------------------
# U5 — Guardrail helper behavior (meta-tests for drift policy)
# ---------------------------------------------------------------------------


def test_membership_helper_allows_any_rank_within_top_k() -> None:
    items = [{"chunk_id": "c"}, {"chunk_id": "b"}, {"chunk_id": "a"}]
    assert_expected_membership(items, key="chunk_id", expected={"b"}, top_k=3)


def test_shape_helper_detects_missing_field() -> None:
    with pytest.raises(AssertionError, match="keys mismatch"):
        assert_keys({"a": 1}, {"a", "b"}, label="test")


def test_shape_helper_detects_missing_evidence_snippet_field() -> None:
    fixture = load_characterization_fixture()
    result = run_retrieval(fixture["questions"]["happy_path"]["text"], fixture)
    incomplete = result.evidence[0].model_dump()
    del incomplete["rerank_score"]
    with pytest.raises(AssertionError, match="keys mismatch"):
        assert_keys(incomplete, EVIDENCE_SNIPPET_KEYS, label="EvidenceSnippet")


def test_score_helper_requires_numeric_not_exact_value() -> None:
    assert_score_fields_present(
        {"dense_score": 0.42, "sparse_score": 0.0, "rerank_score": 0.1},
        CANDIDATE_SCORE_KEYS,
    )
