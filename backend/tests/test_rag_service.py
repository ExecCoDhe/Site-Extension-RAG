"""Unit tests for WorkspaceRagPipeline LCEL composition (Milestone C4)."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.rag.service import NOT_FOUND_ANSWER, WorkspaceRagPipeline, answer_workspace_question
from tests.test_characterization_contract import (
    CITATION_KEYS,
    RETRIEVAL_TRACE_KEYS,
    FixtureEmbeddingClient,
    GroundedGenerationClient,
    NotGroundedGenerationClient,
    assert_keys,
    assert_score_fields_present,
    build_chunks_from_fixture,
    build_embeddings_from_fixture,
    load_characterization_fixture,
)


def _pipeline_settings(tmp_path: Path) -> Settings:
    return Settings(
        retrieval_candidate_limit=5,
        rerank_limit=3,
        parent_context_limit=2,
        qdrant_path=str(tmp_path / "qdrant"),
    )


def _make_pipeline_kwargs(fixture, tmp_path: Path, *, generation_client, chunks=None):
    chunks = chunks if chunks is not None else build_chunks_from_fixture(fixture)
    return {
        "settings": _pipeline_settings(tmp_path),
        "chunks": chunks,
        "embeddings": build_embeddings_from_fixture(fixture),
        "embedding_client": FixtureEmbeddingClient(fixture),
        "generation_client": generation_client,
    }


def test_empty_chunks_returns_not_found(tmp_path) -> None:
    fixture = load_characterization_fixture()
    kwargs = _make_pipeline_kwargs(fixture, tmp_path, generation_client=GroundedGenerationClient())
    kwargs["chunks"] = []

    response = WorkspaceRagPipeline(**kwargs).answer("What is alpha?")

    assert response.answer == NOT_FOUND_ANSWER
    assert response.grounded is False
    assert response.groundedness == "not_grounded"
    assert response.citations == []


def test_no_evidence_returns_not_found(tmp_path) -> None:
    fixture = load_characterization_fixture()
    kwargs = _make_pipeline_kwargs(fixture, tmp_path, generation_client=GroundedGenerationClient())
    question = fixture["questions"]["no_match"]["text"]

    response = WorkspaceRagPipeline(**kwargs).answer(question)

    assert response.answer == NOT_FOUND_ANSWER
    assert response.grounded is False
    assert response.groundedness == "not_grounded"
    assert response.citations == []


def test_grounded_answer_builds_citations_and_trace(tmp_path) -> None:
    fixture = load_characterization_fixture()
    kwargs = _make_pipeline_kwargs(fixture, tmp_path, generation_client=GroundedGenerationClient())
    question = fixture["questions"]["happy_path"]["text"]

    response = WorkspaceRagPipeline(**kwargs).answer(question)

    assert response.grounded is True
    assert response.groundedness == "grounded"
    assert response.citations
    citation = response.citations[0].model_dump()
    assert_keys(citation, CITATION_KEYS, label="Citation")
    assert citation["evidence_id"]
    assert citation["parent_context_id"]
    assert_score_fields_present(
        citation, {"score", "dense_score", "sparse_score", "rerank_score"}
    )
    assert_keys(response.retrieval_trace, RETRIEVAL_TRACE_KEYS, label="retrieval_trace")


def test_unsupported_claims_downgrade_to_not_grounded(tmp_path) -> None:
    fixture = load_characterization_fixture()
    kwargs = _make_pipeline_kwargs(
        fixture, tmp_path, generation_client=NotGroundedGenerationClient()
    )
    question = fixture["questions"]["happy_path"]["text"]

    response = WorkspaceRagPipeline(**kwargs).answer(question)

    assert response.groundedness == "not_grounded"
    assert response.citations == []
    assert response.evidence == []
    assert response.claims


class DeterministicFallbackClient:
    """Client without generate_answer_from_evidence — triggers _deterministic_evidence_answer."""


def test_deterministic_fallback_when_no_generation_method(tmp_path) -> None:
    fixture = load_characterization_fixture()
    kwargs = _make_pipeline_kwargs(
        fixture, tmp_path, generation_client=DeterministicFallbackClient()
    )
    question = fixture["questions"]["happy_path"]["text"]

    response = WorkspaceRagPipeline(**kwargs).answer(question)

    assert response.grounded is True
    assert response.groundedness == "grounded"
    assert response.citations
    assert response.citations[0].evidence_id == response.evidence[0].evidence_id
    assert response.answer == response.evidence[0].snippet


def test_wrapper_delegates_to_pipeline(tmp_path) -> None:
    fixture = load_characterization_fixture()
    kwargs = _make_pipeline_kwargs(fixture, tmp_path, generation_client=GroundedGenerationClient())
    question = fixture["questions"]["happy_path"]["text"]

    wrapper_response = answer_workspace_question(question=question, **kwargs)
    pipeline_response = WorkspaceRagPipeline(**kwargs).answer(question)

    assert wrapper_response.model_dump() == pipeline_response.model_dump()
