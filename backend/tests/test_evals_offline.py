"""Offline deterministic eval tests — CI gate, no network, no secrets."""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.evals.correctness import judge_correctness
from app.rag.service import NOT_FOUND_ANSWER, ChatResponse, answer_workspace_question
from app.workspace import Groundedness
from evals.config import QA_DATASET_NAME, RETRIEVAL_DATASET_NAME, SITES, eval_settings
from evals.langsmith_eval import (
    RETRIEVAL_THRESHOLDS,
    SiteEvalContext,
    decomposition_evaluator,
    hit_evaluator,
    make_correctness_evaluator,
    make_faithfulness_evaluator,
    make_retrieval_target,
    mrr_evaluator,
    recall_at_k_evaluator,
    run_offline_retrieval_eval,
)
from evals.loader import (
    FixtureQueryEmbeddingClient,
    build_site_chunks,
    ephemeral_workspace,
    load_doc_embeddings,
    load_manifest,
)
from evals.schema import load_qa_dataset, load_retrieval_dataset
from evals.upload_datasets import _sync_qa_dataset, _sync_retrieval_dataset


class FakeJudgeClient:
    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload or {
            "faithful": True,
            "score": 0.9,
            "reasoning": "supported by evidence",
            "unsupported_claims": [],
            "correct": True,
        }
        self.prompts: list[str] = []

    def generate_raw(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(self._payload)


def test_retrieval_evaluators_are_deterministic() -> None:
    run = SimpleNamespace(
        outputs={
            "recall_at_k": 1.0,
            "reciprocal_rank": 1.0,
            "hit": True,
            "decomposition_matched": True,
        }
    )
    example = SimpleNamespace(inputs={}, outputs={})

    for evaluator in (
        recall_at_k_evaluator,
        mrr_evaluator,
        hit_evaluator,
        decomposition_evaluator,
    ):
        first = evaluator(run, example)
        second = evaluator(run, example)
        assert first == second

    assert recall_at_k_evaluator(run, example) == {"key": "recall_at_k", "score": 1.0}
    assert mrr_evaluator(run, example) == {"key": "mrr", "score": 1.0}
    assert hit_evaluator(run, example) == {"key": "hit", "score": 1.0}
    assert decomposition_evaluator(run, example) == {"key": "decomposition", "score": 1.0}


def test_retrieval_evaluators_score_mismatch_cases() -> None:
    run = SimpleNamespace(
        outputs={
            "recall_at_k": 0.0,
            "reciprocal_rank": 0.0,
            "hit": False,
            "decomposition_matched": False,
        }
    )
    example = SimpleNamespace(inputs={}, outputs={})
    assert hit_evaluator(run, example)["score"] == 0.0
    assert decomposition_evaluator(run, example)["score"] == 0.0


def test_retrieval_target_scores_against_golden_labels() -> None:
    """Regression: target must pass expected_chunk_ids into run_retrieval_eval."""
    manifest = load_manifest()
    case = next(
        c
        for c in load_retrieval_dataset()
        if c.site == "acme_docs" and c.expected_chunk_ids
    )
    chunks = build_site_chunks(case.site)[2]
    embeddings = load_doc_embeddings(case.site)
    settings = eval_settings(
        sqlite_path=":memory:",
        qdrant_path=":memory:",
        chunking_version=manifest["chunking_version"],
        embedding_model=manifest["embedding_model"],
    )
    contexts = {
        case.site: SiteEvalContext(
            site=case.site,
            settings=settings,
            chunks=chunks,
            embeddings=embeddings,
            embedding_client=FixtureQueryEmbeddingClient(),
            generation_client=None,  # type: ignore[arg-type]
            qdrant_path=":memory:",
        ),
    }
    target = make_retrieval_target(contexts)
    outputs = target({"question": case.question, "site": case.site})
    assert outputs["hit"] is True
    assert outputs["recall_at_k"] > 0.0
    assert outputs["reciprocal_rank"] > 0.0


def test_retrieval_target_without_golden_labels_scores_zero() -> None:
    from app.evals import EvalCase, run_retrieval_eval

    manifest = load_manifest()
    case = next(c for c in load_retrieval_dataset() if c.site == "acme_docs")
    chunks = build_site_chunks(case.site)[2]
    embeddings = load_doc_embeddings(case.site)
    settings = eval_settings(
        sqlite_path=":memory:",
        qdrant_path=":memory:",
        chunking_version=manifest["chunking_version"],
        embedding_model=manifest["embedding_model"],
    )
    metrics = run_retrieval_eval(
        cases=[EvalCase(question=case.question)],
        settings=settings,
        embedding_client=FixtureQueryEmbeddingClient(),
        chunks=chunks,
        embeddings=embeddings,
    )
    assert metrics["results"][0]["hit"] is False
    assert metrics["results"][0]["recall_at_k"] == 0.0


def test_offline_retrieval_eval_meets_golden_thresholds() -> None:
    metrics = run_offline_retrieval_eval()
    assert metrics["case_count"] > 0
    for key, floor in RETRIEVAL_THRESHOLDS.items():
        assert metrics[key] >= floor, f"{key}={metrics[key]} below floor {floor}"


def test_qa_chat_response_shape_in_corpus() -> None:
    case = next(
        case
        for case in load_qa_dataset()
        if case.site == "acme_docs" and case.expected_groundedness == "grounded"
    )
    with ephemeral_workspace(case.site) as handle:
        response = answer_workspace_question(
            question=case.question,
            settings=handle.settings,
            chunks=handle.chunks,
            embeddings=handle.embeddings,
            embedding_client=handle.query_client,
            generation_client=None,
        )

    assert isinstance(response, ChatResponse)
    assert response.grounded is True
    assert response.citations
    for citation in response.citations:
        assert citation.evidence_id
        assert citation.dense_score is not None
        assert citation.sparse_score is not None
        assert citation.rerank_score is not None


def test_qa_chat_response_empty_chunks_returns_not_found() -> None:
    with ephemeral_workspace("acme_docs") as handle:
        response = answer_workspace_question(
            question="How do I set up the Acme desktop agent?",
            settings=handle.settings,
            chunks=[],
            embeddings=handle.embeddings,
            embedding_client=handle.query_client,
            generation_client=None,
        )

    assert response.answer == NOT_FOUND_ANSWER
    assert response.groundedness == Groundedness.NOT_GROUNDED
    assert response.citations == []


def test_faithfulness_evaluator_wiring_with_fake_client() -> None:
    fake = FakeJudgeClient()
    evaluator = make_faithfulness_evaluator(fake)
    run = SimpleNamespace(
        outputs={
            "answer": "alpha content",
            "evidence_snippets": ["alpha content"],
        }
    )
    example = SimpleNamespace(inputs={"question": "What is alpha?"}, outputs={})
    result = evaluator(run, example)
    assert result["key"] == "faithfulness"
    assert result["score"] == 0.9
    assert result["value"] is True
    assert result["comment"] == "supported by evidence"


def test_faithfulness_evaluator_short_circuits_decline() -> None:
    evaluator = make_faithfulness_evaluator(FakeJudgeClient())
    run = SimpleNamespace(
        outputs={
            "answer": NOT_FOUND_ANSWER,
            "evidence_snippets": [],
        }
    )
    example = SimpleNamespace(inputs={"question": "Unknown?"}, outputs={})
    result = evaluator(run, example)
    assert result == {
        "key": "faithfulness",
        "score": 1.0,
        "value": True,
        "comment": "declined; no evidence",
    }


def test_correctness_evaluator_wiring_with_fake_client() -> None:
    fake = FakeJudgeClient({"correct": True, "score": 0.85, "reasoning": "semantically equivalent"})
    evaluator = make_correctness_evaluator(fake)
    run = SimpleNamespace(outputs={"answer": "Twenty four hours."})
    example = SimpleNamespace(
        inputs={"question": "How long?"},
        outputs={"expected_answer": "Tokens expire after twenty four hours."},
    )
    result = evaluator(run, example)
    assert result["key"] == "correctness"
    assert result["score"] == 0.85
    assert result["value"] is True


def test_judge_correctness_happy_path() -> None:
    result = judge_correctness(
        question="How long?",
        generated_answer="Twenty four hours.",
        expected_answer="Tokens expire after twenty four hours.",
        generation_client=FakeJudgeClient({"correct": True, "score": 0.9, "reasoning": "match"}),
    )
    assert result.correct is True
    assert result.score == 0.9


def test_judge_correctness_clamps_score() -> None:
    high = judge_correctness(
        question="q",
        generated_answer="a",
        expected_answer="b",
        generation_client=FakeJudgeClient({"correct": True, "score": 1.4, "reasoning": "high"}),
    )
    low = judge_correctness(
        question="q",
        generated_answer="a",
        expected_answer="b",
        generation_client=FakeJudgeClient({"correct": False, "score": -0.2, "reasoning": "low"}),
    )
    assert high.score == 1.0
    assert low.score == 0.0


def test_judge_correctness_generation_error_fallback() -> None:
    class BrokenClient:
        def generate_raw(self, prompt: str) -> str:
            raise RuntimeError("boom")

    result = judge_correctness(
        question="q",
        generated_answer="a",
        expected_answer="b",
        generation_client=BrokenClient(),
    )
    assert result.correct is False
    assert result.score == 0.0
    assert result.reasoning == "Failed to evaluate: generation error."


def test_judge_correctness_out_of_corpus_reference() -> None:
    result = judge_correctness(
        question="What is the ticker?",
        generated_answer=NOT_FOUND_ANSWER,
        expected_answer="Not covered in the indexed corpus.",
        generation_client=FakeJudgeClient({"correct": True, "score": 1.0, "reasoning": "declined"}),
    )
    assert result.correct is True
    assert result.score == 1.0


def test_sites_matches_manifest() -> None:
    manifest = load_manifest()
    manifest_sites = [entry["site"] for entry in manifest["sites"]]
    assert SITES == manifest_sites
    assert RETRIEVAL_DATASET_NAME
    assert QA_DATASET_NAME


class _FakeDataset:
    def __init__(self, dataset_id: str) -> None:
        self.id = dataset_id


class FakeLangSmithClient:
    def __init__(self) -> None:
        self.datasets: dict[str, _FakeDataset] = {}
        self.examples: dict[str, list[dict]] = {}
        self.deleted: list[str] = []

    def read_dataset(self, *, dataset_name: str) -> _FakeDataset:
        if dataset_name not in self.datasets:
            raise LookupError(dataset_name)
        return self.datasets[dataset_name]

    def create_dataset(self, *, dataset_name: str) -> _FakeDataset:
        dataset = _FakeDataset(f"ds-{dataset_name}")
        self.datasets[dataset_name] = dataset
        self.examples[dataset.id] = []
        return dataset

    def delete_dataset(self, *, dataset_id: str) -> None:
        self.deleted.append(dataset_id)
        for name, dataset in list(self.datasets.items()):
            if dataset.id == dataset_id:
                del self.datasets[name]
                del self.examples[dataset_id]

    def list_examples(self, *, dataset_id: str):
        for item in self.examples.get(dataset_id, []):
            yield SimpleNamespace(inputs=item["inputs"], outputs=item.get("outputs"))

    def create_examples(self, *, dataset_id: str, examples: list[dict]) -> None:
        self.examples.setdefault(dataset_id, []).extend(examples)


def test_upload_datasets_creates_expected_examples() -> None:
    client = FakeLangSmithClient()
    retrieval_created, retrieval_skipped = _sync_retrieval_dataset(client, recreate=False)
    qa_created, qa_skipped = _sync_qa_dataset(client, recreate=False)

    from evals.schema import load_retrieval_dataset as load_retrieval

    assert retrieval_created == len(load_retrieval())
    assert retrieval_skipped == 0
    assert qa_created == len(load_qa_dataset())
    assert qa_skipped == 0

    retrieval_dataset = client.datasets[RETRIEVAL_DATASET_NAME]
    sample = client.examples[retrieval_dataset.id][0]
    assert set(sample["inputs"].keys()) == {"question", "site"}
    assert set(sample["outputs"].keys()) == {
        "expected_chunk_ids",
        "equivalent_chunk_ids",
        "expected_urls",
        "should_decompose",
    }


def test_upload_datasets_is_idempotent() -> None:
    client = FakeLangSmithClient()
    _sync_retrieval_dataset(client, recreate=False)
    _sync_qa_dataset(client, recreate=False)
    retrieval_created, retrieval_skipped = _sync_retrieval_dataset(client, recreate=False)
    qa_created, qa_skipped = _sync_qa_dataset(client, recreate=False)
    assert retrieval_created == 0
    assert qa_created == 0
    assert retrieval_skipped > 0
    assert qa_skipped > 0

