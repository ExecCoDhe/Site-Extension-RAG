"""LangSmith evaluators, targets, offline aggregate, and manual online runner."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from langsmith import Client, evaluate

from app.config import Settings, get_settings
from app.evals import judge_correctness, run_retrieval_eval
from app.evals.faithfulness import evaluate_faithfulness
from app.index.embeddings import LangChainEmbeddingClient
from app.rag.generation import LangChainGenerationClient, RawGenerationClient
from app.rag.service import NOT_FOUND_ANSWER, answer_workspace_question
from app.retrieval.vector_store import QdrantVectorStore
from app.workspace.models import ChildChunkRecord
from evals.config import (
    EVAL_WORKSPACE_ID,
    EXPERIMENT_PREFIX,
    QA_DATASET_NAME,
    RETRIEVAL_DATASET_NAME,
    SITES,
    eval_settings,
)
from evals.loader import (
    FixtureQueryEmbeddingClient,
    build_site_chunks,
    load_doc_embeddings,
    load_manifest,
)
from evals.schema import load_retrieval_dataset

RETRIEVAL_THRESHOLDS = {
    "hit_rate": 0.90,
    "recall_at_k": 0.80,
    "mrr": 0.60,
    "decomposition_accuracy": 0.80,
}

ONLINE_THRESHOLDS = {
    "faithful_rate": 0.90,
    "avg_faithfulness": 0.80,
    "correctness": 0.70,
}


@dataclass
class SiteEvalContext:
    site: str
    settings: Settings
    chunks: list[ChildChunkRecord]
    embeddings: dict[str, list[float]]
    embedding_client: LangChainEmbeddingClient | FixtureQueryEmbeddingClient
    generation_client: LangChainGenerationClient
    qdrant_path: str


def _require_online_env(settings: Settings | None = None) -> Settings:
    resolved = settings or get_settings()
    missing: list[str] = []
    if not resolved.langsmith_api_key:
        missing.append("LANGSMITH_API_KEY")
    if not resolved.gemini_api_key:
        missing.append("GEMINI_API_KEY")
    if missing:
        print(
            f"Missing required environment: {', '.join(missing)}. "
            "Online eval makes paid network calls and is never run in CI.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    os.environ["LANGSMITH_API_KEY"] = resolved.langsmith_api_key  # type: ignore[assignment]
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = resolved.langsmith_project
    return resolved


def build_online_site_contexts(stack: ExitStack) -> dict[str, SiteEvalContext]:
    manifest = load_manifest()
    settings_base = get_settings()
    contexts: dict[str, SiteEvalContext] = {}

    for site in SITES:
        tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
        qdrant_path = os.path.join(tmp_dir, "qdrant")
        chunks = build_site_chunks(site)[2]
        embeddings = load_doc_embeddings(site)
        settings = eval_settings(
            sqlite_path=":memory:",
            qdrant_path=qdrant_path,
            chunking_version=manifest["chunking_version"],
            embedding_model=manifest["embedding_model"],
        )
        QdrantVectorStore(path=settings.qdrant_path).upsert_chunks(
            collection_name=f"workspace_{EVAL_WORKSPACE_ID}",
            chunks=chunks,
            embeddings=embeddings,
        )
        embedding_client = LangChainEmbeddingClient(
            api_key=settings_base.gemini_api_key,
            model=manifest["embedding_model"],
            timeout_seconds=settings_base.gemini_request_timeout_seconds,
        )
        generation_client = LangChainGenerationClient(
            api_key=settings_base.gemini_api_key,
            model=settings_base.gemini_chat_model,
            timeout_seconds=settings_base.gemini_request_timeout_seconds,
        )
        contexts[site] = SiteEvalContext(
            site=site,
            settings=settings,
            chunks=chunks,
            embeddings=embeddings,
            embedding_client=embedding_client,
            generation_client=generation_client,
            qdrant_path=qdrant_path,
        )
    return contexts


def make_retrieval_target(
    contexts: dict[str, SiteEvalContext],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    golden_by_key = {
        (case.site, case.question): case for case in load_retrieval_dataset()
    }

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs["question"]
        site = inputs["site"]
        ctx = contexts[site]
        golden = golden_by_key[(site, question)]
        metrics = run_retrieval_eval(
            cases=[golden.to_eval_case()],
            settings=ctx.settings,
            embedding_client=ctx.embedding_client,
            chunks=ctx.chunks,
            embeddings=ctx.embeddings,
        )
        return dict(metrics["results"][0])

    return target


def make_qa_target(
    contexts: dict[str, SiteEvalContext],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        site = inputs["site"]
        ctx = contexts[site]
        response = answer_workspace_question(
            question=inputs["question"],
            settings=ctx.settings,
            chunks=ctx.chunks,
            embeddings=ctx.embeddings,
            embedding_client=ctx.embedding_client,
            generation_client=ctx.generation_client,
        )
        return {
            "answer": response.answer,
            "groundedness": response.groundedness.value,
            "grounded": response.grounded,
            "evidence_snippets": [item.snippet for item in response.evidence],
            "citation_count": len(response.citations),
        }

    return target


def recall_at_k_evaluator(run: Any, example: Any) -> dict[str, Any]:
    del example
    return {"key": "recall_at_k", "score": float(run.outputs["recall_at_k"])}


def mrr_evaluator(run: Any, example: Any) -> dict[str, Any]:
    del example
    return {"key": "mrr", "score": float(run.outputs["reciprocal_rank"])}


def hit_evaluator(run: Any, example: Any) -> dict[str, Any]:
    del example
    return {"key": "hit", "score": 1.0 if run.outputs["hit"] else 0.0}


def decomposition_evaluator(run: Any, example: Any) -> dict[str, Any]:
    del example
    matched = run.outputs["decomposition_matched"]
    return {"key": "decomposition", "score": 1.0 if matched else 0.0}


def make_faithfulness_evaluator(
    generation_client: RawGenerationClient,
) -> Callable[[Any, Any], dict[str, Any]]:
    def evaluator(run: Any, example: Any) -> dict[str, Any]:
        evidence_snippets = run.outputs.get("evidence_snippets") or []
        answer = run.outputs.get("answer", "")
        if not evidence_snippets and answer == NOT_FOUND_ANSWER:
            return {
                "key": "faithfulness",
                "score": 1.0,
                "value": True,
                "comment": "declined; no evidence",
            }
        result = evaluate_faithfulness(
            question=example.inputs["question"],
            answer=answer,
            evidence_snippets=evidence_snippets,
            generation_client=generation_client,
        )
        return {
            "key": "faithfulness",
            "score": result.score,
            "value": result.faithful,
            "comment": result.reasoning,
        }

    return evaluator


def make_correctness_evaluator(
    generation_client: RawGenerationClient,
) -> Callable[[Any, Any], dict[str, Any]]:
    def evaluator(run: Any, example: Any) -> dict[str, Any]:
        result = judge_correctness(
            question=example.inputs["question"],
            generated_answer=run.outputs["answer"],
            expected_answer=example.outputs["expected_answer"],
            generation_client=generation_client,
        )
        return {
            "key": "correctness",
            "score": result.score,
            "value": result.correct,
            "comment": result.reasoning,
        }

    return evaluator


def run_offline_retrieval_eval() -> dict[str, float | int]:
    manifest = load_manifest()
    all_results: list[dict[str, Any]] = []

    for site in SITES:
        chunks = build_site_chunks(site)[2]
        embeddings = load_doc_embeddings(site)
        client = FixtureQueryEmbeddingClient()
        settings = eval_settings(
            sqlite_path=":memory:",
            qdrant_path=":memory:",
            chunking_version=manifest["chunking_version"],
            embedding_model=manifest["embedding_model"],
        )
        cases = [case.to_eval_case() for case in load_retrieval_dataset() if case.site == site]
        if not cases:
            continue
        metrics = run_retrieval_eval(
            cases=cases,
            settings=settings,
            embedding_client=client,
            chunks=chunks,
            embeddings=embeddings,
        )
        all_results.extend(metrics["results"])

    total = max(len(all_results), 1)
    return {
        "case_count": len(all_results),
        "hit_rate": sum(result["hit"] for result in all_results) / total,
        "recall_at_k": sum(result["recall_at_k"] for result in all_results) / total,
        "mrr": sum(result["reciprocal_rank"] for result in all_results) / total,
        "decomposition_accuracy": sum(result["decomposition_matched"] for result in all_results) / total,
    }


def _experiment_url(client: Client, experiment_name: str) -> str | None:
    try:
        project = client.read_project(project_name=experiment_name)
        return getattr(project, "url", None)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    del argv
    print("=" * 72)
    print("WARNING: langsmith_eval makes paid Gemini + LangSmith API calls.")
    print("Do not run this in CI. Upload datasets first with upload_datasets.")
    print("=" * 72)

    settings = _require_online_env()
    client = Client()

    with ExitStack() as stack:
        contexts = build_online_site_contexts(stack)
        judge = LangChainGenerationClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_chat_model,
            timeout_seconds=settings.gemini_request_timeout_seconds,
        )

        retrieval_result = evaluate(
            make_retrieval_target(contexts),
            data=RETRIEVAL_DATASET_NAME,
            evaluators=[
                recall_at_k_evaluator,
                mrr_evaluator,
                hit_evaluator,
                decomposition_evaluator,
            ],
            experiment_prefix=f"{EXPERIMENT_PREFIX}-retrieval",
            max_concurrency=1,
        )
        qa_result = evaluate(
            make_qa_target(contexts),
            data=QA_DATASET_NAME,
            evaluators=[
                make_faithfulness_evaluator(judge),
                make_correctness_evaluator(judge),
            ],
            experiment_prefix=f"{EXPERIMENT_PREFIX}-qa",
            max_concurrency=1,
        )

    retrieval_name = getattr(retrieval_result, "experiment_name", None) or str(retrieval_result)
    qa_name = getattr(qa_result, "experiment_name", None) or str(qa_result)
    print(f"Retrieval experiment: {retrieval_name}")
    retrieval_url = _experiment_url(client, retrieval_name) if isinstance(retrieval_name, str) else None
    if retrieval_url:
        print(f"Retrieval experiment URL: {retrieval_url}")
    print(f"QA experiment: {qa_name}")
    qa_url = _experiment_url(client, qa_name) if isinstance(qa_name, str) else None
    if qa_url:
        print(f"QA experiment URL: {qa_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
