from fastapi import APIRouter
from pydantic import BaseModel

from app.api.errors import error_response
from app.config import get_settings
from app.evals import EvalCase, run_retrieval_eval
from app.evals.faithfulness import evaluate_faithfulness
from app.index import GoogleEmbeddingClient, MissingGoogleConfiguration
from app.rag import GoogleGenerationClient
from app.workspace import WorkspaceState, workspace_store

router = APIRouter()


class EvalRequest(BaseModel):
    cases: list[EvalCase]


@router.post("/evals/retrieval")
def retrieval_eval(request: EvalRequest) -> dict[str, object]:
    workspace = workspace_store.ensure_workspace()
    if workspace.state != WorkspaceState.READY:
        return error_response(
            code="CHAT_BEFORE_READY",
            message="No ready workspace exists. Ingest the site before running evals.",
            status_code=409,
            retryable=True,
        )

    settings = get_settings()
    try:
        return run_retrieval_eval(
            cases=request.cases,
            settings=settings,
            embedding_client=GoogleEmbeddingClient(
                api_key=settings.gemini_api_key,
                model=settings.gemini_embedding_model,
                timeout_seconds=settings.gemini_request_timeout_seconds,
            ),
            chunks=workspace_store.active_chunks(),
            embeddings=workspace_store.embeddings(settings.gemini_embedding_model),
        )
    except MissingGoogleConfiguration:
        return error_response(
            code="MISSING_API_KEY",
            message="Backend Google API credentials or model configuration are missing or unusable.",
            status_code=503,
            retryable=True,
        )


class GenerationEvalCase(BaseModel):
    question: str
    answer: str
    evidence_snippets: list[str]


class GenerationEvalRequest(BaseModel):
    cases: list[GenerationEvalCase]


@router.post("/evals/generation")
def generation_eval(request: GenerationEvalRequest) -> dict[str, object]:
    settings = get_settings()
    try:
        generation_client = GoogleGenerationClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_chat_model,
            timeout_seconds=settings.gemini_request_timeout_seconds,
        )
    except MissingGoogleConfiguration:
        return error_response(
            code="MISSING_API_KEY",
            message="Backend Google API credentials or model configuration are missing or unusable.",
            status_code=503,
            retryable=True,
        )

    results = []
    for case in request.cases:
        result = evaluate_faithfulness(
            question=case.question,
            answer=case.answer,
            evidence_snippets=case.evidence_snippets,
            generation_client=generation_client,
        )
        results.append(result.model_dump())

    total = max(len(results), 1)
    faithful_count = sum(1 for r in results if r["faithful"])
    avg_score = sum(r["score"] for r in results) / total

    return {
        "case_count": len(results),
        "faithful_rate": faithful_count / total,
        "average_score": round(avg_score, 4),
        "results": results,
    }
