from fastapi import APIRouter
from pydantic import BaseModel

from app.api.errors import error_response
from app.config import get_settings
from app.evals import EvalCase, run_retrieval_eval
from app.index import GoogleEmbeddingClient, MissingGoogleConfiguration
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
