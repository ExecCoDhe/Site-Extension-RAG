import time

from fastapi import APIRouter
from langsmith import traceable
from pydantic import BaseModel, Field

from app.api.errors import error_response
from app.config import get_settings
from app.index import GoogleEmbeddingClient, MissingGoogleConfiguration
from app.rag import GoogleGenerationClient
from app.rag.service import answer_workspace_question
from app.workspace import WorkspaceState, workspace_store

router = APIRouter()


class ChatRequest(BaseModel):
    job_id: str | None = None
    workspace_id: str | None = None
    session_id: str | None = None
    question: str = Field(min_length=1)


@router.post("/chat")
@traceable(name="chat_endpoint")
def chat(request: ChatRequest) -> dict[str, object]:
    workspace = workspace_store.ensure_workspace()
    if workspace.state != WorkspaceState.READY:
        return error_response(
            code="CHAT_BEFORE_READY",
            message="No ready workspace exists. Ingest the site before chatting.",
            status_code=409,
            retryable=True,
        )

    settings = get_settings()
    started_at = time.monotonic()
    try:
        response = answer_workspace_question(
            question=request.question,
            settings=settings,
            chunks=workspace_store.active_chunks(),
            embeddings=workspace_store.embeddings(settings.gemini_embedding_model),
            embedding_client=GoogleEmbeddingClient(
                api_key=settings.gemini_api_key,
                model=settings.gemini_embedding_model,
                timeout_seconds=settings.gemini_request_timeout_seconds,
            ),
            generation_client=GoogleGenerationClient(
                api_key=settings.gemini_api_key,
                model=settings.gemini_chat_model,
                timeout_seconds=settings.gemini_request_timeout_seconds,
            ),
            session_memory=workspace_store.session_memory(request.session_id),
        )
    except MissingGoogleConfiguration:
        return error_response(
            code="MISSING_API_KEY",
            message="Backend Google API credentials or model configuration are missing or unusable.",
            status_code=503,
            retryable=True,
        )

    trace = response.retrieval_trace
    query_plan = trace.get("query_plan", {}) if trace else {}
    trace_id = workspace_store.save_chat_trace(
        question=request.question,
        rewritten_question=str(query_plan.get("rewritten_question", request.question)),
        decomposition={
            "decomposed": bool(query_plan.get("decomposed", False)),
            "subqueries": query_plan.get("subqueries", []),
        },
        candidates=trace.get("candidates", []) if trace else [],
        evidence=[item.model_dump() for item in response.evidence],
        groundedness=response.groundedness.value,
        latency_ms=int((time.monotonic() - started_at) * 1000),
    )
    response.trace_id = trace_id
    try:
        from langsmith import get_current_run_tree
        run_tree = get_current_run_tree()
        if run_tree:
            response.langsmith_run_id = str(run_tree.id)
    except Exception:
        pass
    workspace_store.update_session_memory(
        session_id=request.session_id,
        question=request.question,
        answer=response.answer,
        citations=[citation.model_dump() for citation in response.citations],
    )

    return response.model_dump(exclude={"retrieval_trace"})
