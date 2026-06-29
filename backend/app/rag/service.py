from langchain_core.runnables import RunnableLambda
from langsmith import traceable
from pydantic import BaseModel

from app.config import Settings
from app.index.embeddings import EmbeddingClient
from app.retrieval import RetrievalPipeline
from app.retrieval.models import EvidenceSnippet
from app.retrieval.service import BM25SparseSearchProvider
from app.retrieval.vector_store import LangChainQdrantDenseSearchProvider
from app.workspace import Groundedness
from app.workspace.models import ChildChunkRecord

NOT_FOUND_ANSWER = "The indexed site content does not contain enough information to answer that."


class Citation(BaseModel):
    url: str
    title: str
    chunk_id: str
    score: float
    section: str | None = None
    snippet: str | None = None
    nearby_context: str | None = None
    evidence_id: str | None = None
    parent_context_id: str | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    groundedness: Groundedness = Groundedness.NOT_GROUNDED
    citations: list[Citation]
    evidence: list[EvidenceSnippet] = []
    claims: list[dict[str, object]] = []
    trace_id: str | None = None
    langsmith_run_id: str | None = None
    retrieval_trace: dict[str, object] = {}


class WorkspaceRagPipeline:
    """LCEL composition of the workspace RAG flow (retrieve -> generate -> assemble).

    Mirrors RetrievalPipeline: __init__ takes the dependencies, answer(question)
    builds + invokes the RunnableLambda chain. The generation_client seam and
    _deterministic_evidence_answer fallback are preserved so injected fakes keep
    working.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        chunks: list[ChildChunkRecord],
        embeddings: dict[str, list[float]],
        embedding_client: EmbeddingClient,
        generation_client,
        session_memory: dict[str, object] | None = None,
    ) -> None:
        self._settings = settings
        self._chunks = chunks
        self._embeddings = embeddings
        self._embedding_client = embedding_client
        self._generation_client = generation_client
        self._session_memory = session_memory

    @traceable(name="rag_pipeline")
    def answer(self, question: str) -> ChatResponse:
        if not self._chunks:
            return ChatResponse(
                answer=NOT_FOUND_ANSWER,
                grounded=False,
                groundedness=Groundedness.NOT_GROUNDED,
                citations=[],
            )
        chain = (
            RunnableLambda(self._retrieve_step).with_config(run_name="retrieve")
            | RunnableLambda(self._generate_step).with_config(run_name="generate")
            | RunnableLambda(self._assemble_response_step).with_config(
                run_name="assemble_response"
            )
        )
        return chain.invoke(question)

    def _retrieve_step(self, question: str) -> dict:
        retrieval = RetrievalPipeline(
            settings=self._settings,
            embedding_client=self._embedding_client,
            chunks=self._chunks,
            embeddings=self._embeddings,
            session_memory=self._session_memory,
            dense_search_provider=LangChainQdrantDenseSearchProvider(
                path=self._settings.qdrant_path,
                collection_name=f"workspace_{self._chunks[0].workspace_id}",
            ),
            sparse_search_provider=BM25SparseSearchProvider(chunks=self._chunks),
        ).retrieve(question)
        return {"question": question, "retrieval": retrieval}

    def _generate_step(self, state: dict) -> dict:
        retrieval = state["retrieval"]
        if not retrieval.evidence:
            state["generated"] = None
            return state
        if hasattr(self._generation_client, "generate_answer_from_evidence"):
            state["generated"] = self._generation_client.generate_answer_from_evidence(
                question=state["question"],
                evidence=retrieval.evidence,
            )
        else:
            state["generated"] = _deterministic_evidence_answer(
                state["question"], retrieval.evidence
            )
        return state

    def _assemble_response_step(self, state: dict) -> ChatResponse:
        retrieval = state["retrieval"]
        generated = state["generated"]
        if generated is None:
            return ChatResponse(
                answer=NOT_FOUND_ANSWER,
                grounded=False,
                groundedness=Groundedness.NOT_GROUNDED,
                citations=[],
            )
        evidence_by_id = {item.evidence_id: item for item in retrieval.evidence}
        supporting_evidence_ids = [
            evidence_id
            for evidence_id in generated.supporting_evidence_ids
            if evidence_id in evidence_by_id
        ]
        groundedness = _groundedness_from_generated(
            generated.groundedness,
            supporting_evidence_ids,
            generated.claims,
        )
        if groundedness == Groundedness.NOT_GROUNDED:
            return ChatResponse(
                answer=NOT_FOUND_ANSWER,
                grounded=False,
                groundedness=groundedness,
                citations=[],
                evidence=[],
                claims=generated.claims,
            )
        used_evidence = [evidence_by_id[evidence_id] for evidence_id in supporting_evidence_ids]
        citations = [
            Citation(
                url=item.url,
                title=item.title,
                chunk_id=item.chunk_id,
                score=item.rerank_score,
                section=" > ".join(item.heading_path) if item.heading_path else None,
                snippet=item.snippet,
                nearby_context=item.nearby_context,
                evidence_id=item.evidence_id,
                parent_context_id=item.parent_context_id,
                dense_score=item.dense_score,
                sparse_score=item.sparse_score,
                rerank_score=item.rerank_score,
            )
            for item in used_evidence
        ]
        return ChatResponse(
            answer=generated.answer or NOT_FOUND_ANSWER,
            grounded=groundedness == Groundedness.GROUNDED,
            groundedness=groundedness,
            citations=citations,
            evidence=used_evidence,
            claims=generated.claims,
            retrieval_trace={
                "query_plan": retrieval.query_plan.model_dump(),
                "candidates": [
                    {
                        "chunk_id": candidate.chunk.chunk_id,
                        "dense_score": candidate.dense_score,
                        "sparse_score": candidate.sparse_score,
                        "fused_score": candidate.fused_score,
                        "rerank_score": candidate.rerank_score,
                        "subquery_provenance": candidate.subquery_provenance,
                    }
                    for candidate in retrieval.candidates
                ],
                "parent_contexts": [
                    parent_context.model_dump()
                    for parent_context in retrieval.parent_contexts
                ],
                "trace": retrieval.trace,
            },
        )


def answer_workspace_question(
    *,
    question: str,
    settings: Settings,
    chunks: list[ChildChunkRecord],
    embeddings: dict[str, list[float]],
    embedding_client: EmbeddingClient,
    generation_client,
    session_memory: dict[str, object] | None = None,
) -> ChatResponse:
    return WorkspaceRagPipeline(
        settings=settings,
        chunks=chunks,
        embeddings=embeddings,
        embedding_client=embedding_client,
        generation_client=generation_client,
        session_memory=session_memory,
    ).answer(question)


def _groundedness_from_generated(
    groundedness: str,
    supporting_evidence_ids: list[str],
    claims: list[dict[str, object]] | None = None,
) -> Groundedness:
    if not supporting_evidence_ids:
        return Groundedness.NOT_GROUNDED
    claims = claims or []
    material_claims = [claim for claim in claims if str(claim.get("text", "")).strip()]
    if material_claims:
        supported_claims = [
            claim
            for claim in material_claims
            if claim.get("supported") is True
            and bool(set(_claim_evidence_ids(claim)) & set(supporting_evidence_ids))
        ]
        if len(supported_claims) == len(material_claims):
            return Groundedness.GROUNDED
        if supported_claims:
            return Groundedness.PARTIALLY_GROUNDED
        return Groundedness.NOT_GROUNDED
    try:
        return Groundedness(groundedness)
    except ValueError:
        return Groundedness.PARTIALLY_GROUNDED


def _claim_evidence_ids(claim: dict[str, object]) -> list[str]:
    evidence_ids = claim.get("supporting_evidence_ids", [])
    if not isinstance(evidence_ids, list):
        return []
    return [str(evidence_id) for evidence_id in evidence_ids]


def _deterministic_evidence_answer(question: str, evidence: list[EvidenceSnippet]):
    from app.rag.generation import GeneratedAnswer

    first = evidence[0]
    return GeneratedAnswer(
        answer=first.snippet,
        grounded=True,
        groundedness=Groundedness.GROUNDED.value,
        claims=[
            {
                "text": first.snippet,
                "supporting_evidence_ids": [first.evidence_id],
                "supported": True,
            }
        ],
        supporting_evidence_ids=[first.evidence_id],
    )
