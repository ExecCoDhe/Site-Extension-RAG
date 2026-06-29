from langsmith import traceable
from pydantic import BaseModel

from app.config import Settings
from app.index.embeddings import EmbeddingClient
from app.index.vector_index import VectorIndex
from app.jobs.models import IngestJob
from app.rag.generation import GenerationClient
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


def answer_question(
    *,
    job: IngestJob,
    question: str,
    settings: Settings,
    embedding_client: EmbeddingClient,
    generation_client: GenerationClient,
) -> ChatResponse:
    vector_index = job.vector_index
    if not isinstance(vector_index, VectorIndex):
        return ChatResponse(answer=NOT_FOUND_ANSWER, grounded=False, citations=[])

    query_embedding = embedding_client.embed_query(question)
    hits = vector_index.search(query_embedding, top_k=settings.top_k)
    if not hits:
        return ChatResponse(answer=NOT_FOUND_ANSWER, grounded=False, citations=[])

    generated = generation_client.generate_answer(question=question, hits=hits)
    hits_by_chunk_id = {hit.chunk.chunk_id: hit for hit in hits}
    supporting_ids = [
        chunk_id for chunk_id in generated.supporting_chunk_ids if chunk_id in hits_by_chunk_id
    ]

    if not generated.grounded or not supporting_ids:
        return ChatResponse(answer=NOT_FOUND_ANSWER, grounded=False, citations=[])

    citations = [
        Citation(
            url=hits_by_chunk_id[chunk_id].chunk.url,
            title=hits_by_chunk_id[chunk_id].chunk.title,
            chunk_id=chunk_id,
            score=hits_by_chunk_id[chunk_id].score,
        )
        for chunk_id in supporting_ids
    ]

    return ChatResponse(
        answer=generated.answer,
        grounded=True,
        groundedness=Groundedness.GROUNDED,
        citations=citations,
    )


@traceable(name="rag_pipeline")
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
    if not chunks:
        return ChatResponse(
            answer=NOT_FOUND_ANSWER,
            grounded=False,
            groundedness=Groundedness.NOT_GROUNDED,
            citations=[],
        )

    retrieval = RetrievalPipeline(
        settings=settings,
        embedding_client=embedding_client,
        chunks=chunks,
        embeddings=embeddings,
        session_memory=session_memory,
        dense_search_provider=LangChainQdrantDenseSearchProvider(
            path=settings.qdrant_path,
            collection_name=f"workspace_{chunks[0].workspace_id}",
        ),
        sparse_search_provider=BM25SparseSearchProvider(chunks=chunks),
    ).retrieve(question)

    if not retrieval.evidence:
        return ChatResponse(
            answer=NOT_FOUND_ANSWER,
            grounded=False,
            groundedness=Groundedness.NOT_GROUNDED,
            citations=[],
        )

    if hasattr(generation_client, "generate_answer_from_evidence"):
        generated = generation_client.generate_answer_from_evidence(
            question=question,
            evidence=retrieval.evidence,
        )
    else:
        generated = _deterministic_evidence_answer(question, retrieval.evidence)

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
