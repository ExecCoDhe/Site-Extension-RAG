from pydantic import BaseModel, Field

from app.workspace.models import ChildChunkRecord


class QueryPlan(BaseModel):
    original_question: str
    rewritten_question: str
    decomposed: bool = False
    subqueries: list[str] = Field(default_factory=list)


class RetrievalCandidate(BaseModel):
    chunk: ChildChunkRecord
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    subquery_provenance: list[str] = Field(default_factory=list)


class EvidenceSnippet(BaseModel):
    evidence_id: str
    chunk_id: str
    section_id: str
    parent_context_id: str
    url: str
    title: str
    heading_path: list[str]
    snippet: str
    nearby_context: str | None = None
    dense_score: float
    sparse_score: float
    rerank_score: float


class ParentContext(BaseModel):
    parent_context_id: str
    section_id: str
    url: str
    title: str
    heading_path: list[str]
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    rerank_score: float = 0.0


class RetrievalResult(BaseModel):
    query_plan: QueryPlan
    candidates: list[RetrievalCandidate]
    evidence: list[EvidenceSnippet]
    parent_contexts: list[ParentContext] = Field(default_factory=list)
    trace: dict[str, object] = Field(default_factory=dict)
