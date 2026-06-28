import math
import re
from collections import defaultdict
from typing import Protocol

from langsmith import traceable

from app.config import Settings
from app.index.embeddings import EmbeddingClient
from app.retrieval.models import (
    EvidenceSnippet,
    ParentContext,
    QueryPlan,
    RetrievalCandidate,
    RetrievalResult,
)
from app.workspace.models import ChildChunkRecord


class DenseSearchProvider(Protocol):
    def search_scores(
        self,
        query_embedding: list[float],
        *,
        limit: int,
    ) -> dict[str, float] | None:
        pass


class RetrievalPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        embedding_client: EmbeddingClient,
        chunks: list[ChildChunkRecord],
        embeddings: dict[str, list[float]],
        session_memory: dict[str, object] | None = None,
        dense_search_provider: DenseSearchProvider | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_client = embedding_client
        self._chunks = chunks
        self._embeddings = embeddings
        self._session_memory = session_memory or {}
        self._dense_search_provider = dense_search_provider

    @traceable(name="retrieval_pipeline")
    def retrieve(self, question: str) -> RetrievalResult:
        query_plan = self._query_plan(question)
        candidates_by_id: dict[str, RetrievalCandidate] = {}

        for subquery in query_plan.subqueries:
            subquery_embedding = self._embedding_client.embed_query(subquery)
            candidates = self._retrieve_subquery(subquery, subquery_embedding)
            for rank, candidate in enumerate(candidates, start=1):
                existing = candidates_by_id.get(candidate.chunk.chunk_id)
                rrf = 1 / (60 + rank)
                if existing is None:
                    candidate.fused_score = candidate.dense_score + candidate.sparse_score + rrf
                    candidate.subquery_provenance = [subquery]
                    candidates_by_id[candidate.chunk.chunk_id] = candidate
                else:
                    existing.fused_score += rrf
                    existing.dense_score = max(existing.dense_score, candidate.dense_score)
                    existing.sparse_score = max(existing.sparse_score, candidate.sparse_score)
                    if subquery not in existing.subquery_provenance:
                        existing.subquery_provenance.append(subquery)

        all_candidates = list(candidates_by_id.values())
        deduped = _dedupe_semantic_overlap(all_candidates)
        reranked = self._rerank(query_plan.rewritten_question, deduped)
        parent_contexts = _assemble_parent_contexts(
            reranked=reranked,
            all_candidates=all_candidates,
            limit=self._settings.parent_context_limit,
        )
        parent_context_by_id = {
            parent_context.parent_context_id: parent_context
            for parent_context in parent_contexts
        }
        evidence = [
            EvidenceSnippet(
                evidence_id=f"evidence_{index + 1}",
                chunk_id=candidate.chunk.chunk_id,
                section_id=candidate.chunk.section_id,
                parent_context_id=candidate.chunk.section_id,
                url=candidate.chunk.url,
                title=candidate.chunk.title,
                heading_path=candidate.chunk.heading_path,
                snippet=_snippet(candidate.chunk.text),
                nearby_context=(
                    parent_context_by_id[candidate.chunk.section_id].text
                    if candidate.chunk.section_id in parent_context_by_id
                    else None
                ),
                dense_score=candidate.dense_score,
                sparse_score=candidate.sparse_score,
                rerank_score=candidate.rerank_score,
            )
            for index, candidate in enumerate(reranked[: self._settings.rerank_limit])
        ]
        for item in evidence:
            parent_context = parent_context_by_id.get(item.parent_context_id)
            if parent_context is not None:
                parent_context.evidence_ids.append(item.evidence_id)
        return RetrievalResult(
            query_plan=query_plan,
            candidates=reranked,
            evidence=evidence,
            parent_contexts=parent_contexts,
            trace={
                "candidate_count": len(candidates_by_id),
                "deduped_count": len(deduped),
                "evidence_count": len(evidence),
                "parent_context_count": len(parent_contexts),
            },
        )

    def _query_plan(self, question: str) -> QueryPlan:
        rewritten = self._rewrite(question)
        subqueries = [rewritten]
        if _should_decompose(rewritten):
            subqueries = _subqueries(rewritten, self._settings.decomposition_max_subqueries)
        return QueryPlan(
            original_question=question,
            rewritten_question=rewritten,
            decomposed=len(subqueries) > 1,
            subqueries=subqueries,
        )

    def _rewrite(self, question: str) -> str:
        last_topic = self._session_memory.get("last_topic")
        if last_topic and re.search(r"\b(it|that|this|they|those)\b", question, re.I):
            return f"{question} (referring to {last_topic})"
        return question

    def _retrieve_subquery(
        self,
        subquery: str,
        query_embedding: list[float],
    ) -> list[RetrievalCandidate]:
        query_terms = set(_terms(subquery))
        candidates: list[RetrievalCandidate] = []
        provider_scores = (
            self._dense_search_provider.search_scores(
                query_embedding,
                limit=self._settings.retrieval_candidate_limit,
            )
            if self._dense_search_provider
            else None
        )
        for chunk in self._chunks:
            embedding = self._embeddings.get(chunk.chunk_id)
            dense_score = (
                provider_scores.get(chunk.chunk_id, 0.0)
                if provider_scores is not None
                else _cosine(query_embedding, embedding) if embedding is not None else 0.0
            )
            chunk_terms = set(_terms(chunk.text + " " + " ".join(chunk.heading_path) + " " + chunk.title))
            sparse_score = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            if dense_score <= 0 and sparse_score <= 0:
                continue
            candidates.append(
                RetrievalCandidate(
                    chunk=chunk,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    fused_score=(0.65 * dense_score) + (0.35 * sparse_score),
                )
            )
        return sorted(candidates, key=lambda item: item.fused_score, reverse=True)[
            : self._settings.retrieval_candidate_limit
        ]

    def _rerank(
        self,
        question: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        question_terms = set(_terms(question))
        for candidate in candidates:
            metadata_text = " ".join(candidate.chunk.heading_path + [candidate.chunk.title, candidate.chunk.url])
            metadata_overlap = len(question_terms & set(_terms(metadata_text))) / max(len(question_terms), 1)
            candidate.rerank_score = (
                0.55 * candidate.fused_score
                + 0.30 * candidate.dense_score
                + 0.10 * candidate.sparse_score
                + 0.05 * metadata_overlap
            )
        return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)


def _terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", text.lower()) if len(term) > 2]


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _should_decompose(question: str) -> bool:
    lowered = question.lower()
    compound_markers = [" and ", " vs ", " versus ", " compare ", " difference ", "timeline", " before ", " after "]
    return any(marker in lowered for marker in compound_markers)


def _subqueries(question: str, limit: int) -> list[str]:
    pieces = re.split(r"\s+(?:and|vs|versus)\s+|[,;]", question, flags=re.I)
    cleaned = [piece.strip() for piece in pieces if piece.strip()]
    return cleaned[:limit] if len(cleaned) > 1 else [question]


def _dedupe_semantic_overlap(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    by_section: dict[str, list[RetrievalCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_section[candidate.chunk.section_id or candidate.chunk.chunk_id].append(candidate)
    deduped = [max(group, key=lambda item: item.fused_score) for group in by_section.values()]
    return sorted(deduped, key=lambda item: item.fused_score, reverse=True)


def _assemble_parent_contexts(
    *,
    reranked: list[RetrievalCandidate],
    all_candidates: list[RetrievalCandidate],
    limit: int,
) -> list[ParentContext]:
    by_section: dict[str, list[RetrievalCandidate]] = defaultdict(list)
    for candidate in all_candidates:
        by_section[candidate.chunk.section_id].append(candidate)

    parent_contexts: list[ParentContext] = []
    seen_sections: set[str] = set()
    for candidate in reranked:
        section_id = candidate.chunk.section_id
        if section_id in seen_sections:
            continue
        seen_sections.add(section_id)
        section_candidates = sorted(
            by_section[section_id],
            key=lambda item: item.chunk.token_start,
        )
        text = " ".join(_snippet(item.chunk.text, limit=500) for item in section_candidates)
        parent_contexts.append(
            ParentContext(
                parent_context_id=section_id,
                section_id=section_id,
                url=candidate.chunk.url,
                title=candidate.chunk.title,
                heading_path=candidate.chunk.heading_path,
                text=_snippet(text, limit=1_200),
                rerank_score=candidate.rerank_score,
            )
        )
        if len(parent_contexts) >= limit:
            break

    return parent_contexts


def _snippet(text: str, limit: int = 360) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."
