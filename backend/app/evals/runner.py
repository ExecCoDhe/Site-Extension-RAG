from pydantic import BaseModel, Field

from app.config import Settings
from app.index.embeddings import EmbeddingClient
from app.retrieval import RetrievalPipeline
from app.retrieval.service import BM25SparseSearchProvider
from app.workspace.models import ChildChunkRecord


class EvalCase(BaseModel):
    question: str
    expected_chunk_ids: list[str] = Field(default_factory=list)
    equivalent_chunk_ids: list[str] = Field(default_factory=list)
    expected_urls: list[str] = Field(default_factory=list)
    should_decompose: bool = False
    expected_groundedness: str | None = None


class EvalResult(BaseModel):
    question: str
    hit: bool
    exact_chunk_hit: bool
    equivalent_chunk_hit: bool
    url_hit: bool
    reciprocal_rank: float
    recall_at_k: float
    graded_recall: float
    decomposition_matched: bool
    retrieved_chunk_ids: list[str]
    retrieved_urls: list[str]


def run_retrieval_eval(
    *,
    cases: list[EvalCase],
    settings: Settings,
    embedding_client: EmbeddingClient,
    chunks: list[ChildChunkRecord],
    embeddings: dict[str, list[float]],
) -> dict[str, object]:
    results: list[EvalResult] = []
    sparse_provider = BM25SparseSearchProvider(chunks=chunks)
    for case in cases:
        retrieval = RetrievalPipeline(
            settings=settings,
            embedding_client=embedding_client,
            chunks=chunks,
            embeddings=embeddings,
            sparse_search_provider=sparse_provider,
        ).retrieve(case.question)
        retrieved_chunk_ids = [item.chunk_id for item in retrieval.evidence]
        retrieved_urls = [item.url for item in retrieval.evidence]
        exact_chunk_matches = set(case.expected_chunk_ids) & set(retrieved_chunk_ids)
        equivalent_chunk_matches = set(case.equivalent_chunk_ids) & set(retrieved_chunk_ids)
        exact_chunk_hit = bool(exact_chunk_matches)
        equivalent_chunk_hit = bool(equivalent_chunk_matches)
        url_hit = bool(set(case.expected_urls) & set(retrieved_urls))
        expected_items = max(len(case.expected_chunk_ids) + len(case.expected_urls), 1)
        exact_match_count = len(exact_chunk_matches) + len(set(case.expected_urls) & set(retrieved_urls))
        equivalent_match_count = len(equivalent_chunk_matches)
        recall_at_k = min(1.0, exact_match_count / expected_items)
        graded_recall = min(1.0, (exact_match_count + (0.5 * equivalent_match_count)) / expected_items)
        results.append(
            EvalResult(
                question=case.question,
                hit=exact_chunk_hit or equivalent_chunk_hit or url_hit,
                exact_chunk_hit=exact_chunk_hit,
                equivalent_chunk_hit=equivalent_chunk_hit,
                url_hit=url_hit,
                reciprocal_rank=_reciprocal_rank(
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieved_urls=retrieved_urls,
                    expected_chunk_ids=case.expected_chunk_ids,
                    equivalent_chunk_ids=case.equivalent_chunk_ids,
                    expected_urls=case.expected_urls,
                ),
                recall_at_k=recall_at_k,
                graded_recall=graded_recall,
                decomposition_matched=retrieval.query_plan.decomposed == case.should_decompose,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_urls=retrieved_urls,
            )
        )

    total = max(len(results), 1)
    return {
        "case_count": len(results),
        "hit_rate": sum(result.hit for result in results) / total,
        "exact_chunk_hit_rate": sum(result.exact_chunk_hit for result in results) / total,
        "equivalent_chunk_hit_rate": sum(result.equivalent_chunk_hit for result in results) / total,
        "url_hit_rate": sum(result.url_hit for result in results) / total,
        "recall_at_k": sum(result.recall_at_k for result in results) / total,
        "graded_recall": sum(result.graded_recall for result in results) / total,
        "mrr": sum(result.reciprocal_rank for result in results) / total,
        "decomposition_accuracy": sum(result.decomposition_matched for result in results) / total,
        "results": [result.model_dump() for result in results],
    }


def _reciprocal_rank(
    *,
    retrieved_chunk_ids: list[str],
    retrieved_urls: list[str],
    expected_chunk_ids: list[str],
    equivalent_chunk_ids: list[str],
    expected_urls: list[str],
) -> float:
    expected_chunks = set(expected_chunk_ids) | set(equivalent_chunk_ids)
    expected_url_set = set(expected_urls)
    for index, (chunk_id, url) in enumerate(zip(retrieved_chunk_ids, retrieved_urls, strict=False), start=1):
        if chunk_id in expected_chunks or url in expected_url_set:
            return 1 / index
    return 0.0
