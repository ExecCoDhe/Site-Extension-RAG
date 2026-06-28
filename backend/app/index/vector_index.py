import faiss
import numpy as np
from pydantic import BaseModel

from app.jobs.models import ChunkRecord


class SearchHit(BaseModel):
    chunk: ChunkRecord
    score: float


class VectorIndex:
    def __init__(
        self,
        index: faiss.Index,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self._index = index
        self._chunks = chunks
        self.embeddings = embeddings or []

    @classmethod
    def from_embeddings(
        cls,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> "VectorIndex":
        vectors = _normalize(np.array(embeddings, dtype="float32"))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, chunks=chunks, embeddings=vectors.tolist())

    def search(self, query_embedding: list[float], *, top_k: int) -> list[SearchHit]:
        query = _normalize(np.array([query_embedding], dtype="float32"))
        scores, indexes = self._index.search(query, min(top_k, len(self._chunks)))

        hits: list[SearchHit] = []
        for score, chunk_index in zip(scores[0], indexes[0]):
            if chunk_index < 0:
                continue
            hits.append(SearchHit(chunk=self._chunks[int(chunk_index)], score=float(score)))

        return hits


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms
