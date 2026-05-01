from app.workspace.models import ChildChunkRecord


class QdrantVectorStore:
    """Thin local Qdrant adapter boundary.

    V2 keeps Qdrant as the primary vector-store direction, while tests and local
    runs can still exercise retrieval through stored SQLite vectors when the
    optional client is unavailable.
    """

    def __init__(self, *, path: str) -> None:
        self.path = path

    @property
    def available(self) -> bool:
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            return False
        return True

    def upsert_chunks(
        self,
        *,
        collection_name: str,
        chunks: list[ChildChunkRecord],
        embeddings: dict[str, list[float]],
    ) -> None:
        if not self.available:
            return
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        client = QdrantClient(path=self.path)
        vectors = [vector for vector in embeddings.values() if vector]
        if not vectors:
            return
        dimension = len(vectors[0])
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
        points = []
        for chunk in chunks:
            vector = embeddings.get(chunk.chunk_id)
            if vector is None:
                continue
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "workspace_id": chunk.workspace_id,
                        "url": chunk.url,
                        "title": chunk.title,
                        "heading_path": chunk.heading_path,
                    },
                )
            )
        if points:
            client.upsert(collection_name=collection_name, points=points)


class QdrantDenseSearchProvider:
    def __init__(self, *, path: str, collection_name: str) -> None:
        self.path = path
        self.collection_name = collection_name

    def search_scores(
        self,
        query_embedding: list[float],
        *,
        limit: int,
    ) -> dict[str, float] | None:
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            return None

        try:
            client = QdrantClient(path=self.path)
            if not client.collection_exists(self.collection_name):
                return None
            points = _query_points(
                client=client,
                collection_name=self.collection_name,
                query_embedding=query_embedding,
                limit=limit,
            )
        except Exception:
            return None

        scores: dict[str, float] = {}
        for point in points:
            payload = getattr(point, "payload", None) or {}
            chunk_id = payload.get("chunk_id")
            if chunk_id:
                scores[str(chunk_id)] = float(getattr(point, "score", 0.0))
        return scores


def _query_points(
    *,
    client,
    collection_name: str,
    query_embedding: list[float],
    limit: int,
):
    if hasattr(client, "query_points"):
        result = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True,
        )
        return getattr(result, "points", result)
    return client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=limit,
        with_payload=True,
    )
