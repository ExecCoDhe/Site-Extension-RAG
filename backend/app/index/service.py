from app.chunking import chunk_pages
from app.config import Settings
from app.index.embeddings import EmbeddingClient
from app.index.vector_index import VectorIndex
from app.jobs.models import ChunkRecord, PageRecord


def build_index(
    *,
    job_id: str,
    pages: list[PageRecord],
    settings: Settings,
    embedding_client: EmbeddingClient,
) -> tuple[list[ChunkRecord], VectorIndex]:
    chunks = chunk_pages(
        job_id,
        pages,
        chunk_size=settings.chunk_size_chars,
        chunk_overlap=settings.chunk_overlap_chars,
    )
    if not chunks:
        raise ValueError("No chunks available to index.")

    embeddings = embedding_client.embed_documents([chunk.text for chunk in chunks])
    return chunks, VectorIndex.from_embeddings(chunks, embeddings)
