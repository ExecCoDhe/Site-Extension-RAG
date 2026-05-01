import logging
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, HttpUrl

from app.api.errors import error_response
from app.config import get_settings
from app.crawl.crawler import crawl_site
from app.crawl.security import is_public_http_url, registrable_domain
from app.index import GoogleEmbeddingClient, MissingGoogleConfiguration
from app.retrieval.vector_store import QdrantVectorStore
from app.workspace import workspace_store
from app.workspace.builder import build_workspace_records

router = APIRouter()
logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    url: HttpUrl


@router.post("/ingest", status_code=202)
async def start_ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    url = str(request.url)
    if not is_public_http_url(url):
        raise HTTPException(status_code=422, detail="Seed URL must be a public HTTP(S) URL.")

    hostname = urlparse(url).hostname
    if hostname is None:
        raise HTTPException(status_code=422, detail="Seed URL must include a hostname.")

    settings = get_settings()
    domain = registrable_domain(hostname)
    run = workspace_store.start_ingest_run(
        seed_url=url,
        hostname=hostname,
        registrable_domain=domain,
        included_subdomains=[hostname],
        chunking_version=settings.chunking_version,
        embedding_version=settings.gemini_embedding_model,
    )
    if run is None:
        active_run = workspace_store.active_ingest_run()
        return error_response(
            code="ACTIVE_JOB",
            message="Another site is already ingesting.",
            status_code=409,
            details=active_run.public_summary() if active_run else None,
            retryable=True,
        )

    background_tasks.add_task(run_ingest_job, run.run_id, url)
    return run.public_summary()


@router.get("/ingest/{job_id}/status")
def ingest_status(job_id: str) -> dict[str, object]:
    run = workspace_store.get_run(job_id)
    if run is None:
        return error_response(
            code="CHAT_BEFORE_READY",
            message="No ready artifact exists for that job. Ingest the site again.",
            status_code=409,
            retryable=True,
        )

    return run.public_summary()


@router.get("/workspace/status")
def workspace_status() -> dict[str, object]:
    return workspace_store.ensure_workspace().public_summary()


async def run_ingest_job(job_id: str, url: str) -> None:
    try:
        await _run_ingest_job(job_id, url)
    except Exception:
        logger.exception("Ingest job failed.")
        workspace_store.fail_run(
            job_id,
            code="INGEST_FAILED",
            message="Ingest failed unexpectedly. Check backend logs and try again.",
        )


async def _run_ingest_job(job_id: str, url: str) -> None:
    settings = get_settings()

    try:
        result = await crawl_site(
            url,
            timeout_seconds=settings.crawl_timeout_seconds,
            max_pages=settings.max_crawl_pages,
            user_agent=settings.crawl_user_agent,
            allow_registrable_domain=True,
        )
    except Exception:
        workspace_store.fail_run(
            job_id,
            code="NO_PAGES_INDEXED",
            message="No readable public pages were indexed for this hostname.",
        )
        return

    if result.timed_out:
        workspace_store.fail_run(
            job_id,
            code="INGEST_TIMEOUT",
            message="Ingest timed out after one minute.",
        )
        return

    if not result.pages:
        workspace_store.fail_run(
            job_id,
            code="NO_PAGES_INDEXED",
            message="No readable public pages were indexed for this hostname.",
        )
        return

    active_hashes = workspace_store.active_page_hashes()
    changed_pages = []
    retained_canonical_urls = []
    for page in result.pages:
        canonical_url = page.canonical_url or page.url
        if active_hashes.get(canonical_url) == page.content_hash:
            retained_canonical_urls.append(canonical_url)
        else:
            changed_pages.append(page)

    unchanged_count = len(retained_canonical_urls)
    page_records, sections, child_chunks = build_workspace_records(
        workspace_id=workspace_store.workspace_id,
        run_id=job_id,
        pages=changed_pages,
        settings=settings,
    )
    if not child_chunks and not retained_canonical_urls:
        workspace_store.fail_run(
            job_id,
            code="NO_PAGES_INDEXED",
            message="No readable public pages were indexed for this hostname.",
        )
        return

    child_embeddings = []
    if child_chunks:
        try:
            child_embeddings = GoogleEmbeddingClient(
                api_key=settings.gemini_api_key,
                model=settings.gemini_embedding_model,
            ).embed_documents([chunk.text for chunk in child_chunks])
        except MissingGoogleConfiguration as error:
            workspace_store.fail_run(
                job_id,
                code="MISSING_API_KEY",
                message=str(error),
            )
            return

    if len(child_embeddings) != len(child_chunks):
        workspace_store.fail_run(
            job_id,
            code="MISSING_API_KEY",
            message="Backend Google API credentials or model configuration are missing or unusable.",
        )
        return

    child_embedding_by_chunk_id = {
        chunk.chunk_id: vector
        for chunk, vector in zip(child_chunks, child_embeddings, strict=True)
    }

    workspace_store.replace_active_content(
        run_id=job_id,
        pages=page_records,
        sections=sections,
        chunks=child_chunks,
        embeddings=child_embedding_by_chunk_id,
        embedding_version=settings.gemini_embedding_model,
        rendered_fallback_count=result.rendered_fallback_count,
        skipped_count=result.skipped_count + unchanged_count,
        retained_canonical_urls=retained_canonical_urls,
    )
    QdrantVectorStore(path=settings.qdrant_path).upsert_chunks(
        collection_name=f"workspace_{workspace_store.workspace_id}",
        chunks=child_chunks,
        embeddings=child_embedding_by_chunk_id,
    )
