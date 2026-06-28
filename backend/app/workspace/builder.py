
from app.chunking import build_hierarchical_chunks
from app.config import Settings
from app.jobs.models import PageRecord
from app.workspace.models import AcquisitionMethod, PageVersionRecord


def build_workspace_records(
    *,
    workspace_id: str,
    run_id: str,
    pages: list[PageRecord],
    settings: Settings,
) -> tuple[list[PageVersionRecord], list, list]:
    sections, chunks = build_hierarchical_chunks(
        workspace_id=workspace_id,
        run_id=run_id,
        pages=pages,
        chunking_version=settings.chunking_version,
        token_budget=settings.child_chunk_token_budget,
        token_overlap=settings.child_chunk_token_overlap,
    )

    # Build a URL → page_id index from chunks to avoid O(pages × sections × chunks)
    url_to_page_id: dict[str, str] = {}
    chunk_page_ids: set[str] = {chunk.page_id for chunk in chunks}
    for section in sections:
        if section.page_id in chunk_page_ids:
            for chunk in chunks:
                if chunk.page_id == section.page_id and chunk.url not in url_to_page_id:
                    url_to_page_id[chunk.url] = section.page_id

    page_records = []
    for page in pages:
        page_id = url_to_page_id.get(page.url)
        if page_id is None:
            continue
        page_records.append(
            PageVersionRecord(
                page_id=page_id,
                workspace_id=workspace_id,
                run_id=run_id,
                canonical_url=page.canonical_url or page.url,
                discovered_url=page.url,
                title=page.title,
                acquisition_method=AcquisitionMethod(page.acquisition_method),
                content_hash=page.content_hash or "",
                quality_score=page.quality_score,
                quality_signals=page.quality_signals,
                boilerplate_removed=page.boilerplate_removed,
                clean_text=page.clean_text,
            )
        )
    return page_records, sections, chunks
