from uuid import NAMESPACE_URL, uuid5

from app.crawl.models import PageRecord
from app.workspace.models import ChildChunkRecord, ParentSectionRecord


def build_hierarchical_chunks(
    *,
    workspace_id: str,
    run_id: str,
    pages: list[PageRecord],
    chunking_version: str,
    token_budget: int,
    token_overlap: int,
) -> tuple[list[ParentSectionRecord], list[ChildChunkRecord]]:
    sections: list[ParentSectionRecord] = []
    chunks: list[ChildChunkRecord] = []

    for page_index, page in enumerate(pages):
        page_id = _stable_id("page", run_id, page.canonical_url or page.url, page.content_hash or "")
        section_texts = _section_texts(page)
        offset = 0

        for section_index, (heading_path, section_text) in enumerate(section_texts):
            section_id = _stable_id("section", page_id, str(section_index), " / ".join(heading_path))
            end_offset = offset + len(section_text)
            section = ParentSectionRecord(
                section_id=section_id,
                page_id=page_id,
                workspace_id=workspace_id,
                heading_path=heading_path,
                section_index=section_index,
                text=section_text,
                start_offset=offset,
                end_offset=end_offset,
            )
            sections.append(section)

            for chunk_index, (text, token_start, token_end) in enumerate(
                _token_windows(section_text, token_budget, token_overlap)
            ):
                chunks.append(
                    ChildChunkRecord(
                        chunk_id=_stable_id("chunk", section_id, str(chunk_index), chunking_version),
                        section_id=section_id,
                        page_id=page_id,
                        workspace_id=workspace_id,
                        chunking_version=chunking_version,
                        title=page.title,
                        url=page.url,
                        heading_path=heading_path,
                        text=text,
                        token_start=token_start,
                        token_end=token_end,
                    )
                )
            offset = end_offset + 1

    return sections, chunks


def _section_texts(page: PageRecord) -> list[tuple[list[str], str]]:
    normalized = " ".join(page.clean_text.split())
    heading_path = page.heading_paths[0] if page.heading_paths else [page.title]
    return [(heading_path, normalized)] if normalized else []


def _token_windows(text: str, token_budget: int, token_overlap: int) -> list[tuple[str, int, int]]:
    tokens = text.split()
    if not tokens:
        return []
    if len(tokens) <= token_budget:
        return [(" ".join(tokens), 0, len(tokens))]

    step = max(1, token_budget - token_overlap)
    windows: list[tuple[str, int, int]] = []
    start = 0
    while start < len(tokens):
        end = min(start + token_budget, len(tokens))
        windows.append((" ".join(tokens[start:end]), start, end))
        if end >= len(tokens):
            break
        start += step
    return windows


def _stable_id(*parts: str) -> str:
    return str(uuid5(NAMESPACE_URL, ":".join(parts)))
