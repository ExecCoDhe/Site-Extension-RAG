from app.chunking import build_hierarchical_chunks, chunk_pages
from app.jobs.models import PageRecord


def test_short_page_produces_single_chunk_with_source_metadata() -> None:
    chunks = chunk_pages(
        "job_1",
        [PageRecord(url="https://example.com/a", title="A", clean_text="alpha beta")],
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "job_1:0000:0000"
    assert chunks[0].url == "https://example.com/a"
    assert chunks[0].title == "A"
    assert chunks[0].text == "alpha beta"


def test_long_page_uses_fixed_windows_with_overlap() -> None:
    chunks = chunk_pages(
        "job_1",
        [PageRecord(url="https://example.com/a", title="A", clean_text="abcdefghij")],
        chunk_size=4,
        chunk_overlap=1,
    )

    assert [chunk.text for chunk in chunks] == ["abcd", "defg", "ghij"]


def test_hierarchical_chunks_preserve_heading_path_and_parent_link() -> None:
    sections, chunks = build_hierarchical_chunks(
        workspace_id="default",
        run_id="run_1",
        pages=[
            PageRecord(
                url="https://example.com/a",
                title="A",
                clean_text="alpha beta gamma delta",
                heading_paths=[["A", "Overview"]],
            )
        ],
        chunking_version="test",
        token_budget=2,
        token_overlap=0,
    )

    assert len(sections) == 1
    assert [chunk.heading_path for chunk in chunks] == [["A", "Overview"], ["A", "Overview"]]
    assert chunks[0].section_id == sections[0].section_id
