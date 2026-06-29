from app.chunking import build_hierarchical_chunks
from app.crawl.models import PageRecord


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
