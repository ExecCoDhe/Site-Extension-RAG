"""Print deterministic chunk IDs for eval fixture authoring."""

from __future__ import annotations

import argparse
import json
import sys

from app.config import Settings
from app.retrieval import RetrievalPipeline
from evals.config import DATASETS_DIR, eval_settings
from evals.loader import build_site_chunks, get_site_entry, load_manifest, load_pages


def _preview(text: str, limit: int = 80) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _print_chunks(site: str) -> None:
    _pages, _sections, chunks = build_site_chunks(site)
    print(f"site={site} chunk_count={len(chunks)}")
    for chunk in chunks:
        print(
            json.dumps(
                {
                    "chunk_id": chunk.chunk_id,
                    "section_id": chunk.section_id,
                    "page_id": chunk.page_id,
                    "url": chunk.url,
                    "token_start": chunk.token_start,
                    "token_end": chunk.token_end,
                    "preview": _preview(chunk.text),
                }
            )
        )


def _eval_settings_for_plan() -> Settings:
    manifest = load_manifest()
    return eval_settings(
        sqlite_path=":memory:",
        qdrant_path=":memory:",
        chunking_version=manifest["chunking_version"],
        embedding_model=manifest["embedding_model"],
    )


def _print_query_plans() -> None:
    retrieval_path = DATASETS_DIR / "retrieval.jsonl"
    qa_path = DATASETS_DIR / "qa.jsonl"
    if not retrieval_path.exists() and not qa_path.exists():
        print("no datasets found; skipping query-plan dump", file=sys.stderr)
        return

    settings = _eval_settings_for_plan()
    questions: list[tuple[str, str]] = []
    for path in (retrieval_path, qa_path):
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            questions.append((f"{path.name}:{line_number}", record["question"]))

    class _NoopEmbeddingClient:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise NotImplementedError

        def embed_query(self, text: str) -> list[float]:
            return [0.0]

    for label, question in questions:
        plan = RetrievalPipeline(
            settings=settings,
            embedding_client=_NoopEmbeddingClient(),
            chunks=[],
            embeddings={},
        )._query_plan(question)
        print(
            json.dumps(
                {
                    "source": label,
                    "question": question,
                    "decomposed": plan.decomposed,
                    "subqueries": plan.subqueries,
                }
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump deterministic eval fixture chunk IDs.")
    parser.add_argument("site", nargs="?", help="Fixture site name (e.g. acme_docs)")
    parser.add_argument(
        "--query-plans",
        action="store_true",
        help="Dump query plans for committed datasets (requires retrieval.jsonl / qa.jsonl).",
    )
    args = parser.parse_args(argv)

    if args.query_plans:
        _print_query_plans()
        return 0

    if not args.site:
        parser.error("site is required unless --query-plans is set")

    get_site_entry(args.site)
    load_pages(args.site)
    _print_chunks(args.site)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
