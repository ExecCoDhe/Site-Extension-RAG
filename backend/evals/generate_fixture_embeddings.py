"""Generate committed Gemini embeddings for eval fixtures (manual, network)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.index.embeddings import LangChainEmbeddingClient
from app.retrieval import RetrievalPipeline
from evals.config import FIXTURES_DIR, eval_settings
from evals.loader import build_site_chunks, load_manifest
from evals.schema import load_qa_dataset, load_retrieval_dataset


def _round_vector(vector: list[float]) -> list[float]:
    return [round(value, 6) for value in vector]


def _collect_subqueries() -> list[str]:
    manifest = load_manifest()
    settings = eval_settings(
        sqlite_path=":memory:",
        qdrant_path=":memory:",
        chunking_version=manifest["chunking_version"],
        embedding_model=manifest["embedding_model"],
    )

    class _NoopEmbeddingClient:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise NotImplementedError

        def embed_query(self, text: str) -> list[float]:
            return [0.0]

    pipeline = RetrievalPipeline(
        settings=settings,
        embedding_client=_NoopEmbeddingClient(),
        chunks=[],
        embeddings={},
    )
    questions: list[str] = []
    for case in load_retrieval_dataset():
        questions.append(case.question)
    for case in load_qa_dataset():
        questions.append(case.question)

    subqueries: list[str] = []
    seen: set[str] = set()
    for question in questions:
        plan = pipeline._query_plan(question)
        for subquery in plan.subqueries:
            if subquery not in seen:
                seen.add(subquery)
                subqueries.append(subquery)
    return subqueries


def _write_json(path: Path, payload: dict[str, list[float]]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate_embeddings(*, force: bool = False) -> None:
    settings = get_settings()
    if not settings.gemini_api_key:
        print(
            "GEMINI_API_KEY is not configured. This script makes paid network calls and is never run in CI.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("=" * 72)
    print("WARNING: generate_fixture_embeddings makes paid Gemini API calls.")
    print("Do not run this in CI. Commit the generated JSON artifacts instead.")
    print("=" * 72)

    manifest = load_manifest()
    embedding_dim = manifest["embedding_dim"]
    client = LangChainEmbeddingClient(
        api_key=settings.gemini_api_key,
        model=manifest["embedding_model"],
        timeout_seconds=settings.gemini_request_timeout_seconds,
    )

    for entry in manifest["sites"]:
        site = entry["site"]
        embeddings_path = FIXTURES_DIR / site / "embeddings.json"
        _pages, _sections, chunks = build_site_chunks(site)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if embeddings_path.exists() and not force:
            existing = json.loads(embeddings_path.read_text(encoding="utf-8"))
            if set(existing.keys()) == set(chunk_ids):
                print(f"skip {site}: embeddings.json already up to date")
                continue

        print(f"embedding documents for {site} ({len(chunks)} chunks)...")
        vectors = client.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError(f"embedding count mismatch for {site}")
        payload = {
            chunk.chunk_id: _round_vector(vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        }
        for vector in payload.values():
            if len(vector) != embedding_dim:
                raise RuntimeError(f"unexpected embedding dim {len(vector)} for {site}")
        _write_json(embeddings_path, payload)
        print(f"wrote {embeddings_path}")

    query_path = FIXTURES_DIR / "query_embeddings.json"
    existing_queries: dict[str, list[float]] = {}
    if query_path.exists():
        existing_queries = json.loads(query_path.read_text(encoding="utf-8"))

    subqueries = _collect_subqueries()
    pending = [subquery for subquery in subqueries if force or subquery not in existing_queries]
    if pending:
        print(f"embedding {len(pending)} subqueries (RETRIEVAL_QUERY)...")
        for subquery in pending:
            vector = client.embed_query(subquery)
            if len(vector) != embedding_dim:
                raise RuntimeError(f"unexpected query embedding dim {len(vector)}")
            existing_queries[subquery] = _round_vector(vector)
    else:
        print("skip query embeddings: already up to date")

    for subquery in subqueries:
        if subquery not in existing_queries:
            raise RuntimeError(f"missing query embedding for subquery: {subquery!r}")

    _write_json(query_path, {subquery: existing_queries[subquery] for subquery in subqueries})

    manifest_path = FIXTURES_DIR / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["generated_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    print(f"updated {manifest_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate committed eval fixture embeddings.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all document and query embeddings even when files look current.",
    )
    args = parser.parse_args(argv)
    generate_embeddings(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
