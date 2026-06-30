"""Idempotent sync of committed golden datasets to LangSmith (manual, network)."""

from __future__ import annotations

import argparse
from typing import Any

from langsmith import Client

from evals.config import QA_DATASET_NAME, RETRIEVAL_DATASET_NAME
from evals.langsmith_eval import _require_online_env
from evals.schema import load_qa_dataset, load_retrieval_dataset


def _dataset_id(client: Client, dataset_name: str, *, recreate: bool) -> str:
    if recreate:
        try:
            existing = client.read_dataset(dataset_name=dataset_name)
            client.delete_dataset(dataset_id=existing.id)
        except Exception:
            pass
        created = client.create_dataset(dataset_name=dataset_name)
        return str(created.id)

    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        return str(existing.id)
    except Exception:
        created = client.create_dataset(dataset_name=dataset_name)
        return str(created.id)


def _existing_questions(client: Client, dataset_id: str) -> set[str]:
    questions: set[str] = set()
    for example in client.list_examples(dataset_id=dataset_id):
        inputs = example.inputs or {}
        question = inputs.get("question")
        if isinstance(question, str):
            questions.add(question)
    return questions


def _sync_retrieval_dataset(client: Client, *, recreate: bool) -> tuple[int, int]:
    dataset_id = _dataset_id(client, RETRIEVAL_DATASET_NAME, recreate=recreate)
    existing = set() if recreate else _existing_questions(client, dataset_id)
    to_create: list[dict[str, Any]] = []
    skipped = 0

    for case in load_retrieval_dataset():
        if case.question in existing:
            skipped += 1
            continue
        to_create.append(
            {
                "inputs": {"question": case.question, "site": case.site},
                "outputs": {
                    "expected_chunk_ids": case.expected_chunk_ids,
                    "equivalent_chunk_ids": case.equivalent_chunk_ids,
                    "expected_urls": case.expected_urls,
                    "should_decompose": case.should_decompose,
                },
            }
        )

    if to_create:
        client.create_examples(dataset_id=dataset_id, examples=to_create)
    return len(to_create), skipped


def _sync_qa_dataset(client: Client, *, recreate: bool) -> tuple[int, int]:
    dataset_id = _dataset_id(client, QA_DATASET_NAME, recreate=recreate)
    existing = set() if recreate else _existing_questions(client, dataset_id)
    to_create: list[dict[str, Any]] = []
    skipped = 0

    for case in load_qa_dataset():
        if case.question in existing:
            skipped += 1
            continue
        to_create.append(
            {
                "inputs": {"question": case.question, "site": case.site},
                "outputs": {
                    "expected_answer": case.expected_answer,
                    "expected_groundedness": case.expected_groundedness,
                    "expected_urls": case.expected_urls,
                    "expected_chunk_ids": case.expected_chunk_ids,
                },
            }
        )

    if to_create:
        client.create_examples(dataset_id=dataset_id, examples=to_create)
    return len(to_create), skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload committed eval datasets to LangSmith.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate datasets before uploading examples.",
    )
    args = parser.parse_args(argv)

    print("=" * 72)
    print("WARNING: upload_datasets makes LangSmith API calls.")
    print("Do not run this in CI. Requires LANGSMITH_API_KEY.")
    print("=" * 72)

    _require_online_env()
    client = Client()

    retrieval_created, retrieval_skipped = _sync_retrieval_dataset(client, recreate=args.recreate)
    qa_created, qa_skipped = _sync_qa_dataset(client, recreate=args.recreate)

    print(
        f"{RETRIEVAL_DATASET_NAME}: created {retrieval_created}, skipped {retrieval_skipped}"
    )
    print(f"{QA_DATASET_NAME}: created {qa_created}, skipped {qa_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
