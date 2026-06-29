import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.evals import EvalCase
from evals.config import DATASETS_DIR


class RetrievalGoldenCase(BaseModel):
    site: str
    question: str
    expected_chunk_ids: list[str] = Field(default_factory=list)
    equivalent_chunk_ids: list[str] = Field(default_factory=list)
    expected_urls: list[str] = Field(default_factory=list)
    should_decompose: bool = False
    expected_groundedness: str | None = None

    def to_eval_case(self) -> EvalCase:
        return EvalCase(
            question=self.question,
            expected_chunk_ids=self.expected_chunk_ids,
            equivalent_chunk_ids=self.equivalent_chunk_ids,
            expected_urls=self.expected_urls,
            should_decompose=self.should_decompose,
            expected_groundedness=self.expected_groundedness,
        )


class QAGoldenCase(BaseModel):
    site: str
    question: str
    expected_answer: str
    expected_groundedness: str
    expected_urls: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)


def _load_jsonl_dataset(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    if not path.exists():
        raise FileNotFoundError(f"missing dataset file: {path}")
    records: list[BaseModel] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            records.append(model.model_validate(payload))
        except (json.JSONDecodeError, ValidationError) as error:
            raise ValueError(f"{path.name} line {line_number}: {error}") from error
    return records


def load_retrieval_dataset() -> list[RetrievalGoldenCase]:
    records = _load_jsonl_dataset(DATASETS_DIR / "retrieval.jsonl", RetrievalGoldenCase)
    return [record for record in records if isinstance(record, RetrievalGoldenCase)]


def load_qa_dataset() -> list[QAGoldenCase]:
    records = _load_jsonl_dataset(DATASETS_DIR / "qa.jsonl", QAGoldenCase)
    return [record for record in records if isinstance(record, QAGoldenCase)]
