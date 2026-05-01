from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.api.errors import ErrorBody


class JobState(StrEnum):
    INGESTING = "ingesting"
    READY = "ready"
    ERROR = "error"


class PageRecord(BaseModel):
    url: str
    title: str
    clean_text: str
    canonical_url: str | None = None
    acquisition_method: str = "html"
    content_hash: str | None = None
    quality_score: float = 1.0
    quality_signals: dict[str, object] = Field(default_factory=dict)
    boilerplate_removed: list[str] = Field(default_factory=list)
    heading_paths: list[list[str]] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    chunk_id: str
    url: str
    title: str
    text: str
    section_id: str | None = None
    page_id: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    token_start: int = 0
    token_end: int = 0


class IngestJob(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    job_id: str
    state: JobState
    hostname: str
    page_count: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error: ErrorBody | None = None
    pages: list[PageRecord] = Field(default_factory=list, exclude=True)
    chunks: list[ChunkRecord] = Field(default_factory=list, exclude=True)
    vector_index: Any | None = Field(default=None, exclude=True)

    def public_summary(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "hostname": self.hostname,
            "page_count": self.page_count,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error.model_dump() if self.error else None,
        }
