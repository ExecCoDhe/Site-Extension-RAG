from enum import StrEnum

from pydantic import BaseModel, Field


class WorkspaceState(StrEnum):
    IDLE = "idle"
    INGESTING = "ingesting"
    READY = "ready"
    ERROR = "error"


class RunState(StrEnum):
    INGESTING = "ingesting"
    READY = "ready"
    ERROR = "error"


class Groundedness(StrEnum):
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    NOT_GROUNDED = "not_grounded"


class AcquisitionMethod(StrEnum):
    HTML = "html"
    RENDERED_FALLBACK = "rendered_fallback"


class WorkspaceSummary(BaseModel):
    workspace_id: str
    state: WorkspaceState
    hostname: str | None = None
    registrable_domain: str | None = None
    included_subdomains: list[str] = Field(default_factory=list)
    active_run_id: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    last_synced_at: str | None = None
    active_chunking_version: str | None = None
    active_embedding_version: str | None = None

    def public_summary(self) -> dict[str, object]:
        return self.model_dump()


class IngestRunSummary(BaseModel):
    run_id: str
    workspace_id: str
    state: RunState
    seed_url: str
    hostname: str
    registrable_domain: str
    page_count: int = 0
    chunk_count: int = 0
    fetched_count: int = 0
    skipped_count: int = 0
    rendered_fallback_count: int = 0
    started_at: str
    completed_at: str | None = None
    error: dict[str, object] | None = None

    @property
    def job_id(self) -> str:
        return self.run_id

    def public_summary(self) -> dict[str, object]:
        payload = self.model_dump()
        payload["job_id"] = self.run_id
        payload["state"] = self.state.value
        return payload


class PageVersionRecord(BaseModel):
    page_id: str
    workspace_id: str
    run_id: str
    canonical_url: str
    discovered_url: str
    title: str
    acquisition_method: AcquisitionMethod
    content_hash: str
    quality_score: float
    quality_signals: dict[str, object]
    boilerplate_removed: list[str]
    clean_text: str
    is_active: bool = True
    is_stale: bool = False


class ParentSectionRecord(BaseModel):
    section_id: str
    page_id: str
    workspace_id: str
    heading_path: list[str]
    section_index: int
    text: str
    start_offset: int
    end_offset: int


class ChildChunkRecord(BaseModel):
    chunk_id: str
    section_id: str
    page_id: str
    workspace_id: str
    chunking_version: str
    title: str
    url: str
    heading_path: list[str]
    text: str
    token_start: int
    token_end: int
    is_active: bool = True
