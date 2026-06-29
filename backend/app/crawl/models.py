from pydantic import BaseModel, Field


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
