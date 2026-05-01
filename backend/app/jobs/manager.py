from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

from app.api.errors import ErrorBody, ErrorCode
from app.jobs.models import ChunkRecord, IngestJob, JobState, PageRecord


class InMemoryJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, IngestJob] = {}
        self._lock = Lock()

    def create_ingest_job(self, hostname: str) -> IngestJob | None:
        with self._lock:
            if self.active_ingest_job() is not None:
                return None

            job = IngestJob(
                job_id=f"job_{uuid4().hex}",
                state=JobState.INGESTING,
                hostname=hostname,
            )
            self._jobs[job.job_id] = job
            return job

    def active_ingest_job(self) -> IngestJob | None:
        for job in self._jobs.values():
            if job.state == JobState.INGESTING:
                return job
        return None

    def get(self, job_id: str) -> IngestJob | None:
        normalized_job_id = self._normalize_job_id(job_id)
        return self._jobs.get(normalized_job_id)

    def add_page(self, job_id: str, page: PageRecord) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.pages.append(page)
            job.page_count = len(job.pages)

    def mark_ready(
        self,
        job_id: str,
        *,
        chunks: list[ChunkRecord],
        vector_index: object,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.state = JobState.READY
            job.completed_at = datetime.now(timezone.utc)
            job.chunks = chunks
            job.vector_index = vector_index

    def fail(
        self,
        job_id: str,
        *,
        code: ErrorCode,
        message: str,
        retryable: bool = True,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.state = JobState.ERROR
            job.completed_at = datetime.now(timezone.utc)
            job.error = ErrorBody(
                code=code,
                message=message,
                details=None,
                retryable=retryable,
            )

    def reset(self) -> None:
        with self._lock:
            self._jobs.clear()

    def _normalize_job_id(self, job_id: str) -> str:
        if job_id in self._jobs or job_id.startswith("job_"):
            return job_id

        prefixed_job_id = f"job_{job_id}"
        if prefixed_job_id in self._jobs:
            return prefixed_job_id

        return job_id


job_manager = InMemoryJobManager()
