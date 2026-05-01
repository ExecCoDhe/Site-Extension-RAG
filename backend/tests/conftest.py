import pytest

from app.jobs import job_manager
from app.workspace import workspace_store


@pytest.fixture(autouse=True)
def reset_jobs() -> None:
    job_manager.reset()
    workspace_store.reset()
