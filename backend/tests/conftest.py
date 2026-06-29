import pytest

from app.workspace import workspace_store


@pytest.fixture(autouse=True)
def reset_state() -> None:
    workspace_store.reset()
