import logging

from fastapi.testclient import TestClient

from app.logging_config import NOISY_LOGGERS
from app.main import app


def test_health_returns_ok() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_logging_policy() -> None:
    assert logging.getLogger().level == logging.INFO

    for logger_name in NOISY_LOGGERS:
        assert logging.getLogger(logger_name).level >= logging.WARNING
