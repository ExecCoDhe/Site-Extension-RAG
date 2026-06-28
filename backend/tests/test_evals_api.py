from fastapi.testclient import TestClient

from app.main import app


def test_generation_eval_before_ready_returns_error() -> None:
    client = TestClient(app)

    response = client.post(
        "/evals/generation",
        json={
            "cases": [
                {
                    "question": "What is alpha?",
                    "answer": "Alpha.",
                    "evidence_snippets": ["alpha content"],
                }
            ]
        },
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CHAT_BEFORE_READY"
