from fastapi.testclient import TestClient

from app.main import app


def test_chrome_extension_origin_preflight_is_allowed() -> None:
    client = TestClient(app)

    response = client.options(
        "/health",
        headers={
            "Origin": "chrome-extension://example-extension-id",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
