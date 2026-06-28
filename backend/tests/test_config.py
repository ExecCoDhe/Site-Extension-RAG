from app.config import Settings, _parse_cors_allow_origins


def test_cors_allow_origins_accepts_bare_wildcard() -> None:
    assert _parse_cors_allow_origins("*") == ["*"]
    assert Settings(cors_allow_origins="*").cors_allow_origins == ["*"]


def test_cors_allow_origins_accepts_json_array() -> None:
    assert _parse_cors_allow_origins('["*"]') == ["*"]
    assert Settings(cors_allow_origins='["https://example.com"]').cors_allow_origins == [
        "https://example.com"
    ]
