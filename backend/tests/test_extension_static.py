from pathlib import Path

EXTENSION_DIR = Path(__file__).resolve().parents[2] / "extension"


def test_extension_uses_vanilla_mv3_manifest() -> None:
    manifest = (EXTENSION_DIR / "manifest.json").read_text(encoding="utf-8")

    assert '"manifest_version": 3' in manifest
    assert '"activeTab"' in manifest
    assert '"storage"' in manifest
    assert "http://127.0.0.1:8000/*" in manifest


def test_extension_stays_ui_and_transport_only() -> None:
    forbidden_terms = [
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "gemini-",
        "faiss",
        "embedding",
        "chunking",
        "content_scripts",
        "document.body",
        "innerHTML",
        "React",
        "Vue",
        "Svelte",
    ]

    for path in EXTENSION_DIR.glob("*"):
        if path.suffix not in {".html", ".css", ".js", ".json"}:
            continue

        content = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in content, f"{term} should not appear in {path.name}"


def test_extension_renders_workspace_and_expandable_evidence_context() -> None:
    popup_html = (EXTENSION_DIR / "popup.html").read_text(encoding="utf-8")
    popup_js = (EXTENSION_DIR / "popup.js").read_text(encoding="utf-8")

    assert 'id="chunk-count"' in popup_html
    assert 'id="sync-stats"' in popup_html
    assert "citation.nearby_context" in popup_js
    assert "document.createElement(\"details\")" in popup_js
