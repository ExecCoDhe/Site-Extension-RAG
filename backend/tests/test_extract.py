from app.extract import extract_page


def test_extract_page_returns_only_clean_record_and_links() -> None:
    html = """
    <html>
      <head><title>Example Title</title></head>
      <body>
        <main>
          <h1>Example Heading</h1>
          <p>This is useful readable text for the extractor.</p>
          <a href="/next">Next</a>
        </main>
      </body>
    </html>
    """

    extracted = extract_page("https://example.com/page", html)

    assert extracted.record is not None
    assert extracted.record.url == "https://example.com/page"
    assert extracted.record.title == "Example Title"
    assert "useful readable text" in extracted.record.clean_text
    assert "https://example.com/next" in extracted.links
    assert extracted.record.content_hash
    assert extracted.record.quality_signals["text_length"] > 0
    assert extracted.record.boilerplate_removed == []
    assert not hasattr(extracted.record, "html")


def test_extract_page_marks_js_shell_for_rendered_fallback() -> None:
    html = """
    <html>
      <body>
        <div id="root"></div>
        <script></script><script></script><script></script>
      </body>
    </html>
    """

    extracted = extract_page("https://example.com/app", html)

    assert extracted.rendered_fallback_recommended


def test_extract_page_handles_nested_boilerplate_roles() -> None:
    html = """
    <html>
      <body>
        <div role="navigation">
          <div role="navigation">Nested nav</div>
        </div>
        <main>
          <h1>Theology</h1>
          <p>Theology is a readable article body that should be indexed.</p>
        </main>
      </body>
    </html>
    """

    extracted = extract_page("https://en.wikipedia.org/wiki/Theology", html)

    assert extracted.record is not None
    assert "readable article body" in extracted.record.clean_text
    assert "Nested nav" not in extracted.record.clean_text


def test_extract_page_without_readable_text_returns_no_record() -> None:
    extracted = extract_page("https://example.com/empty", "<html><body></body></html>")

    assert extracted.record is None


def test_extract_page_uses_html_canonical_link() -> None:
    html = """
    <html>
      <head>
        <title>Print</title>
        <link rel="canonical" href="https://example.com/article" />
      </head>
      <body><main><p>Readable print view content for indexing.</p></main></body>
    </html>
    """

    extracted = extract_page("https://example.com/article/print", html)

    assert extracted.record is not None
    assert extracted.record.canonical_url == "https://example.com/article"
    assert extracted.record.url == "https://example.com/article/print"
