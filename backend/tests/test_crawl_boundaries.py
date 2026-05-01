import httpx
import pytest

from app.crawl.crawler import crawl_site
from app.crawl.security import is_public_http_url, same_hostname, same_registrable_domain


def test_same_hostname_rejects_off_host_and_subdomain() -> None:
    seed = "https://example.com/docs"

    assert same_hostname(seed, "https://example.com/other")
    assert not same_hostname(seed, "https://other.example.com/docs")
    assert not same_hostname(seed, "https://example.org/docs")
    assert same_registrable_domain(seed, "https://docs.example.com/guide")
    assert not same_registrable_domain(seed, "https://example.org/docs")


def test_private_and_unsupported_urls_are_not_public(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.crawl.security._resolve_host",
        lambda hostname: {"127.0.0.1"} if hostname == "example.com" else set(),
    )

    assert not is_public_http_url("http://localhost:8000")
    assert not is_public_http_url("file:///tmp/page.html")
    assert not is_public_http_url("https://example.com/page")


def test_unresolved_hosts_are_not_public(monkeypatch) -> None:
    monkeypatch.setattr("app.crawl.security._resolve_host", lambda hostname: set())

    assert not is_public_http_url("https://example.com/page")


@pytest.mark.anyio
async def test_crawl_site_stops_at_page_limit_and_skips_failed_links(monkeypatch) -> None:
    monkeypatch.setattr("app.crawl.crawler.is_public_http_url", lambda url: True)

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/bad"):
            return httpx.Response(500, request=request)

        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="""
            <html>
              <head><title>Page</title></head>
              <body>
                <main><p>Readable page text for indexing.</p></main>
                <a href="/bad">Bad</a>
                <a href="/next#section">Next</a>
              </body>
            </html>
            """,
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        result = await crawl_site(
            "https://example.com/start",
            timeout_seconds=60,
            max_pages=1,
            user_agent="test-agent",
            client=client,
        )

    assert not result.timed_out
    assert len(result.pages) == 1
    assert result.pages[0].url == "https://example.com/start"
