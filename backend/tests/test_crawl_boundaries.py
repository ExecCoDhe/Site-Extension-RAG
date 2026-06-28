from urllib.parse import urlparse

import httpx
import pytest

from app.crawl.crawler import crawl_site
from app.crawl.security import is_public_http_url, same_hostname, same_registrable_domain, same_site


def test_same_site_treats_www_as_equivalent() -> None:
    assert same_site("https://example.com/a", "https://www.example.com/b")
    assert same_site("https://www.example.com/a", "https://example.com/b")
    assert not same_site("https://example.com/a", "https://other.example.com/b")
    assert not same_site("https://example.com/a", "https://example.org/b")


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


@pytest.mark.anyio
async def test_crawl_site_rejects_other_language_wikipedia_subdomains(monkeypatch) -> None:
    monkeypatch.setattr("app.crawl.crawler.is_public_http_url", lambda url: True)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "de.wikipedia.org" in url or "fr.wikipedia.org" in url:
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="""
                <html>
                  <head><title>Foreign language page</title></head>
                  <body><main><p>Should not be indexed.</p></main></body>
                </html>
                """,
                request=request,
            )

        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="""
            <html>
              <head><title>English topic</title></head>
              <body>
                <main><p>English Wikipedia article about a topic.</p></main>
                <a href="https://de.wikipedia.org/wiki/Thema">Deutsch</a>
                <a href="https://fr.wikipedia.org/wiki/Sujet">Français</a>
                <a href="https://en.wikipedia.org/wiki/Related">Related</a>
              </body>
            </html>
            """,
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        result = await crawl_site(
            "https://en.wikipedia.org/wiki/Topic",
            timeout_seconds=60,
            max_pages=10,
            user_agent="test-agent",
            allow_registrable_domain=False,
            client=client,
        )

    assert not result.timed_out
    assert len(result.pages) == 2
    seed_hostname = urlparse("https://en.wikipedia.org/wiki/Topic").hostname
    for page in result.pages:
        assert urlparse(page.url).hostname == seed_hostname


@pytest.mark.anyio
async def test_crawl_site_rejects_subdomain_when_registrable_domain_disabled(monkeypatch) -> None:
    monkeypatch.setattr("app.crawl.crawler.is_public_http_url", lambda url: True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="""
            <html>
              <head><title>Home</title></head>
              <body>
                <main><p>Root site page with readable content.</p></main>
                <a href="https://docs.example.com/guide">Docs subdomain</a>
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
            max_pages=5,
            user_agent="test-agent",
            allow_registrable_domain=False,
            client=client,
        )

    assert not result.timed_out
    assert len(result.pages) == 1
    assert result.pages[0].url == "https://example.com/start"
    assert all(urlparse(page.url).hostname == "example.com" for page in result.pages)


@pytest.mark.anyio
async def test_crawl_site_accepts_www_and_bare_hostname_links(monkeypatch) -> None:
    monkeypatch.setattr("app.crawl.crawler.is_public_http_url", lambda url: True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="""
            <html>
              <head><title>Start</title></head>
              <body>
                <main><p>Starting page on www host.</p></main>
                <a href="https://example.com/next">Next on bare host</a>
              </body>
            </html>
            """,
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        result = await crawl_site(
            "https://www.example.com/start",
            timeout_seconds=60,
            max_pages=2,
            user_agent="test-agent",
            allow_registrable_domain=False,
            client=client,
        )

    assert not result.timed_out
    assert len(result.pages) == 2
    hostnames = {urlparse(page.url).hostname for page in result.pages}
    assert hostnames == {"www.example.com", "example.com"}


@pytest.mark.anyio
async def test_crawl_site_deduplicates_same_canonical_and_content_hash(monkeypatch) -> None:
    monkeypatch.setattr("app.crawl.crawler.is_public_http_url", lambda url: True)
    canonical = "https://example.com/article"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/print"):
            body = f"""
            <html>
              <head>
                <title>Print view</title>
                <link rel="canonical" href="{canonical}" />
              </head>
              <body><main><p>Duplicate article content for dedup testing.</p></main></body>
            </html>
            """
        else:
            body = f"""
            <html>
              <head>
                <title>Article</title>
                <link rel="canonical" href="{canonical}" />
              </head>
              <body>
                <main>
                  <p>Duplicate article content for dedup testing.</p>
                  <a href="https://example.com/article/print"></a>
                </main>
              </body>
            </html>
            """

        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=body,
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        result = await crawl_site(
            "https://example.com/article",
            timeout_seconds=60,
            max_pages=5,
            user_agent="test-agent",
            allow_registrable_domain=False,
            client=client,
        )

    assert not result.timed_out
    assert len(result.pages) == 1
    assert result.pages[0].canonical_url == canonical
    assert result.skipped_count >= 1


@pytest.mark.anyio
async def test_crawl_site_ignores_out_of_scope_canonical_for_dedup(monkeypatch) -> None:
    monkeypatch.setattr("app.crawl.crawler.is_public_http_url", lambda url: True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="""
            <html>
              <head>
                <title>Article</title>
                <link rel="canonical" href="https://other.example.com/shared" />
              </head>
              <body><main><p>Readable article body on the primary host.</p></main></body>
            </html>
            """,
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        result = await crawl_site(
            "https://example.com/article",
            timeout_seconds=60,
            max_pages=1,
            user_agent="test-agent",
            allow_registrable_domain=False,
            client=client,
        )

    assert not result.timed_out
    assert len(result.pages) == 1
    assert result.pages[0].canonical_url == "https://other.example.com/shared"
    assert result.pages[0].url == "https://example.com/article"
