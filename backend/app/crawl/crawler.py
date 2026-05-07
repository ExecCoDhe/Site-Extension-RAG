import time
from collections import deque
from xml.etree import ElementTree
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit, urldefrag

import httpx
from pydantic import BaseModel, Field

from app.crawl.security import is_public_http_url, registrable_domain, same_hostname, same_registrable_domain
from app.extract.html import extract_page
from app.jobs.models import PageRecord


class CrawlResult(BaseModel):
    pages: list[PageRecord] = Field(default_factory=list)
    timed_out: bool = False
    skipped_count: int = 0
    rendered_fallback_count: int = 0


async def crawl_site(
    seed_url: str,
    *,
    timeout_seconds: int,
    max_pages: int,
    user_agent: str,
    allow_registrable_domain: bool = True,
    client: httpx.AsyncClient | None = None,
) -> CrawlResult:
    started_at = time.monotonic()
    queue: deque[str] = deque([_normalize_url(seed_url)])
    seen: set[str] = set()
    pages: list[PageRecord] = []
    skipped_count = 0
    rendered_fallback_count = 0

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            follow_redirects=True,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml",
            },
        )

    try:
        while queue:
            if len(pages) >= max_pages:
                return CrawlResult(
                    pages=pages,
                    skipped_count=skipped_count,
                    rendered_fallback_count=rendered_fallback_count,
                )

            if time.monotonic() - started_at >= timeout_seconds:
                return CrawlResult(
                    pages=pages,
                    timed_out=True,
                    skipped_count=skipped_count,
                    rendered_fallback_count=rendered_fallback_count,
                )

            url = _normalize_url(queue.popleft())
            canonical_url = _canonicalize_url(url)
            if canonical_url in seen:
                continue
            seen.add(canonical_url)

            if not _within_scope(seed_url, url, allow_registrable_domain) or not is_public_http_url(url):
                skipped_count += 1
                continue

            try:
                response = await client.get(url, timeout=10)
                response.raise_for_status()
            except httpx.HTTPError:
                skipped_count += 1
                continue

            content_type = response.headers.get("content-type", "")
            if _is_sitemap(final_url := _normalize_url(str(response.url)), content_type):
                for link in _parse_sitemap_links(response.text):
                    normalized_link = _normalize_url(link)
                    if (
                        _canonicalize_url(normalized_link) not in seen
                        and _within_scope(seed_url, normalized_link, allow_registrable_domain)
                    ):
                        queue.append(normalized_link)
                continue

            if "text/html" not in content_type.lower():
                skipped_count += 1
                continue

            if not _within_scope(seed_url, final_url, allow_registrable_domain) or not is_public_http_url(final_url):
                skipped_count += 1
                continue

            extracted = extract_page(final_url, response.text)
            if extracted.rendered_fallback_recommended:
                rendered_fallback_count += 1
            if extracted.record:
                extracted.record.acquisition_method = (
                    "rendered_fallback" if extracted.rendered_fallback_recommended else "html"
                )
                pages.append(extracted.record)

            for link in extracted.links:
                normalized_link = _normalize_url(link)
                if _canonicalize_url(normalized_link) not in seen and _within_scope(
                    seed_url,
                    normalized_link,
                    allow_registrable_domain,
                ):
                    queue.append(normalized_link)

            for sitemap_link in _sitemap_candidates(final_url):
                if _canonicalize_url(sitemap_link) not in seen and _within_scope(seed_url, sitemap_link, allow_registrable_domain):
                    queue.append(sitemap_link)

    finally:
        if owns_client:
            await client.aclose()

    return CrawlResult(
        pages=pages,
        skipped_count=skipped_count,
        rendered_fallback_count=rendered_fallback_count,
    )


def _normalize_url(url: str) -> str:
    return urldefrag(url)[0]


def _canonicalize_url(url: str) -> str:
    parts = urlsplit(_normalize_url(url))
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            (parts.path or "/").rstrip("/") or "/",
            "",
            "",
        )
    )


def _within_scope(seed_url: str, candidate_url: str, allow_registrable_domain: bool) -> bool:
    if same_hostname(seed_url, candidate_url):
        return True
    return allow_registrable_domain and same_registrable_domain(seed_url, candidate_url)


def _sitemap_candidates(url: str) -> list[str]:
    parsed = urlparse(url)
    if parsed.path not in {"", "/"}:
        return []
    root = f"{parsed.scheme}://{parsed.netloc}"
    return [urljoin(root, "/sitemap.xml")]


def _is_sitemap(url: str, content_type: str) -> bool:
    lowered_type = content_type.lower()
    lowered_url = url.lower()
    # Require path to look like a sitemap, not just any .xml file
    is_sitemap_url = "sitemap" in lowered_url and lowered_url.endswith(".xml")
    is_xml_content = "xml" in lowered_type
    return is_sitemap_url and is_xml_content


def _parse_sitemap_links(xml_text: str) -> list[str]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return []
    links: list[str] = []
    for element in root.iter():
        if element.tag.endswith("loc") and element.text:
            links.append(element.text.strip())
    return links
