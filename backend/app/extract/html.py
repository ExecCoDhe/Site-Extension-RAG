import hashlib
from urllib.parse import urljoin, urlsplit, urlunsplit

import trafilatura
from bs4 import BeautifulSoup
from pydantic import BaseModel

from app.crawl.models import PageRecord


class ExtractedPage(BaseModel):
    record: PageRecord | None
    links: list[str]
    rendered_fallback_recommended: bool = False


def extract_page(url: str, html: str) -> ExtractedPage:
    soup = BeautifulSoup(html, "lxml")
    removed = _remove_boilerplate(soup)
    title = _extract_title(soup, url)
    heading_paths = _heading_paths(soup)
    clean_text = trafilatura.extract(
        str(soup),
        include_comments=False,
        include_tables=False,
    )

    if not clean_text:
        clean_text = soup.get_text(" ", strip=True)

    links = [
        urljoin(url, href)
        for href in (anchor.get("href") for anchor in soup.find_all("a"))
        if href
    ]

    record = None
    if clean_text and clean_text.strip():
        normalized_text = clean_text.strip()
        quality_signals = _quality_signals(html=html, clean_text=normalized_text)
        record = PageRecord(
            url=url,
            canonical_url=_extract_canonical_url(soup, url),
            title=title,
            clean_text=normalized_text,
            content_hash=hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
            quality_score=float(quality_signals["quality_score"]),
            quality_signals=quality_signals,
            boilerplate_removed=removed,
            heading_paths=heading_paths,
        )

    return ExtractedPage(
        record=record,
        links=links,
        rendered_fallback_recommended=_looks_like_js_shell(html, record.clean_text if record else ""),
    )


def _extract_title(soup: BeautifulSoup, fallback: str) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    return fallback


def _remove_boilerplate(soup: BeautifulSoup) -> list[str]:
    removed: list[str] = []
    for selector in ["nav", "footer", "aside", "script", "style", "noscript"]:
        for element in soup.find_all(selector):
            removed.append(selector)
            element.decompose()

    for element in soup.find_all(attrs={"role": ["navigation", "contentinfo"]}):
        if element.attrs is None:
            continue
        removed.append(f"role:{element.get('role')}")
        element.decompose()

    return removed


def _heading_paths(soup: BeautifulSoup) -> list[list[str]]:
    paths: list[list[str]] = []
    stack: list[tuple[int, str]] = []
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = int(heading.name[1])
        text = heading.get_text(" ", strip=True)
        if not text:
            continue
        stack = [(existing_level, value) for existing_level, value in stack if existing_level < level]
        stack.append((level, text))
        paths.append([value for _, value in stack])
    return paths


def _quality_signals(*, html: str, clean_text: str) -> dict[str, object]:
    text_length = len(clean_text)
    html_length = max(len(html), 1)
    text_density = text_length / html_length
    link_count = html.lower().count("<a ")
    quality_score = min(1.0, (text_length / 800) + min(text_density, 0.4))
    return {
        "text_length": text_length,
        "text_density": round(text_density, 4),
        "link_count": link_count,
        "quality_score": round(quality_score, 4),
        "js_shell": _looks_like_js_shell(html, clean_text),
    }


def _looks_like_js_shell(html: str, clean_text: str) -> bool:
    lowered = html.lower()
    has_app_root = any(marker in lowered for marker in ['id="root"', "id='root'", 'id="app"', "id='app'"])
    heavy_script = lowered.count("<script") >= 3
    return len(clean_text.split()) < 40 and (has_app_root or heavy_script)


def _extract_canonical_url(soup: BeautifulSoup, page_url: str) -> str:
    for link in soup.find_all("link", rel=True):
        rel = link.get("rel")
        if isinstance(rel, list):
            rel_tokens = [part.lower() for part in rel]
        else:
            rel_tokens = [part.lower() for part in str(rel).split()]
        if "canonical" not in rel_tokens:
            continue
        href = link.get("href")
        if not href or not str(href).strip():
            continue
        absolute = urljoin(page_url, str(href).strip())
        parts = urlsplit(absolute)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            continue
        return _canonicalize_url(absolute)
    return _canonicalize_url(page_url)


def _canonicalize_url(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path.rstrip("/") or "/", "", ""))
