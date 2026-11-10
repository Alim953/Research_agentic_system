"""Web tools: search backends and a page fetcher with basic SSRF protection."""

from __future__ import annotations

import ipaddress
import re
import socket
from abc import ABC, abstractmethod
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .models import SearchHit

USER_AGENT = "Mozilla/5.0 (compatible; ResearchAssistantBot/1.0)"


# ---------------------------------------------------------------- search ----
class SearchBackend(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchHit]: ...


class DuckDuckGoSearch(SearchBackend):
    """Keyless web search via the ``ddgs`` package."""

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        try:
            from ddgs import DDGS
        except ImportError:
            try:  # older package name
                from duckduckgo_search import DDGS
            except ImportError:
                raise RuntimeError("Install a search package: pip install ddgs") from None
        rows = DDGS().text(query, max_results=max_results) or []
        hits = []
        for r in rows:
            url = r.get("href") or r.get("url")
            if url:
                hits.append(SearchHit(r.get("title", "") or url, url, r.get("body", "") or ""))
        return hits


class MockSearch(SearchBackend):
    """Offline backend returning predictable results (some URLs repeat across queries)."""

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        slug = re.sub(r"\W+", "-", query.lower()).strip("-")[:40]
        hits = [
            SearchHit("Shared reference page", "https://example.com/shared", "A page every query finds."),
            *[
                SearchHit(f"Result {i} for {query}", f"https://example.com/{slug}/{i}",
                          f"Snippet {i} about {query}.")
                for i in range(1, max_results)
            ],
        ]
        return hits[:max_results]


# ----------------------------------------------------------------- fetch ----
def is_public_url(url: str) -> bool:
    """True only for http(s) URLs whose host resolves exclusively to public addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            return False
    return bool(infos)


def extract_text(html: str, max_chars: int = 8000) -> str:
    """Pull readable text out of an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text("\n")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text[:max_chars]


class PageFetcher:
    def __init__(self, timeout: float = 10.0, max_bytes: int = 1_500_000,
                 max_chars: int = 8000, max_redirects: int = 3):
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_chars = max_chars
        self.max_redirects = max_redirects

    def fetch(self, url: str) -> str:
        """Return page text, or raise ``RuntimeError`` with a short reason."""
        for _ in range(self.max_redirects + 1):
            if not is_public_url(url):
                raise RuntimeError("blocked or unresolvable URL")
            try:
                resp = requests.get(
                    url, headers={"User-Agent": USER_AGENT}, timeout=self.timeout,
                    stream=True, allow_redirects=False,
                )
            except requests.RequestException as exc:
                raise RuntimeError(type(exc).__name__) from exc
            with resp:
                if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("Location")
                    if not loc:
                        raise RuntimeError("redirect without location")
                    url = urljoin(url, loc)
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                ctype = resp.headers.get("Content-Type", "")
                if "html" not in ctype and "text" not in ctype:
                    raise RuntimeError(f"unsupported content type ({ctype.split(';')[0] or 'unknown'})")
                body = resp.raw.read(self.max_bytes, decode_content=True)
                encoding = resp.encoding or "utf-8"
            html = body.decode(encoding, errors="replace")
            text = extract_text(html, self.max_chars) if "html" in ctype else html[: self.max_chars]
            if not text.strip():
                raise RuntimeError("no readable text")
            return text
        raise RuntimeError("too many redirects")


class MockFetcher:
    def fetch(self, url: str) -> str:
        if "/3" in url:
            raise RuntimeError("HTTP 403")  # exercise the fallback path
        return f"Mock page content for {url}. " + "This page explains the topic in detail. " * 20
