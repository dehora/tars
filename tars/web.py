"""Fetch and extract text content from web pages."""

import ipaddress
import json
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urlparse

_MAX_CONTENT_LENGTH = 12_000
_TIMEOUT = 15
_USER_AGENT = "tars/1.0"
_SKIP_TAGS = {"script", "style", "noscript", "svg"}
_IMAGE_SKIP_TAGS = {"script", "style", "noscript", "svg", "header", "footer", "nav", "aside"}
_IMAGE_SKIP_KEYWORDS = (
    "banner", "header", "footer", "nav", "sidebar", "aside",
    "advert", "ads", "sponsor", "promo", "cookie",
)


class _TextExtractor(HTMLParser):
    """Strip HTML tags and extract visible text content."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _extract_text(html: str) -> str:
    """Extract visible text from HTML, collapsing whitespace."""
    extractor = _TextExtractor()
    extractor.feed(html)
    text = extractor.get_text()
    # Collapse runs of whitespace into single spaces/newlines
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


class _ImageExtractor(HTMLParser):
    """Extract image sources from HTML body, avoiding common boilerplate."""

    def __init__(self) -> None:
        super().__init__()
        self._stack: list[tuple[str, dict[str, str]]] = []
        self._urls: list[str] = []
        self._in_body = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag_lower = tag.lower()
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        if tag_lower == "body":
            self._in_body = True
        self._stack.append((tag_lower, attrs_dict))
        if tag_lower == "img" and self._in_body:
            if self._should_skip(attrs_dict):
                return
            src = attrs_dict.get("src", "").strip()
            if src:
                self._urls.append(src)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower == "body":
            self._in_body = False
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag_lower:
                del self._stack[i]
                break

    def _should_skip(self, attrs: dict[str, str]) -> bool:
        if any(tag in _IMAGE_SKIP_TAGS for tag, _ in self._stack):
            return True
        if any(self._attrs_match_skip(a) for _, a in self._stack):
            return True
        if self._attrs_match_skip(attrs):
            return True
        return False

    def _attrs_match_skip(self, attrs: dict[str, str]) -> bool:
        for key in ("class", "id"):
            value = attrs.get(key, "").lower()
            if any(k in value for k in _IMAGE_SKIP_KEYWORDS):
                return True
        return False

    def get_urls(self) -> list[str]:
        return self._urls


def _extract_image_urls(html: str, base_url: str) -> list[str]:
    """Extract and normalize image URLs from HTML."""
    extractor = _ImageExtractor()
    extractor.feed(html)
    urls = []
    seen = set()
    for src in extractor.get_urls():
        if src.startswith("data:"):
            continue
        if src.lower().endswith(".svg"):
            continue
        full = urllib.parse.urljoin(base_url, src)
        if full not in seen:
            seen.add(full)
            urls.append(full)
    return urls


def _fetch_html(url: str) -> tuple[str | None, str | None]:
    """Fetch a URL and return raw HTML or an error message."""
    if not url:
        return None, "url is required"

    if not url.startswith(("http://", "https://")):
        return None, f"Invalid URL scheme: {url!r} — must start with http:// or https://"

    hostname = urlparse(url).hostname or ""
    if not hostname or _is_private_host(hostname):
        return None, "URL points to a private/internal address"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return None, f"Failed to fetch URL: {e}"

    return raw, None


def _is_private_host(hostname: str) -> bool:
    """Check if a hostname resolves to a private/loopback/link-local IP."""
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        return True  # can't resolve — reject
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            return True
    return False


def _run_web_tool(name: str, args: dict) -> str:
    """Fetch a URL and return its text content."""
    url = args.get("url", "").strip()
    if not url:
        return json.dumps({"error": "url is required"})

    if not url.startswith(("http://", "https://")):
        return json.dumps({"error": f"Invalid URL scheme: {url!r} — must start with http:// or https://"})

    hostname = urlparse(url).hostname or ""
    if not hostname or _is_private_host(hostname):
        return json.dumps({"error": "URL points to a private/internal address"})

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch URL: {e}"})

    content = _extract_text(raw)
    truncated = len(content) > _MAX_CONTENT_LENGTH
    if truncated:
        content = content[:_MAX_CONTENT_LENGTH]

    return json.dumps({"url": url, "content": content, "truncated": truncated})
