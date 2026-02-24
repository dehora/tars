"""Fetch and extract text content from web pages."""

import ipaddress
import json
import re
import socket
import urllib.parse
import urllib.request
from html import unescape
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
_BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "nav", "aside",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "blockquote", "pre", "figure",
}


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
        if tag_lower in ("img", "source") and self._in_body:
            if self._should_skip(attrs_dict):
                return
            for src in _select_image_sources(attrs_dict):
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
        if any(_attrs_match_skip(tag, a) for tag, a in self._stack):
            return True
        if _attrs_match_skip("", attrs):
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


class _MarkdownExtractor(HTMLParser):
    """Extract markdown with inline images, preserving approximate order."""

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self._base_url = base_url
        self._parts: list[str] = []
        self._stack: list[tuple[str, dict[str, str]]] = []
        self._in_body = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag_lower = tag.lower()
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        if tag_lower == "body":
            self._in_body = True
        self._stack.append((tag_lower, attrs_dict))
        if tag_lower in _SKIP_TAGS:
            self._skip_depth += 1
        if not self._in_body or self._skip_depth > 0:
            return
        if tag_lower in _BLOCK_TAGS:
            self._parts.append("\n")
        if tag_lower in ("img", "source"):
            if self._should_skip(attrs_dict):
                return
            for src in _select_image_sources(attrs_dict):
                full = urllib.parse.urljoin(self._base_url, src)
                if full.startswith("data:") or full.lower().endswith(".svg"):
                    continue
                self._parts.append(f"\n![]({full})\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower == "body":
            self._in_body = False
        if tag_lower in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag_lower:
                del self._stack[i]
                break
        if self._in_body and tag_lower in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._in_body or self._skip_depth > 0:
            return
        text = " ".join(data.split())
        if text:
            self._parts.append(text + " ")

    def _should_skip(self, attrs: dict[str, str]) -> bool:
        if any(tag in _IMAGE_SKIP_TAGS for tag, _ in self._stack):
            return True
        if any(_attrs_match_skip(tag, a) for tag, a in self._stack):
            return True
        if _attrs_match_skip("", attrs):
            return True
        return False

    def get_markdown(self) -> str:
        raw = "".join(self._parts)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        return "\n".join(line for line in lines if line).strip()


def _extract_markdown_with_images(html: str, base_url: str, max_len: int = _MAX_CONTENT_LENGTH) -> str:
    """Extract markdown with inline images from HTML body."""
    extractor = _MarkdownExtractor(base_url)
    extractor.feed(html)
    content = extractor.get_markdown()
    if len(content) > max_len:
        content = content[:max_len]
    return content


def _select_image_sources(attrs: dict[str, str]) -> list[str]:
    """Select candidate image sources from common attributes."""
    for key in ("data-src", "data-original", "data-lazy-src", "src"):
        value = attrs.get(key, "").strip()
        if value:
            return [value]
    for key in ("data-srcset", "srcset"):
        value = attrs.get(key, "").strip()
        if not value:
            continue
        candidates = _split_srcset(value)
        if candidates:
            return [candidates[-1]]
    return []


def _split_srcset(value: str) -> list[str]:
    """Parse a srcset value and return URLs in order."""
    urls: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        url = part.split()[0].strip()
        if url:
            urls.append(url)
    return urls


def _attrs_match_skip(tag: str, attrs: dict[str, str]) -> bool:
    if tag in ("body", "html"):
        return False
    for key in ("class", "id"):
        value = attrs.get(key, "").lower()
        if any(k in value for k in _IMAGE_SKIP_KEYWORDS):
            return True
    return False


def _extract_html_title(html: str) -> str:
    """Extract a best-effort title from HTML head."""
    if not html:
        return ""
    og_match = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    if og_match:
        return unescape(og_match.group(1)).strip()
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if title_match:
        return unescape(title_match.group(1)).strip()
    return ""


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
