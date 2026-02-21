"""Fetch and extract text content from web pages."""

import json
import urllib.request
from html.parser import HTMLParser

_MAX_CONTENT_LENGTH = 12_000
_TIMEOUT = 15
_USER_AGENT = "tars/1.0"
_SKIP_TAGS = {"script", "style", "noscript", "svg"}


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


def _run_web_tool(name: str, args: dict) -> str:
    """Fetch a URL and return its text content."""
    url = args.get("url", "").strip()
    if not url:
        return json.dumps({"error": "url is required"})

    if not url.startswith(("http://", "https://")):
        return json.dumps({"error": f"Invalid URL scheme: {url!r} — must start with http:// or https://"})

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
