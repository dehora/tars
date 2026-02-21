"""Capture web pages to obsidian vault."""

import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from tars.core import chat
from tars.notes import _notes_dir
from tars.web import _run_web_tool

_CAPTURES_DIR = "17 tars captures"

_SUMMARIZE_PROMPT = """\
Extract the main article content from this web page text. Remove all site \
boilerplate (navigation, footers, sidebars, cookie notices, ads). Produce \
clean markdown with:

1. A title line: `# <article title>`
2. The article content in markdown, preserving headings, lists, and code blocks
3. Strip author bios, related posts, comment sections

Return ONLY the cleaned markdown, no commentary."""


def _extract_title(content: str, url: str) -> str:
    """Extract a title from content or fall back to URL slug."""
    # Try first markdown heading
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            title = line[2:].strip()
            if title:
                return title
    # Fall back to URL path slug
    path = urlparse(url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1] if path else "untitled"
    return slug.replace("-", " ").replace("_", " ").title() or "Untitled"


def _sanitize_filename(title: str) -> str:
    """Make a title safe for use as a filename."""
    # Remove characters unsafe for filenames
    clean = re.sub(r'[<>:"/\\|?*]', "", title)
    # Collapse whitespace
    clean = " ".join(clean.split())
    # Truncate to reasonable length
    return clean[:120] or "Untitled"


def capture(url: str, provider: str, model: str, *, raw: bool = False) -> str:
    """Fetch a URL, optionally summarize, and save to obsidian vault."""
    notes = _notes_dir()
    if notes is None:
        return json.dumps({"error": "TARS_NOTES_DIR not configured"})

    if not url:
        return json.dumps({"error": "url is required"})

    # Fetch the page
    fetch_result = json.loads(_run_web_tool("web_read", {"url": url}))
    if "error" in fetch_result:
        return json.dumps(fetch_result)

    content = fetch_result["content"]
    if not content.strip():
        return json.dumps({"error": "page returned no content"})

    # Summarize or use raw
    if raw:
        body = content
    else:
        messages = [{"role": "user", "content": f"{_SUMMARIZE_PROMPT}\n\n---\n\n{content}"}]
        body = chat(messages, provider, model, use_tools=False)

    # Extract title and build file
    title = _extract_title(body, url)
    filename = _sanitize_filename(title)
    today = datetime.now().strftime("%Y-%m-%d")

    captures_dir = notes / _CAPTURES_DIR
    captures_dir.mkdir(parents=True, exist_ok=True)
    path = captures_dir / f"{filename}.md"

    # YAML frontmatter + content
    frontmatter = f"---\nsource: {url}\ncaptured: {today}\n---\n\n"
    path.write_text(frontmatter + body, encoding="utf-8", errors="replace")

    return json.dumps({"ok": True, "path": str(path), "title": title})
