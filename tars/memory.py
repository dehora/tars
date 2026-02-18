import json
import os
import re
from pathlib import Path

MEMORY_PLACEHOLDER_RE = re.compile(r"<!--\s*tars:memory\b.*?-->\n?", re.DOTALL | re.IGNORECASE)

RECENT_SESSIONS_LIMIT = 2

_MEMORY_FILES = {
    "semantic": "Memory.md",
    "procedural": "Procedural.md",
}


def _memory_dir() -> Path | None:
    d = os.environ.get("TARS_MEMORY_DIR")
    if not d:
        return None
    p = Path(d)
    return p if p.is_dir() else None


def _memory_file(section: str) -> Path | None:
    d = _memory_dir()
    if d is None:
        return None
    filename = _MEMORY_FILES.get(section)
    return d / filename if filename else None


def _load_memory() -> str:
    """Load Memory.md (semantic) — always included in system prompt."""
    p = _memory_file("semantic")
    if p is None or not p.exists():
        return ""
    return p.read_text()


def _load_recent_sessions() -> str:
    """Load the most recent session logs for context."""
    d = _memory_dir()
    if d is None:
        return ""
    sessions_dir = d / "sessions"
    if not sessions_dir.is_dir():
        return ""
    files = sorted(sessions_dir.glob("*.md"), reverse=True)[:RECENT_SESSIONS_LIMIT]
    if not files:
        return ""
    # Load in chronological order (oldest first)
    parts = []
    for f in reversed(files):
        parts.append(f.read_text().strip())
    return "\n\n---\n\n".join(parts)


def _load_context() -> str:
    """Load today.md and yesterday.md from context/ for episodic context."""
    d = _memory_dir()
    if d is None:
        return ""
    context_dir = d / "context"
    if not context_dir.is_dir():
        return ""
    parts = []
    for name in ("today.md", "yesterday.md"):
        p = context_dir / name
        if p.exists():
            parts.append(p.read_text().strip())
    return "\n\n---\n\n".join(parts)


def _append_to_file(p: Path, content: str) -> None:
    """Append a list item to a memory file, replacing comment placeholders."""
    text = p.read_text() if p.exists() else ""
    # Remove only the dedicated placeholder comment, not arbitrary HTML comments.
    text = MEMORY_PLACEHOLDER_RE.sub("", text)
    text = text.rstrip() + f"\n- {content}\n"
    p.write_text(text)


def _run_memory_tool(name: str, args: dict) -> str:
    if name == "memory_recall":
        d = _memory_dir()
        if d is None:
            return json.dumps({"error": "Memory not configured (TARS_MEMORY_DIR not set)"})
        result = {}
        for section, filename in _MEMORY_FILES.items():
            p = d / filename
            if p.exists():
                result[section] = p.read_text()
        if not result:
            return json.dumps({"error": "No memory files found"})
        return json.dumps(result)

    if name == "memory_update":
        # Search both files for the old entry
        d = _memory_dir()
        if d is None:
            return json.dumps({"error": "Memory not configured (TARS_MEMORY_DIR not set)"})
        old_line = f"- {args['old_content'].strip()}"
        new_line = f"- {args['new_content'].strip()}"
        for filename in _MEMORY_FILES.values():
            p = d / filename
            if not p.exists():
                continue
            text = p.read_text()
            if old_line in text:
                text = text.replace(old_line, new_line, 1)
                p.write_text(text)
                return json.dumps({"ok": True, "old": args["old_content"], "new": args["new_content"]})
        return json.dumps({"error": f"Could not find existing entry: {args['old_content']}"})

    # memory_remember
    section = args["section"]
    if section not in _MEMORY_FILES:
        return json.dumps({"error": "Invalid section; must be semantic or procedural"})
    p = _memory_file(section)
    if p is None:
        return json.dumps({"error": "Memory not configured (TARS_MEMORY_DIR not set)"})
    content = args["content"].strip()
    _append_to_file(p, content)
    return json.dumps({"ok": True, "section": section, "content": content})
