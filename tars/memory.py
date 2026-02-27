import json
import os
import re
from datetime import datetime
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
    return p.read_text(encoding="utf-8", errors="replace")


def _load_procedural() -> str:
    """Load Procedural.md — learned rules included in system prompt."""
    p = _memory_file("procedural")
    if p is None or not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


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
        parts.append(f.read_text(encoding="utf-8", errors="replace").strip())
    return "\n\n---\n\n".join(parts)



def load_memory_files() -> dict[str, str]:
    """Load all memory files as {section: content}."""
    md = _memory_dir()
    if not md:
        return {}
    result = {}
    for section, filename in _MEMORY_FILES.items():
        p = md / filename
        if p.exists():
            result[section] = p.read_text(encoding="utf-8", errors="replace")
    return result


def _save_feedback(filename: str, header: str, user_msg: str, assistant_msg: str, note: str = "") -> str:
    """Append a flagged exchange to a feedback file in the memory dir."""
    md = _memory_dir()
    if not md:
        return "no memory dir configured"
    path = md / filename
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    entry = f"\n## {ts}\n- input: {user_msg}\n- got: {assistant_msg}\n"
    if note:
        entry += f"- note: {note}\n"
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        text = f"# {header}\n"
    text = text.rstrip() + "\n" + entry
    path.write_text(text, encoding="utf-8", errors="replace")
    return "feedback saved"


def load_feedback() -> tuple[str, str]:
    """Load corrections.md and rewards.md content."""
    md = _memory_dir()
    if not md:
        return "", ""
    corrections = ""
    rewards = ""
    cp = md / "corrections.md"
    rp = md / "rewards.md"
    if cp.exists():
        corrections = cp.read_text(encoding="utf-8", errors="replace")
    if rp.exists():
        rewards = rp.read_text(encoding="utf-8", errors="replace")
    return corrections, rewards


def archive_feedback() -> None:
    """Move corrections.md and rewards.md into feedback/ with timestamp suffix."""
    md = _memory_dir()
    if not md:
        return
    fb_dir = md / "feedback"
    fb_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    for name in ("corrections.md", "rewards.md"):
        p = md / name
        if p.exists():
            p.rename(fb_dir / f"{p.stem}-{ts}.md")


def save_correction(user_msg: str, assistant_msg: str, note: str = "") -> str:
    """Flag a wrong response."""
    return _save_feedback("corrections.md", "Corrections", user_msg, assistant_msg, note)


def save_reward(user_msg: str, assistant_msg: str, note: str = "") -> str:
    """Flag a good response."""
    return _save_feedback("rewards.md", "Rewards", user_msg, assistant_msg, note)


def _append_to_file(p: Path, content: str) -> None:
    """Append a list item to a memory file, replacing comment placeholders."""
    text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    # Remove only the dedicated placeholder comment, not arbitrary HTML comments.
    text = MEMORY_PLACEHOLDER_RE.sub("", text)
    text = text.rstrip() + f"\n- {content}\n"
    p.write_text(text, encoding="utf-8", errors="replace")


def _run_memory_tool(name: str, args: dict) -> str:
    if name == "memory_recall":
        d = _memory_dir()
        if d is None:
            return json.dumps({"error": "Memory not configured (TARS_MEMORY_DIR not set)"})
        result = {}
        for section, filename in _MEMORY_FILES.items():
            p = d / filename
            if p.exists():
                result[section] = p.read_text(encoding="utf-8", errors="replace")
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
            text = p.read_text(encoding="utf-8", errors="replace")
            if old_line in text:
                text = text.replace(old_line, new_line, 1)
                p.write_text(text, encoding="utf-8", errors="replace")
                return json.dumps({"ok": True, "old": args["old_content"], "new": args["new_content"]})
        return json.dumps({"error": f"Could not find existing entry: {args['old_content']}"})

    if name == "memory_forget":
        d = _memory_dir()
        if d is None:
            return json.dumps({"error": "Memory not configured (TARS_MEMORY_DIR not set)"})
        target = f"- {args['content'].strip()}"
        for filename in _MEMORY_FILES.values():
            p = d / filename
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            if target in text:
                text = text.replace(target + "\n", "", 1)
                p.write_text(text, encoding="utf-8", errors="replace")
                return json.dumps({"ok": True, "removed": args["content"]})
        return json.dumps({"error": f"Could not find entry: {args['content']}"})

    # memory_remember
    section = args["section"]
    if section not in _MEMORY_FILES:
        return json.dumps({"error": "Invalid section; must be semantic or procedural"})
    p = _memory_file(section)
    if p is None:
        return json.dumps({"error": "Memory not configured (TARS_MEMORY_DIR not set)"})
    content = args["content"].strip()
    existing = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    if f"- {content}" in existing:
        return json.dumps({"ok": True, "section": section, "content": content, "note": "already exists"})
    _append_to_file(p, content)
    return json.dumps({"ok": True, "section": section, "content": content})
