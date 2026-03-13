import json
import os
from datetime import datetime
from pathlib import Path

_NOTE_READ_MAX_BYTES = 50_000


def _notes_dir() -> Path | None:
    """Return notes vault path from TARS_NOTES_DIR, or None."""
    val = os.environ.get("TARS_NOTES_DIR")
    if not val:
        return None
    return Path(val)


def _validate_note_path(path_str: str) -> tuple[Path, str | None]:
    """Validate a relative vault path. Returns (resolved_path, error_or_None)."""
    d = _notes_dir()
    if d is None:
        return Path(), "TARS_NOTES_DIR not configured"

    if not path_str or not path_str.strip():
        return Path(), "path is required"

    path_str = path_str.strip()

    if path_str.startswith("/"):
        return Path(), "absolute paths not allowed"

    if ".." in Path(path_str).parts:
        return Path(), "path traversal not allowed"

    if not path_str.endswith(".md"):
        path_str += ".md"

    resolved = (d / path_str).resolve()
    if not resolved.is_relative_to(d.resolve()):
        return Path(), "path traversal not allowed"

    return resolved, None


def daily_note(content: str) -> str:
    """Append content to today's daily note. Creates file if needed."""
    d = _notes_dir()
    if d is None:
        return json.dumps({"error": "TARS_NOTES_DIR not configured"})
    journal_dir = d / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    path = journal_dir / f"{today}.md"
    if not path.exists():
        path.write_text(f"# {today}\n\n", encoding="utf-8", errors="replace")
    with open(path, "a", encoding="utf-8", errors="replace") as f:
        f.write(f"- {content}\n")
    return json.dumps({"ok": True, "path": str(path)})


def note_write(path_str: str, content: str, *, overwrite: bool = False) -> str:
    """Create a new note in the vault. Refuses overwrite unless flag set."""
    resolved, err = _validate_note_path(path_str)
    if err:
        return json.dumps({"error": err})

    if not content or not content.strip():
        return json.dumps({"error": "content is required"})

    if resolved.exists() and not overwrite:
        rel = resolved.relative_to(_notes_dir().resolve())
        return json.dumps({"error": f"file already exists: {rel}", "path": str(rel)})

    resolved.parent.mkdir(parents=True, exist_ok=True)
    overwritten = resolved.exists()
    resolved.write_text(content, encoding="utf-8", errors="replace")

    rel = str(resolved.relative_to(_notes_dir().resolve()))
    if overwritten:
        return json.dumps({"ok": True, "path": rel, "overwritten": True})
    return json.dumps({"ok": True, "path": rel, "created": True})


def note_read(path_str: str) -> str:
    """Read a note from the vault by path."""
    resolved, err = _validate_note_path(path_str)
    if err:
        return json.dumps({"error": err})

    if not resolved.exists():
        rel = resolved.relative_to(_notes_dir().resolve())
        return json.dumps({"error": f"file not found: {rel}", "path": str(rel)})

    data = resolved.read_bytes()
    truncated = len(data) > _NOTE_READ_MAX_BYTES
    if truncated:
        data = data[:_NOTE_READ_MAX_BYTES]
    text = data.decode("utf-8", errors="replace")

    rel = str(resolved.relative_to(_notes_dir().resolve()))
    return json.dumps({"ok": True, "path": rel, "content": text, "truncated": truncated})


def note_append(path_str: str, content: str) -> str:
    """Append content to a vault note. Creates file if missing."""
    resolved, err = _validate_note_path(path_str)
    if err:
        return json.dumps({"error": err})

    if not content or not content.strip():
        return json.dumps({"error": "content is required"})

    resolved.parent.mkdir(parents=True, exist_ok=True)
    created = not resolved.exists()

    with open(resolved, "a", encoding="utf-8", errors="replace") as f:
        if not created:
            f.write("\n")
        f.write(content)

    rel = str(resolved.relative_to(_notes_dir().resolve()))
    return json.dumps({"ok": True, "path": rel, "created": created})


def _run_note_tool(name: str, args: dict) -> str:
    """Dispatch note tool calls."""
    if name == "note_daily":
        content = args.get("content", "").strip()
        if not content:
            return json.dumps({"error": "content is required"})
        return daily_note(content)
    if name == "note_write":
        path = args.get("path", "")
        content = args.get("content", "")
        overwrite = args.get("overwrite", False)
        return note_write(path, content, overwrite=bool(overwrite))
    if name == "note_read":
        path = args.get("path", "")
        return note_read(path)
    if name == "note_append":
        path = args.get("path", "")
        content = args.get("content", "")
        return note_append(path, content)
    return json.dumps({"error": f"Unknown note tool: {name}"})
