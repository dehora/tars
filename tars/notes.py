import json
import os
from datetime import datetime
from pathlib import Path


def _notes_dir() -> Path | None:
    """Return notes vault path from TARS_NOTES_DIR, or None."""
    val = os.environ.get("TARS_NOTES_DIR")
    if not val:
        return None
    return Path(val)


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


def _run_note_tool(name: str, args: dict) -> str:
    """Dispatch note tool calls."""
    if name == "note_daily":
        content = args.get("content", "").strip()
        if not content:
            return json.dumps({"error": "content is required"})
        return daily_note(content)
    return json.dumps({"error": f"Unknown note tool: {name}"})
