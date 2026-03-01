import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tars.core import chat
from tars.memory import _memory_dir

SESSION_COMPACTION_INTERVAL = 10

SUMMARIZE_PROMPT = """\
Summarize this conversation concisely for a session log.
Include what was discussed, decisions made, and which tools were used (not their payloads).
Use bullet points. Be brief."""

SUMMARIZE_INCREMENTAL_PROMPT = """\
Summarize only what is NEW in this conversation since the previous summary.
Do not repeat anything already covered. Include new topics discussed, decisions made, \
and which tools were used (not their payloads).
Use bullet points. Be brief."""


def _escape_prompt_text(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _session_path(channel: str = "") -> Path | None:
    """Returns a timestamped session file path, or None if memory not configured."""
    d = _memory_dir()
    if d is None:
        return None
    sessions_dir = d / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    suffix = f"-{channel}" if channel else ""
    return sessions_dir / f"{ts}{suffix}.md"


def _summarize_session(
    messages: list[dict], provider: str, model: str, *, previous_summary: str = "",
) -> str:
    """Ask the model to summarize the conversation for session logging."""
    def _format_content(value: object) -> str:
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
            except (TypeError, ValueError):
                text = repr(value)
        return _escape_prompt_text(text)

    convo_text = "\n".join(
        f"{m.get('role', 'unknown')}: {_format_content(m.get('content'))}"
        for m in messages
        if m.get("content") is not None
    )
    if previous_summary:
        escaped_previous = _escape_prompt_text(previous_summary)
        prompt = (
            f"{SUMMARIZE_INCREMENTAL_PROMPT}\n\n"
            f"<previous-summary>\n{escaped_previous}\n</previous-summary>\n\n"
            f"<conversation>\n{convo_text}\n</conversation>"
        )
    else:
        prompt = f"{SUMMARIZE_PROMPT}\n\n<conversation>\n{convo_text}\n</conversation>"
    return chat([{"role": "user", "content": prompt}], provider, model, use_tools=False)


@dataclass
class SessionInfo:
    path: Path
    date: str       # "2026-02-19 23:10"
    title: str      # extracted topic line
    filename: str   # stem for reference
    channel: str = ""  # "cli", "web", "email", "telegram"


def _extract_title(path: Path) -> str:
    """Extract a short topic from a session file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return path.stem
    lines = text.splitlines()
    # Look for **Topics Discussed:** or **Discussion Topics:** or **Topic**:
    for i, line in enumerate(lines):
        if re.match(r"\*\*(Topics?\s+Discussed|Discussion\s+Topics)\s*:?\*\*", line, re.IGNORECASE):
            # Take the next non-blank line, strip "- " prefix
            for subsequent in lines[i + 1 :]:
                stripped = subsequent.strip()
                if stripped:
                    if stripped.startswith("- "):
                        stripped = stripped[2:]
                    return stripped[:80]
    # Fallback: first non-heading, non-blank line, truncated
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("**Session"):
            if stripped.startswith("- "):
                stripped = stripped[2:]
            return stripped[:80]
    return path.stem


def session_count() -> int:
    """Count total session files."""
    d = _memory_dir()
    if d is None:
        return 0
    sessions_dir = d / "sessions"
    if not sessions_dir.is_dir():
        return 0
    return len(list(sessions_dir.glob("*.md")))


_KNOWN_CHANNELS = {"cli", "web", "email", "telegram"}


def _parse_session_filename(stem: str) -> tuple[str, str]:
    """Parse date and channel from a session filename stem.

    Filenames: '2026-02-18T22-38-51' or '2026-02-18T22-38-51-cli'.
    Returns (date_str, channel).
    """
    parts = stem.split("T")
    if len(parts) != 2:
        return stem, ""
    time_part = parts[1]
    # Check for channel suffix: last segment after the time digits
    segments = time_part.split("-")
    channel = ""
    if len(segments) >= 4 and segments[-1] in _KNOWN_CHANNELS:
        channel = segments[-1]
        time_part = "-".join(segments[:-1])
    date_str = f"{parts[0]} {time_part[:5].replace('-', ':')}"
    return date_str, channel


def list_sessions(*, limit: int = 10) -> list[SessionInfo]:
    """List recent sessions with date and topic, newest first."""
    d = _memory_dir()
    if d is None:
        return []
    sessions_dir = d / "sessions"
    if not sessions_dir.is_dir():
        return []
    files = sorted(sessions_dir.glob("*.md"), key=lambda p: p.name, reverse=True)
    result = []
    for f in files[:limit]:
        date_str, channel = _parse_session_filename(f.stem)
        title = _extract_title(f)
        result.append(SessionInfo(
            path=f, date=date_str, title=title, filename=f.stem, channel=channel,
        ))
    return result


def load_session(filename: str) -> str | None:
    """Load a session file's content by filename (stem, no extension)."""
    d = _memory_dir()
    if d is None:
        return None
    path = d / "sessions" / f"{filename}.md"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _save_session(path: Path, summary: str, *, is_compaction: bool = False) -> None:
    """Write or append a session summary to the session file."""
    if path.exists():
        existing = path.read_text(encoding="utf-8", errors="replace")
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        label = "Compaction" if is_compaction else "Final"
        text = f"{existing}\n## {label} {ts}\n\n{summary}\n"
    else:
        parts = path.stem.split("T")
        if len(parts) == 2:
            ts = f"{parts[0]} {parts[1].replace('-', ':')}"
        else:
            ts = path.stem
        text = f"# Session {ts}\n\n{summary}\n"
    path.write_text(text, encoding="utf-8", errors="replace")


