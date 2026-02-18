import json
import re
from datetime import datetime, timedelta
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

ROLLUP_PROMPT = """\
Summarize all of today's sessions into a single daily context summary.
Include key topics, decisions, and tools used. Use bullet points. Be brief."""

CONTEXT_DATE_RE = re.compile(r"<!--\s*tars:date\s+(\d{4}-\d{2}-\d{2})\s*-->")


def _session_path() -> Path | None:
    """Returns a timestamped session file path, or None if memory not configured."""
    d = _memory_dir()
    if d is None:
        return None
    sessions_dir = d / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return sessions_dir / f"{ts}.md"


def _summarize_session(
    messages: list[dict], provider: str, model: str, *, previous_summary: str = "",
) -> str:
    """Ask the model to summarize the conversation for session logging."""
    def _escape_prompt_text(text: str) -> str:
        return text.replace("<", "&lt;").replace(">", "&gt;")

    def _format_content(value: object) -> str:
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
            except TypeError:
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
    return chat([{"role": "user", "content": prompt}], provider, model)


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


def _rollup_context(provider: str, model: str) -> None:
    """Summarize today's sessions into context/today.md, rotating yesterday.md."""
    d = _memory_dir()
    if d is None:
        return
    sessions_dir = d / "sessions"
    if not sessions_dir.is_dir():
        return

    today = datetime.now().strftime("%Y-%m-%d")
    today_files = sorted(sessions_dir.glob(f"{today}*.md"))
    if not today_files:
        return

    context_dir = d / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    today_path = context_dir / "today.md"
    yesterday_path = context_dir / "yesterday.md"

    # Rotate: if today.md exists with a different date (or no marker), move it to yesterday.md
    if today_path.exists():
        text = today_path.read_text(encoding="utf-8", errors="replace")
        match = CONTEXT_DATE_RE.search(text)
        if not match or match.group(1) != today:
            today_path.replace(yesterday_path)

    # Clean up: if yesterday.md is older than yesterday, delete it
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    if yesterday_path.exists():
        text = yesterday_path.read_text(encoding="utf-8", errors="replace")
        match = CONTEXT_DATE_RE.search(text)
        if match and match.group(1) != today and match.group(1) != yesterday:
            yesterday_path.unlink()

    # Concatenate today's session files and ask for a rollup summary
    session_texts = []
    for f in today_files:
        session_texts.append(f.read_text(encoding="utf-8", errors="replace").strip())
    combined = "\n\n---\n\n".join(session_texts)

    escaped_combined = combined.replace("<", "&lt;").replace(">", "&gt;")
    prompt = f"{ROLLUP_PROMPT}\n\n<sessions>\n{escaped_combined}\n</sessions>"
    summary = chat([{"role": "user", "content": prompt}], provider, model)

    today_path.write_text(
        f"<!-- tars:date {today} -->\n# Context {today}\n\n{summary}\n",
        encoding="utf-8",
        errors="replace",
    )
