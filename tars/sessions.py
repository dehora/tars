import json
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


