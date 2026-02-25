"""Shared slash command dispatch for CLI, email, and Telegram channels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tars.format import format_tool_result
from tars.tools import run_tool

if TYPE_CHECKING:
    from tars.conversation import Conversation

_FLAGS = {"--due", "--project", "--priority"}


def _parse_todoist_add(tokens: list[str]) -> dict:
    """Parse '/todoist add content --due D --project P --priority N' into args dict.

    Flag values are greedy — they consume tokens until the next flag or end.
    This lets --due accept multi-word values like 'tomorrow 3pm'.
    """
    args: dict = {}
    content_parts: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in _FLAGS and i + 1 < len(tokens):
            flag = t[2:]  # "due", "project", "priority"
            i += 1
            val_parts: list[str] = []
            while i < len(tokens) and tokens[i] not in _FLAGS:
                val_parts.append(tokens[i])
                i += 1
            val = " ".join(val_parts)
            if flag == "priority":
                try:
                    args[flag] = int(val)
                except ValueError:
                    args[flag] = 1  # default priority
            else:
                args[flag] = val
        else:
            content_parts.append(t)
            i += 1
    args["content"] = " ".join(content_parts)
    return args


_TODOIST_PARSE_PROMPT = """\
Extract task details from this text. Return ONLY valid JSON with these fields:
- "content": the task description (required)
- "project": project name if mentioned, otherwise omit
- "due": due date if mentioned (e.g. "today", "tomorrow", "friday"), otherwise omit
- "priority": 1-4 if mentioned (4=urgent), otherwise omit

Examples:
"eggs to Groceries" → {"content": "eggs", "project": "Groceries"}
"buy milk --due tomorrow" → {"content": "buy milk", "due": "tomorrow"}
"call dentist p3" → {"content": "call dentist", "priority": 3}
"fix the bike" → {"content": "fix the bike"}

Text: """


def _parse_todoist_natural(text: str, provider: str, model: str) -> dict:
    """Use the model to extract task fields from natural language."""
    import json as _json

    from tars.core import chat

    messages = [{"role": "user", "content": f"{_TODOIST_PARSE_PROMPT}{text}"}]
    try:
        raw = chat(messages, provider, model, use_tools=False)
        # Extract JSON from response (model might wrap in ```json blocks)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = _json.loads(raw)
        if isinstance(result, dict) and result.get("content"):
            return result
    except Exception as e:
        import sys

        print(f"  todoist parse failed, using raw text: {e}", file=sys.stderr)
    return {"content": text}


def _export_conversation(conv: Conversation | None) -> str:
    """Format conversation messages as markdown for export."""
    if conv is None or not conv.messages:
        return "No conversation to export."
    lines = [f"# Conversation {conv.id}", ""]
    for msg in conv.messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"## {role}")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def dispatch(
    text: str,
    provider: str = "",
    model: str = "",
    conv: Conversation | None = None,
) -> str | None:
    """Dispatch slash commands shared across all channels.

    Returns the command result string, or None if not a recognized command.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None

    parts = stripped.split()
    cmd = parts[0]

    try:
        if cmd == "/todoist":
            return _dispatch_todoist(parts, provider, model)
        if cmd == "/weather":
            return _run_tool("weather_now", {})
        if cmd == "/forecast":
            return _run_tool("weather_forecast", {})
        if cmd == "/memory":
            return _run_tool("memory_recall", {})
        if cmd == "/remember":
            if len(parts) < 3:
                return "Usage: /remember <semantic|procedural> <text>"
            return _run_tool(
                "memory_remember",
                {"section": parts[1], "content": " ".join(parts[2:])},
            )
        if cmd == "/note":
            if len(parts) < 2:
                return "Usage: /note <text>"
            return _run_tool("note_daily", {"content": " ".join(parts[1:])})
        if cmd == "/read":
            if len(parts) < 2:
                return "Usage: /read <url>"
            return _run_tool("web_read", {"url": parts[1]})
        if cmd == "/capture":
            return _dispatch_capture(parts, provider, model)
        if cmd == "/brief":
            from tars.brief import build_brief_sections, format_brief_text

            sections = build_brief_sections()
            return format_brief_text(sections)
        if cmd == "/search":
            if len(parts) < 2:
                return "Usage: /search <query>"
            return _dispatch_search(" ".join(parts[1:]))
        if cmd == "/find":
            if len(parts) < 2:
                return "Usage: /find <query>"
            return _dispatch_find(" ".join(parts[1:]))
        if cmd == "/sessions":
            return _dispatch_sessions()
        if cmd == "/export":
            return _export_conversation(conv)
    except Exception as e:
        return f"Tool error: {e}"

    return None  # Unrecognized command


def _run_tool(name: str, args: dict) -> str:
    """Run a tool and return its formatted result."""
    raw = run_tool(name, args, quiet=True)
    return format_tool_result(name, raw)


def _dispatch_todoist(parts: list[str], provider: str, model: str) -> str:
    sub = parts[1] if len(parts) > 1 else ""
    if sub == "add" and len(parts) > 2:
        raw_text = " ".join(parts[2:])
        has_flags = any(p.startswith("--") for p in parts[2:])
        if has_flags or not provider:
            args = _parse_todoist_add(parts[2:])
        else:
            args = _parse_todoist_natural(raw_text, provider, model)
        if not args.get("content"):
            return "Usage: /todoist add <text> [--due D] [--project P] [--priority N]"
        return _run_tool("todoist_add_task", args)
    if sub == "today":
        return _run_tool("todoist_today", {})
    if sub == "upcoming":
        try:
            days = int(parts[2]) if len(parts) > 2 else 7
        except ValueError:
            return "Usage: /todoist upcoming [days]"
        return _run_tool("todoist_upcoming", {"days": days})
    if sub == "complete" and len(parts) > 2:
        return _run_tool("todoist_complete_task", {"ref": " ".join(parts[2:])})
    return "Usage: /todoist add|today|upcoming|complete ..."


def _dispatch_capture(parts: list[str], provider: str, model: str) -> str:
    if len(parts) < 2:
        return "Usage: /capture <url> [--raw]"
    from tars.capture import capture as _capture

    raw_flag = "--raw" in parts
    url = next((p for p in parts[1:] if p != "--raw"), "")
    if not url:
        return "Usage: /capture <url> [--raw]"
    result = _capture(url, provider, model, raw=raw_flag)
    return format_tool_result("capture", result)


def _dispatch_search(query: str) -> str:
    from tars.search import search

    results = search(query, mode="hybrid", limit=5)
    if not results:
        return "No results."
    lines = []
    for i, r in enumerate(results, 1):
        source = r.file_title or r.file_path
        lines.append(f"{i}. [{r.score:.3f}] {source}")
        preview = r.content.strip().splitlines()
        if preview:
            lines.append(f"   {preview[0][:80]}")
    return "\n".join(lines)


def _dispatch_find(query: str) -> str:
    from tars.search import search_notes

    results = search_notes(query, limit=5)
    if not results:
        return "No results."
    lines = []
    for i, r in enumerate(results, 1):
        source = r.file_title or r.file_path
        lines.append(f"{i}. [{r.score:.3f}] {source}")
        preview = r.content.strip().splitlines()
        if preview:
            lines.append(f"   {preview[0][:80]}")
    return "\n".join(lines)


def _dispatch_sessions() -> str:
    from tars.sessions import list_sessions

    sessions = list_sessions(limit=10)
    if not sessions:
        return "No sessions found."
    lines = [f"{s.date}  {s.title}" for s in sessions]
    return "\n".join(lines)
