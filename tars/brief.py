"""Daily brief helpers (todoist + weather)."""

import re
from datetime import datetime, timezone

from tars.colors import blue, bold, cyan, dim, red, yellow
from tars.format import format_tool_result
from tars.memory import _load_pinned
from tars.strava import _load_tokens
from tars.tools import run_tool


def build_brief_sections() -> list[tuple[str, str]]:
    """Run briefing tools and return (label, content) sections."""
    sections: list[tuple[str, str]] = []
    try:
        pinned = _load_pinned()
        if pinned.strip():
            sections.append(("pinned", pinned.strip()))
    except OSError:
        pass
    for label, tool_name in [
        ("tasks", "todoist_today"),
        ("weather", "weather_now"),
        ("forecast", "weather_forecast"),
    ]:
        try:
            raw = run_tool(tool_name, {}, quiet=True)
            sections.append((label, format_tool_result(tool_name, raw)))
        except Exception as e:
            sections.append((label, f"unavailable: {e}"))
    if _load_tokens() is not None:
        try:
            is_monday = datetime.now(timezone.utc).weekday() == 0
            if is_monday:
                tool_name = "strava_analysis"
                tool_args = {"period": "last-week"}
            else:
                tool_name = "strava_activities"
                tool_args = {"period": "1d", "limit": 5}
            raw = run_tool(tool_name, tool_args, quiet=True)
            sections.append(("strava", format_tool_result(tool_name, raw)))
        except Exception as e:
            sections.append(("strava", f"unavailable: {e}"))
    return sections


def build_daily_context() -> str:
    """Build lightweight daily context for system prompt injection.

    Fetches tasks and weather only (pinned is already in the prompt,
    strava is too slow for startup). Returns empty string on failure.
    """
    parts: list[str] = []
    for label, tool_name in [
        ("tasks", "todoist_today"),
        ("weather", "weather_now"),
    ]:
        try:
            raw = run_tool(tool_name, {}, quiet=True)
            parts.append(f"[{label}]\n{format_tool_result(tool_name, raw)}")
        except Exception:
            pass
    return "\n\n".join(parts)


def build_review_sections(provider: str, model: str) -> list[tuple[str, str]]:
    """Run memory tidy + feedback review and return (label, content) sections."""
    from tars.commands import _dispatch_tidy, _dispatch_review

    sections: list[tuple[str, str]] = []
    for label, fn in [("tidy", _dispatch_tidy), ("review", _dispatch_review)]:
        try:
            result = fn(provider, model)
            sections.append((label, result))
        except Exception as e:
            sections.append((label, f"unavailable: {e}"))
    return sections


def format_brief_text(sections: list[tuple[str, str]]) -> str:
    """Format brief sections for plain-text output (email/Telegram)."""
    lines: list[str] = []
    for label, content in sections:
        lines.append(f"[{label}]")
        for line in content.splitlines():
            lines.append(f"  {line}")
        lines.append("")
    return "\n".join(lines).strip()


_PRIORITY_COLORS = {
    "p4": lambda s: bold(red(s)),
    "p3": yellow,
    "p2": dim,
    "p1": dim,
}

_TEMP_RE = re.compile(r"(\d+\.?\d*°)")
_PRECIP_RE = re.compile(r"(🌧\d+%)")


def _colorize_task_line(line: str) -> str:
    m = re.search(r"\[(p[1-4])\]", line)
    if m:
        tag = m.group(1)
        color_fn = _PRIORITY_COLORS.get(tag, dim)
        return color_fn(line)
    return line


def _colorize_weather_line(line: str) -> str:
    line = _TEMP_RE.sub(lambda m: cyan(m.group(1)), line)
    line = _PRECIP_RE.sub(lambda m: blue(m.group(1)), line)
    return line


def format_brief_cli(sections: list[tuple[str, str]]) -> str:
    """Format brief sections with ANSI colors for CLI output."""
    lines: list[str] = []
    for label, content in sections:
        lines.append(bold(f"[{label}]"))
        for line in content.splitlines():
            if content.startswith("unavailable:"):
                lines.append(f"  {dim(line)}")
            elif label == "tasks":
                lines.append(f"  {_colorize_task_line(line)}")
            elif label in ("weather", "forecast"):
                lines.append(f"  {_colorize_weather_line(line)}")
            else:
                lines.append(f"  {line}")
        lines.append("")
    return "\n".join(lines).strip()
