"""Daily brief helpers (todoist + weather)."""

import re

from tars.colors import blue, bold, cyan, dim, red, yellow
from tars.format import format_tool_result
from tars.tools import run_tool


def build_brief_sections() -> list[tuple[str, str]]:
    """Run briefing tools and return (label, content) sections."""
    sections: list[tuple[str, str]] = []
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

_TEMP_RE = re.compile(r"(\d+°)")
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
