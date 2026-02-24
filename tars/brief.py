"""Daily brief helpers (todoist + weather)."""

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
    """Format brief sections for plain-text output (email/CLI)."""
    lines: list[str] = []
    for label, content in sections:
        lines.append(f"[{label}]")
        for line in content.splitlines():
            lines.append(f"  {line}")
        lines.append("")
    return "\n".join(lines).strip()
