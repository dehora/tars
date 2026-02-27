"""Multi-model routing: cheap model for chat, escalation model for tools."""

import re
import sys
from dataclasses import dataclass, field

from tars.config import ModelConfig


@dataclass(frozen=True)
class RouteResult:
    provider: str
    model: str
    tool_hints: list[str] = field(default_factory=list)


# Tool names from tars/tools.py — direct mentions always escalate.
_TOOL_NAMES = {
    "todoist_add_task", "todoist_today", "todoist_upcoming",
    "todoist_complete_task", "weather_now", "weather_forecast",
    "memory_remember", "memory_update", "memory_forget", "memory_recall",
    "memory_search", "notes_search", "note_daily", "web_read",
}

# Keyword patterns that suggest tool intent, mapped to relevant tool names.
_TOOL_HINT_PATTERNS: list[tuple[str, list[str]]] = [
    # Todoist
    (r"\badd\s+task\b",                ["todoist_add_task"]),
    (r"\btodo\b",                      ["todoist_add_task", "todoist_today"]),
    (r"\btodoist\b",                   ["todoist_add_task", "todoist_today", "todoist_upcoming"]),
    (r"\bremind\s+me\b",              ["todoist_add_task"]),
    (r"\bbuy\b",                       ["todoist_add_task"]),
    (r"\bgroceries\b",                ["todoist_add_task"]),
    (r"\btasks?\s+(today|upcoming|due)\b", ["todoist_today", "todoist_upcoming"]),
    (r"\bcomplete\s+task\b",          ["todoist_complete_task"]),
    # Weather
    (r"\bweather\b",                   ["weather_now", "weather_forecast"]),
    (r"\bforecast\b",                  ["weather_forecast"]),
    (r"\btemperature\b",              ["weather_now"]),
    (r"\bwill\s+it\s+rain\b",        ["weather_now"]),
    # Memory
    (r"\bremember\b",                  ["memory_remember"]),
    (r"\bforget\b",                    ["memory_forget"]),
    (r"\brecall\b",                    ["memory_recall"]),
    # Notes
    (r"\bnote:",                       ["note_daily"]),
    (r"\bnote\s+that\b",             ["note_daily"]),
    (r"\bjot\s+down\b",              ["note_daily"]),
    # Search
    (r"\bsearch\s+for\b",            ["memory_search"]),
    (r"\blook\s+up\b",               ["memory_search"]),
    # Notes vault
    (r"\bmy\s+notes?\b",             ["notes_search"]),
    (r"\bdaily\s+notes?\b",          ["notes_search", "note_daily"]),
    (r"\bin\s+my\s+vault\b",         ["notes_search"]),
    (r"\bobsidian\b",                 ["notes_search"]),
    # Web
    (r"https?://",                     ["web_read"]),
    (r"\bread\s+this\b",             ["web_read"]),
]

_COMPILED_HINT_PATTERNS = [(re.compile(p, re.IGNORECASE), hints) for p, hints in _TOOL_HINT_PATTERNS]


def _has_tool_intent(text: str) -> tuple[str | None, list[str]]:
    """Check whether text contains signals suggesting tool use.

    Returns (trigger_string, deduplicated_tool_hints).
    """
    lower = text.lower()
    for name in _TOOL_NAMES:
        if name in lower:
            return name, [name]

    trigger = None
    seen: set[str] = set()
    hints: list[str] = []
    for pat, pat_hints in _COMPILED_HINT_PATTERNS:
        m = pat.search(text)
        if m:
            if trigger is None:
                trigger = m.group()
            for h in pat_hints:
                if h not in seen:
                    seen.add(h)
                    hints.append(h)
    return trigger, hints


def route_message(user_input: str, config: ModelConfig) -> RouteResult:
    """Decide which provider/model to use for a message.

    Returns a RouteResult with provider, model, and tool_hints.
    """
    default_provider = config.primary_provider
    default_model = config.primary_model
    esc_provider = config.remote_provider
    esc_model = config.remote_model

    if config.routing_policy != "tool":
        print(
            f"  [router] {default_provider}:{default_model} (routing={config.routing_policy})",
            file=sys.stderr,
        )
        return RouteResult(default_provider, default_model)

    if esc_provider is None or esc_model is None:
        print(
            f"  [router] {default_provider}:{default_model} (no escalation configured)",
            file=sys.stderr,
        )
        return RouteResult(default_provider, default_model)

    if default_provider == esc_provider and default_model == esc_model:
        print(
            f"  [router] {default_provider}:{default_model} (default matches remote)",
            file=sys.stderr,
        )
        return RouteResult(default_provider, default_model)

    trigger, hints = _has_tool_intent(user_input)
    if trigger:
        print(
            f"  [router] escalating to {esc_provider}:{esc_model} (matched: {trigger!r}, hints: {hints})",
            file=sys.stderr,
        )
        return RouteResult(esc_provider, esc_model, hints)

    print(f"  [router] {default_provider}:{default_model} (no tool intent)", file=sys.stderr)
    return RouteResult(default_provider, default_model)
