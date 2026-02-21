"""Multi-model routing: cheap model for chat, escalation model for tools."""

import re
import sys

from tars.core import escalation_config

# Tool names from tars/tools.py — direct mentions always escalate.
_TOOL_NAMES = {
    "todoist_add_task", "todoist_today", "todoist_upcoming",
    "todoist_complete_task", "weather_now", "weather_forecast",
    "memory_remember", "memory_update", "memory_forget", "memory_recall",
    "memory_search", "note_daily", "web_read",
}

# Keyword patterns that suggest tool intent.  Each pattern is compiled as
# case-insensitive and checked against the full message text.
_TOOL_PATTERNS = [
    # Todoist
    r"\badd\s+task\b", r"\btodo\b", r"\btodoist\b", r"\bremind\s+me\b",
    r"\bbuy\b", r"\bgroceries\b", r"\btasks?\s+(today|upcoming|due)\b",
    r"\bcomplete\s+task\b",
    # Weather
    r"\bweather\b", r"\bforecast\b", r"\btemperature\b",
    r"\bwill\s+it\s+rain\b",
    # Memory
    r"\bremember\b", r"\bforget\b", r"\brecall\b",
    # Notes
    r"\bnote:", r"\bnote\s+that\b", r"\bjot\s+down\b",
    # Search
    r"\bsearch\s+for\b", r"\blook\s+up\b",
    # Web
    r"https?://", r"\bread\s+this\b",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _TOOL_PATTERNS]


def _has_tool_intent(text: str) -> str | None:
    """Check whether text contains signals suggesting tool use.

    Returns the matched trigger string, or None.
    """
    lower = text.lower()
    for name in _TOOL_NAMES:
        if name in lower:
            return name
    for pat, raw in zip(_COMPILED_PATTERNS, _TOOL_PATTERNS):
        m = pat.search(text)
        if m:
            return m.group()
    return None


def route_message(
    user_input: str, default_provider: str, default_model: str,
) -> tuple[str, str]:
    """Decide which provider/model to use for a message.

    Returns (provider, model) — either the default or the escalation target.
    """
    esc = escalation_config()
    if esc is None:
        print(f"  [router] {default_provider}:{default_model} (no escalation configured)", file=sys.stderr)
        return default_provider, default_model

    # No point escalating if default is already claude
    if default_provider == "claude":
        print(f"  [router] {default_provider}:{default_model} (default is already claude)", file=sys.stderr)
        return default_provider, default_model

    trigger = _has_tool_intent(user_input)
    if trigger:
        esc_provider, esc_model = esc
        print(f"  [router] escalating to {esc_provider}:{esc_model} (matched: {trigger!r})", file=sys.stderr)
        return esc_provider, esc_model

    print(f"  [router] {default_provider}:{default_model} (no tool intent)", file=sys.stderr)
    return default_provider, default_model
