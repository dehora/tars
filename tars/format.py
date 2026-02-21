"""Human-readable formatting for tool results."""

import json


def _precip_icon(prob: int) -> str:
    if prob >= 70:
        return "\U0001f327"  # rain cloud
    if prob >= 30:
        return "\U0001f326"  # sun behind rain
    return ""


def format_todoist_list(raw: str) -> str:
    """Format todoist_today / todoist_upcoming JSON into a readable list."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data:
        return data["error"]
    results = data.get("results", [])
    if not results:
        return "no tasks"
    lines = []
    for i, task in enumerate(results, 1):
        content = task.get("content", "")
        priority = task.get("priority", 1)
        due = task.get("due", {})
        due_str = due.get("string", "") if due else ""
        duration = task.get("duration")
        parts = [f"{i}. [p{priority}] {content}"]
        if due_str:
            parts.append(f"(due: {due_str}")
            if duration:
                parts[-1] += f", {duration['amount']}{duration['unit'][0]}"
            parts[-1] += ")"
        lines.append(" ".join(parts))
    return "\n".join(lines)


def format_weather_now(raw: str) -> str:
    """Format weather_now JSON into a compact summary."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data:
        return data["error"]
    cur = data.get("current", {})
    temp = cur.get("temperature_c", "?")
    conditions = cur.get("conditions", "")
    wind = cur.get("wind_speed_kmh", "?")
    precip = cur.get("precipitation_mm", 0)
    line1 = f"{temp}\u00b0C, {conditions.lower()}"
    if precip and precip > 0:
        line1 += f", {precip}mm"
    line1 += f", wind {wind} km/h"
    hours = data.get("next_hours", [])
    if hours:
        parts = []
        for h in hours:
            t = h.get("time", "")[-5:]
            tc = h.get("temp_c", "?")
            prob = h.get("precip_prob_pct", 0)
            icon = _precip_icon(prob)
            if prob > 0:
                parts.append(f"{t} {tc}\u00b0{icon}{prob}%")
            else:
                parts.append(f"{t} {tc}\u00b0")
        line2 = "next: " + ", ".join(parts)
        return f"{line1}\n{line2}"
    return line1


def format_weather_forecast(raw: str) -> str:
    """Format weather_forecast JSON into an hourly table."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data:
        return data["error"]
    hourly = data.get("hourly", [])
    if not hourly:
        return "no forecast data"
    # Show every 3 hours in two columns for compactness.
    entries = []
    for h in hourly:
        t = h.get("time", "")[-5:]
        tc = h.get("temp_c", "?")
        prob = h.get("precip_prob_pct", 0)
        icon = _precip_icon(prob)
        if prob > 0:
            entries.append(f"{t} {tc:>4}\u00b0 {icon}{prob}%")
        else:
            entries.append(f"{t} {tc:>4}\u00b0")
    # Every 3 hours
    selected = entries[::3]
    mid = (len(selected) + 1) // 2
    lines = []
    for i in range(mid):
        left = selected[i] if i < len(selected) else ""
        right = selected[i + mid] if i + mid < len(selected) else ""
        if right:
            lines.append(f"{left:<22s} {right}")
        else:
            lines.append(left)
    return "\n".join(lines)


def format_todoist_action(raw: str) -> str:
    """Format todoist add/complete result."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data:
        return data["error"]
    if data.get("ok"):
        return "done"
    return raw


def format_memory_recall(raw: str) -> str:
    """Format memory_recall JSON into readable sections."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data:
        return data["error"]
    lines = []
    for section, content in data.items():
        lines.append(f"[{section}]")
        lines.append(content.strip())
        lines.append("")
    return "\n".join(lines).rstrip()


def format_web_read(raw: str) -> str:
    """Format web_read JSON into readable text."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data:
        return data["error"]
    url = data.get("url", "")
    content = data.get("content", "")
    truncated = data.get("truncated", False)
    lines = [f"[{url}]", "", content]
    if truncated:
        lines.append("\n(content truncated)")
    return "\n".join(lines)


def format_capture(raw: str) -> str:
    """Format capture result."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data:
        return data["error"]
    return f"captured: {data['title']} \u2192 {data['path']}"


def format_stats(raw: str) -> str:
    """Format system stats into readable lines."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data:
        return data["error"]
    lines = [
        f"db: {data.get('db_size_mb', '?')} MB, {data.get('files', '?')} files, {data.get('chunks', '?')} chunks",
        f"embedding: {data.get('embedding_model', '?')} ({data.get('embedding_dim', '?')}d)",
        f"sessions: {data.get('sessions', '?')}",
    ]
    return "\n".join(lines)


_FORMATTERS = {
    "todoist_today": format_todoist_list,
    "todoist_upcoming": format_todoist_list,
    "todoist_add_task": format_todoist_action,
    "todoist_complete_task": format_todoist_action,
    "weather_now": format_weather_now,
    "weather_forecast": format_weather_forecast,
    "memory_recall": format_memory_recall,
    "note_daily": format_todoist_action,
    "web_read": format_web_read,
    "capture": format_capture,
}


def format_tool_result(name: str, raw: str) -> str:
    """Format a tool result for human display. Falls back to raw if no formatter."""
    formatter = _FORMATTERS.get(name)
    if formatter:
        return formatter(raw)
    return raw
