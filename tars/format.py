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


def format_strava_activities(raw: str) -> str:
    """Format strava_activities JSON into a readable list."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(data, dict):
        if "error" in data:
            return data["error"]
        # Single activity by ID
        return _format_single_activity(data)
    if not data:
        return "no activities found"
    lines = []
    for i, a in enumerate(data, 1):
        name = a.get("name", "")
        atype = a.get("type", "")
        dist = a.get("distance_km", 0)
        moving = a.get("moving_time_min", 0)
        date = (a.get("start_date") or "")[:10]
        parts = [f"{i}. {name}"]
        if atype:
            parts[0] = f"{i}. [{atype}] {name}"
        details = [f"{dist}km", f"{moving}min"]
        pace = a.get("pace_min_per_km")
        speed = a.get("speed_kmh")
        if pace:
            mins = int(pace)
            secs = int((pace - mins) * 60)
            details.append(f"{mins}:{secs:02d}/km")
        elif speed:
            details.append(f"{speed}km/h")
        hr = a.get("average_heartrate")
        if hr:
            details.append(f"hr:{int(hr)}")
        elev = a.get("elevation_gain_m")
        if elev:
            details.append(f"+{int(elev)}m")
        parts.append(f"({', '.join(details)})")
        if date:
            parts.append(date)
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _format_single_activity(data: dict) -> str:
    """Format a single activity with detail (laps/splits)."""
    name = data.get("name", "")
    atype = data.get("type", "")
    dist = data.get("distance_km", 0)
    moving = data.get("moving_time_min", 0)
    date = (data.get("start_date") or "")[:10]
    header = f"[{atype}] {name}" if atype else name
    lines = [header, f"  {dist}km, {moving}min"]
    pace = data.get("pace_min_per_km")
    speed = data.get("speed_kmh")
    if pace:
        mins = int(pace)
        secs = int((pace - mins) * 60)
        lines[-1] += f", {mins}:{secs:02d}/km"
    elif speed:
        lines[-1] += f", {speed}km/h"
    hr = data.get("average_heartrate")
    if hr:
        lines[-1] += f", hr:{int(hr)}"
    elev = data.get("elevation_gain_m")
    if elev:
        lines[-1] += f", +{int(elev)}m"
    if date:
        lines[-1] += f"  ({date})"
    laps = data.get("laps")
    if laps:
        lines.append("  laps:")
        for lap in laps:
            n = lap.get("name", "")
            d = lap.get("distance_km", 0)
            t = lap.get("moving_time_min", 0)
            lhr = lap.get("average_heartrate")
            lap_line = f"    {n}: {d}km {t}min"
            if lhr:
                lap_line += f" hr:{int(lhr)}"
            lines.append(lap_line)
    splits = data.get("splits")
    if splits:
        lines.append("  splits:")
        for s in splits:
            sn = s.get("split", "")
            sd = s.get("distance_km", 0)
            st = s.get("moving_time_min", 0)
            shr = s.get("average_heartrate")
            split_line = f"    {sn}: {sd}km {st}min"
            if shr:
                split_line += f" hr:{int(shr)}"
            lines.append(split_line)
    return "\n".join(lines)


def format_strava_user(raw: str) -> str:
    """Format strava_user JSON into a readable summary."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data:
        return data["error"]
    lines = []
    profile = data.get("profile")
    if profile:
        name = profile.get("name", "")
        loc_parts = [p for p in [profile.get("city"), profile.get("state"), profile.get("country")] if p]
        loc = ", ".join(loc_parts)
        lines.append(name)
        if loc:
            lines.append(f"  {loc}")
        w = profile.get("weight_kg")
        if w:
            lines.append(f"  {w}kg")
    stats = data.get("stats")
    if stats:
        if lines:
            lines.append("")
        for label, key in [("YTD run", "ytd_run"), ("YTD ride", "ytd_ride"), ("YTD swim", "ytd_swim"),
                           ("All run", "all_run"), ("All ride", "all_ride"), ("All swim", "all_swim")]:
            totals = stats.get(key, {})
            count = totals.get("count", 0)
            if count == 0:
                continue
            dist = totals.get("distance_km", 0)
            hours = totals.get("moving_time_hours", 0)
            elev = totals.get("elevation_gain_m", 0)
            line = f"  {label}: {count} activities, {dist}km, {hours}h"
            if elev:
                line += f", +{int(elev)}m"
            lines.append(line)
    zones = data.get("zones")
    if zones:
        hr_zones = zones.get("heart_rate", [])
        if hr_zones:
            if lines:
                lines.append("")
            lines.append("  HR zones:")
            for i, z in enumerate(hr_zones, 1):
                zmin = z.get("min", "?")
                zmax = z.get("max", "?")
                if zmax == -1:
                    lines.append(f"    Z{i}: {zmin}+")
                else:
                    lines.append(f"    Z{i}: {zmin}-{zmax}")
    gear = data.get("gear")
    if gear:
        if lines:
            lines.append("")
        lines.append("  gear:")
        for g in gear:
            gtype = g.get("type", "")
            gname = g.get("name", "")
            gdist = g.get("distance_km", 0)
            lines.append(f"    [{gtype}] {gname}: {int(gdist)}km")
    return "\n".join(lines) if lines else raw


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
    "strava_activities": format_strava_activities,
    "strava_user": format_strava_user,
}


def format_tool_result(name: str, raw: str) -> str:
    """Format a tool result for human display. Falls back to raw if no formatter."""
    formatter = _FORMATTERS.get(name)
    if formatter:
        return formatter(raw)
    return raw
