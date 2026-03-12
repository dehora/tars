"""Human-readable formatting for tool results."""

import json

_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float | int], invert: bool = False) -> str:
    """Render a list of numeric values as a Unicode sparkline.

    When invert=True, lower values get taller bars (useful for pace where
    lower = faster = better).

    Returns empty string if fewer than 2 values or all values are equal.
    """
    nums = [v for v in values if v is not None]
    if len(nums) < 2:
        return ""
    lo, hi = min(nums), max(nums)
    if lo == hi:
        return _SPARK_BLOCKS[3] * len(nums)
    span = hi - lo
    last = len(_SPARK_BLOCKS) - 1
    if invert:
        return "".join(_SPARK_BLOCKS[min(int((hi - v) / span * last), last)] for v in nums)
    return "".join(_SPARK_BLOCKS[min(int((v - lo) / span * last), last)] for v in nums)


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
        task_id = task.get("id", "")
        parts = [f"{i}. [p{priority}] {content}"]
        if due_str:
            parts.append(f"(due: {due_str}")
            if duration:
                parts[-1] += f", {duration['amount']}{duration['unit'][0]}"
            parts[-1] += ")"
        if task_id:
            parts.append(f"(id:{task_id})")
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
        wt = a.get("workout_type")
        parts = [f"{i}. {name}"]
        if atype:
            tag = f"{atype}/{wt}" if wt else atype
            parts[0] = f"{i}. [{tag}] {name}"
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
        cad = a.get("average_cadence")
        if cad:
            details.append(f"cad:{cad}")
        elev = a.get("elevation_gain_m")
        if elev:
            details.append(f"+{int(elev)}m")
        parts.append(f"({', '.join(details)})")
        aid = a.get("id", "")
        if aid:
            parts.append(f"(id:{aid})")
        if date:
            parts.append(date)
        lines.append(" ".join(parts))

    # Sparklines for trends across activities (oldest→newest)
    sparks = []
    paces = [a.get("pace_min_per_km") for a in reversed(data)]
    if all(v is not None for v in paces):
        pace_spark = sparkline(paces, invert=True)
        if pace_spark:
            sparks.append(f"pace: {pace_spark}")
    hrs = [a.get("average_heartrate") for a in reversed(data)]
    if all(v is not None for v in hrs):
        hr_spark = sparkline(hrs)
        if hr_spark:
            sparks.append(f"hr: {hr_spark}")
    elevs = [a.get("elevation_gain_m") for a in reversed(data)]
    if all(v is not None for v in elevs):
        elev_spark = sparkline(elevs)
        if elev_spark:
            sparks.append(f"elev: {elev_spark}")
    if sparks:
        lines.append("  " + "  ".join(sparks))

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
    cad = data.get("average_cadence")
    if cad:
        lines[-1] += f", cad:{cad}"
    watts = data.get("average_watts")
    if watts:
        w_str = f", {int(watts)}w"
        w_avg = data.get("weighted_average_watts")
        if w_avg and w_avg != int(watts):
            w_str += f" ({int(w_avg)}w NP)"
        lines[-1] += w_str
    elev = data.get("elevation_gain_m")
    if elev:
        lines[-1] += f", +{int(elev)}m"
    suffer = data.get("suffer_score")
    if suffer:
        lines[-1] += f", effort:{suffer}"
    if date:
        lines[-1] += f"  ({date})"
    desc = data.get("description")
    if desc:
        lines.append(f"  note: {desc}")
    extras = []
    cal = data.get("calories")
    if cal:
        extras.append(f"{cal}kcal")
    rpe = data.get("perceived_exertion")
    if rpe:
        extras.append(f"RPE:{rpe}")
    if extras:
        lines.append(f"  {', '.join(extras)}")
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
        split_paces = [
            s["moving_time_min"] / s["distance_km"]
            for s in splits
            if s.get("distance_km", 0) > 0 and s.get("moving_time_min") is not None
        ]
        split_hrs = [s.get("average_heartrate") for s in splits]
        sparks = []
        sp = sparkline(split_paces, invert=True)
        if sp:
            sparks.append(f"pace: {sp}")
        sh = sparkline(split_hrs)
        if sh:
            sparks.append(f"hr: {sh}")
        if sparks:
            lines.append("    " + "  ".join(sparks))
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


def format_strava_summary(raw: str) -> str:
    """Format strava_summary JSON into a compact per-type breakdown."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data:
        return data["error"]

    period = data.get("period", "")
    total_count = data.get("count", 0)
    by_type = data.get("by_type", {})

    if total_count == 0:
        return f"no activities for {period}"

    lines = [f"{period}: {total_count} activities"]

    for atype, stats in by_type.items():
        count = stats.get("count", 0)
        dist = stats.get("total_distance_km", 0)
        hours = stats.get("total_time_hours", 0)
        elev = stats.get("total_elevation_m", 0)

        parts = [f"{dist}km", f"{hours}h"]

        pace = stats.get("avg_pace_min_per_km")
        speed = stats.get("avg_speed_kmh")
        if pace:
            mins = int(pace)
            secs = int((pace - mins) * 60)
            parts.append(f"avg {mins}:{secs:02d}/km")
        elif speed:
            parts.append(f"avg {speed}km/h")

        if elev:
            parts.append(f"+{elev}m")

        hr = stats.get("avg_heartrate")
        if hr:
            parts.append(f"hr:{int(hr)}")

        cad = stats.get("avg_cadence")
        if cad:
            parts.append(f"cad:{cad}")

        suffer = stats.get("avg_suffer_score")
        if suffer:
            parts.append(f"effort:{suffer:.0f}")

        lines.append(f"  {atype} x{count}: {', '.join(parts)}")

    return "\n".join(lines)


def _delta_str(change: float, pct: float | None) -> str:
    sign = "+" if change >= 0 else ""
    s = f"{sign}{change:.1f}"
    if pct is not None:
        s += f" / {sign}{pct:.1f}%"
    return s


def _pace_fmt(pace: float) -> str:
    mins = int(pace)
    secs = int((pace - mins) * 60)
    return f"{mins}:{secs:02d}"


def format_strava_compare(raw: str) -> str:
    """Format strava_compare JSON into a readable comparison."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data:
        return data["error"]

    pa = data.get("period_a", "?")
    pb = data.get("period_b", "?")
    count_a = data.get("count_a", 0)
    count_b = data.get("count_b", 0)
    by_type = data.get("by_type", {})

    if count_a == 0 and count_b == 0:
        return f"{pa} vs {pb}: no activities in either period"

    lines = [f"{pa} vs {pb} ({count_a} vs {count_b} activities)"]

    for atype, entry in by_type.items():
        sa = entry.get("period_a", {})
        sb = entry.get("period_b", {})
        delta = entry.get("delta", {})

        ca = sa.get("count", 0)
        cb = sb.get("count", 0)
        lines.append(f"  {atype}: {ca} vs {cb}")

        for label, key in [("distance", "total_distance_km"), ("time", "total_time_hours"),
                           ("elevation", "total_elevation_m")]:
            va = sa.get(key)
            vb = sb.get(key)
            if va is not None or vb is not None:
                d = delta.get(key, {})
                suffix = "km" if "distance" in key else ("h" if "time" in key else "m")
                val_str = f"{va}{suffix}" if va is not None else "—"
                change = d.get("change")
                pct = d.get("pct")
                if change is not None:
                    lines.append(f"    {label}: {val_str} ({_delta_str(change, pct)})")
                else:
                    lines.append(f"    {label}: {val_str}")

        pace_a = sa.get("avg_pace_min_per_km")
        pace_b = sb.get("avg_pace_min_per_km")
        if pace_a is not None:
            d = delta.get("avg_pace_min_per_km", {})
            change = d.get("change")
            direction = ""
            if change is not None:
                direction = " faster" if change < 0 else " slower" if change > 0 else ""
            lines.append(f"    pace: {_pace_fmt(pace_a)}/km{direction}")

        speed_a = sa.get("avg_speed_kmh")
        if speed_a is not None:
            d = delta.get("avg_speed_kmh", {})
            change = d.get("change")
            pct = d.get("pct")
            extra = ""
            if change is not None:
                extra = f" ({_delta_str(change, pct)})"
            lines.append(f"    speed: {speed_a}km/h{extra}")

        hr_a = sa.get("avg_heartrate")
        if hr_a is not None:
            d = delta.get("avg_heartrate", {})
            change = d.get("change")
            pct = d.get("pct")
            extra = ""
            if change is not None:
                extra = f" ({_delta_str(change, pct)})"
            lines.append(f"    hr: {int(hr_a)}{extra}")

        cad_a = sa.get("avg_cadence")
        if cad_a is not None:
            lines.append(f"    cadence: {cad_a}")

    return "\n".join(lines)


def format_strava_analysis(raw: str) -> str:
    """Format strava_analysis JSON into a compact analysis block."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data:
        return data["error"]

    period = data.get("period", "?")
    dates = data.get("period_dates", {})
    count = data.get("count", 0)
    overall = data.get("overall", {})
    by_type = data.get("by_type", {})

    if count == 0 and data.get("compare_count", 0) == 0:
        return f"no activities for {period}"

    # Period header with cross-type totals
    date_range = ""
    if dates.get("after") and dates.get("before"):
        date_range = f" ({dates['after']} – {dates['before']})"
    dist = overall.get("total_distance_km", 0)
    hours = overall.get("total_time_hours", 0)
    elev = overall.get("total_elevation_m", 0)
    header_parts = [f"{count} activities", f"{dist}km", f"{hours}h"]
    if elev:
        header_parts.append(f"+{elev}m")
    lines = [f"{period}{date_range}: {', '.join(header_parts)}"]

    # Per-type lines
    for atype, stats in by_type.items():
        lines.append(_format_type_line(atype, stats))

    # Compare period
    compare_period = data.get("compare_period")
    compare_dates = data.get("compare_period_dates", {})
    compare_count = data.get("compare_count", 0)
    compare_overall = data.get("compare_overall", {})
    compare_by_type = data.get("compare_by_type", {})

    if compare_period is not None:
        date_range_b = ""
        if compare_dates.get("after") and compare_dates.get("before"):
            date_range_b = f" ({compare_dates['after']} – {compare_dates['before']})"
        cdist = compare_overall.get("total_distance_km", 0)
        chours = compare_overall.get("total_time_hours", 0)
        celev = compare_overall.get("total_elevation_m", 0)
        cparts = [f"{compare_count} activities", f"{cdist}km", f"{chours}h"]
        if celev:
            cparts.append(f"+{celev}m")
        lines.append(f"{compare_period}{date_range_b}: {', '.join(cparts)}")

        for atype, stats in compare_by_type.items():
            lines.append(_format_type_line(atype, stats))

    # Changes line
    overall_delta = data.get("overall_delta", {})
    if overall_delta:
        change_parts = []
        for label, key, suffix in [
            ("distance", "total_distance_km", "km"),
            ("time", "total_time_hours", "h"),
            ("elevation", "total_elevation_m", "m"),
        ]:
            d = overall_delta.get(key)
            if d:
                change = d.get("change", 0)
                pct = d.get("pct")
                sign = "+" if change >= 0 else ""
                s = f"{label} {sign}{change:.1f}{suffix}"
                if pct is not None:
                    s += f"/{sign}{pct:.0f}%"
                change_parts.append(s)
        if change_parts:
            lines.append(f"Changes: {', '.join(change_parts)}")

    return "\n".join(lines)


def _format_type_line(atype: str, stats: dict) -> str:
    """Format a single activity type summary line."""
    count = stats.get("count", 0)
    dist = stats.get("total_distance_km", 0)
    hours = stats.get("total_time_hours", 0)
    parts = [f"{dist}km", f"{hours}h"]

    pace = stats.get("avg_pace_min_per_km")
    speed = stats.get("avg_speed_kmh")
    if pace:
        parts.append(f"{_pace_fmt(pace)}/km")
    elif speed:
        parts.append(f"{speed}km/h")

    hr = stats.get("avg_heartrate")
    if hr:
        parts.append(f"hr:{int(hr)}")

    return f"  {atype} x{count}: {', '.join(parts)}"


def format_strava_routes(raw: str) -> str:
    """Format strava_routes JSON into a readable list or detail view."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data:
        return data["error"]

    # Single route detail
    if "id" in data and "name" in data:
        return _format_route_detail(data)

    # Route list
    if "routes" in data:
        routes = data["routes"]
        if not routes:
            return "no routes found"
        lines = []
        for i, r in enumerate(routes, 1):
            rtype = r.get("type") or "?"
            sub = r.get("sub_type") or ""
            tag = f"{rtype}/{sub}" if sub else rtype
            name = r.get("name", "")
            dist = r.get("distance_km", 0)
            elev = r.get("elevation_gain_m", 0)
            est = r.get("estimated_time_min")
            starred = " *" if r.get("starred") else ""
            parts = [f"{dist}km", f"+{int(elev)}m"]
            if est is not None:
                parts.append(f"~{int(est)}min")
            rid = r.get("id", "")
            lines.append(f"{i}. [{tag}] {name}{starred} — {', '.join(parts)} (id:{rid})")
        return "\n".join(lines)

    # Starred segments
    if "segments" in data:
        segs = data["segments"]
        if not segs:
            return "no starred segments"
        lines = []
        for i, s in enumerate(segs, 1):
            name = s.get("name", "")
            dist = s.get("distance_km", 0)
            grade = s.get("average_grade", 0)
            cat = s.get("climb_category", 0)
            parts = [f"{dist}km", f"{grade}% avg"]
            if cat > 0:
                parts.append(f"cat {cat}")
            pr = s.get("pr")
            if pr:
                t = pr.get("time_sec", 0)
                pr_min = int(t // 60)
                pr_sec = int(t % 60)
                parts.append(f"PR {pr_min}:{pr_sec:02d}")
            loc_parts = [s.get(k) for k in ("city", "state", "country") if s.get(k)]
            if loc_parts:
                parts.append(", ".join(loc_parts))
            lines.append(f"{i}. {name} — {', '.join(parts)}")
        return "\n".join(lines)

    return raw


def _format_route_detail(data: dict) -> str:
    """Format a single route with optional segments."""
    rtype = data.get("type") or "?"
    sub = data.get("sub_type") or ""
    tag = f"{rtype}/{sub}" if sub else rtype
    name = data.get("name", "")
    dist = data.get("distance_km", 0)
    elev = data.get("elevation_gain_m", 0)
    est = data.get("estimated_time_min")
    starred = " *" if data.get("starred") else ""

    lines = [f"[{tag}] {name}{starred}"]
    parts = [f"{dist}km", f"+{int(elev)}m"]
    if est is not None:
        parts.append(f"~{int(est)}min")
    lines.append(f"  {', '.join(parts)}")

    desc = data.get("description")
    if desc:
        lines.append(f"  {desc}")

    segments = data.get("segments", [])
    if segments:
        lines.append("  segments:")
        for s in segments:
            sname = s.get("name", "")
            sdist = s.get("distance_km", 0)
            grade = s.get("average_grade", 0)
            cat = s.get("climb_category", 0)
            sparts = [f"{sdist}km", f"{grade}% avg"]
            if cat > 0:
                sparts.append(f"cat {cat}")
            pr = s.get("pr")
            if pr:
                t = pr.get("time_sec", 0)
                pr_min = int(t // 60)
                pr_sec = int(t % 60)
                sparts.append(f"PR {pr_min}:{pr_sec:02d}")
            lines.append(f"    {sname}: {', '.join(sparts)}")

    return "\n".join(lines)


_ZONE_INSIGHTS = {
    "Polarised": "classic polarised — most time easy, hard sessions are hard.",
    "Pyramidal": "pyramidal — well-distributed, most time easy.",
    "Threshold-Heavy": "moderate zone is elevated — check for zone 3 trap.",
    "Unstructured": "no clear pattern — consider structuring intensity.",
}


def _zone_bar(pct: float, width: int = 20) -> str:
    """Render a percentage as a bar of block characters."""
    filled = round(pct / 100 * width)
    filled = max(0, min(width, filled))
    return "\u2588" * filled + "\u2591" * (width - filled)


def format_strava_zones(raw: str) -> str:
    """Format strava_zones JSON into a zone distribution chart."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if "error" in data and "classification" not in data:
        return data["error"]

    period = data.get("period", "?")
    classification = data.get("classification", "?")
    zone_pct = data.get("zone_pct", {})
    total_hours = data.get("total_hours", 0)
    analysed = data.get("activities_analysed", 0)
    skipped = data.get("activities_skipped", {})
    boundaries = data.get("zone_boundaries", {})

    total_activities = analysed + skipped.get("no_hr", 0) + skipped.get("too_short", 0) + skipped.get("over_cap", 0)

    lines = [f"Training Zones ({period})"]
    lines.append(f"  {analysed} activities \u00b7 {total_hours}h \u00b7 HR data: {analysed}/{total_activities}")
    lines.append("")

    low = zone_pct.get("low", 0)
    mod = zone_pct.get("mod", 0)
    high = zone_pct.get("high", 0)

    lines.append(f"  Low  {_zone_bar(low)} {low:4.0f}%")
    lines.append(f"  Mod  {_zone_bar(mod)} {mod:4.0f}%")
    lines.append(f"  High {_zone_bar(high)} {high:4.0f}%")

    lines.append("")
    insight = _ZONE_INSIGHTS.get(classification, "")
    lines.append(f"  \u2192 {classification}: {insight}")

    low_max = boundaries.get("low_max")
    mod_max = boundaries.get("mod_max")
    if low_max and mod_max:
        lines.append(f"  Low: <{low_max}  Mod: {low_max}-{mod_max - 1}  High: >={mod_max}")

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
    "strava_activities": format_strava_activities,
    "strava_user": format_strava_user,
    "strava_summary": format_strava_summary,
    "strava_compare": format_strava_compare,
    "strava_analysis": format_strava_analysis,
    "strava_routes": format_strava_routes,
    "strava_zones": format_strava_zones,
}


def format_tool_result(name: str, raw: str) -> str:
    """Format a tool result for human display. Falls back to raw if no formatter."""
    formatter = _FORMATTERS.get(name)
    if formatter:
        return formatter(raw)
    return raw
