"""Strava integration — OAuth token lifecycle and tool dispatch."""

import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tars.memory import _memory_dir

_VALID_ACTIVITY_TYPES = {
    "Run", "Ride", "Swim", "Walk", "Hike", "VirtualRide", "VirtualRun",
    "EBikeRide", "Handcycle", "Wheelchair", "WeightTraining", "Yoga",
    "Workout", "Rowing", "Canoeing", "Kayaking", "StandUpPaddling",
    "Surfing", "Crossfit", "Elliptical", "RockClimbing", "StairStepper",
    "Velomobile", "Windsurf", "Kitesurf", "Snowboard", "Ski",
    "NordicSki", "BackcountrySki", "AlpineSki", "IceSkate",
    "InlineSkate", "Skateboard", "Golf", "Soccer", "Tennis",
    "Badminton", "Pickleball", "TableTennis", "Squash", "RacquetBall",
    "Trail Run",
}

_VALID_SORT = {"recent", "oldest"}

_VALID_USER_SECTIONS = {"profile", "stats", "zones", "gear"}

_WORKOUT_TYPE_LABELS = {1: "race", 2: "long_run", 3: "workout", 11: "race", 12: "workout"}

_PACE_TYPES = {"Run", "Walk", "Hike"}
_SPEED_TYPES = {"Ride", "VirtualRide", "EBikeRide"}

_ROUTE_TYPE = {1: "Ride", 2: "Run"}
_ROUTE_SUB_TYPE = {1: "road", 2: "MTB", 3: "cross", 4: "trail", 5: "mixed"}
_VALID_ROUTE_ACTIONS = {"list", "detail", "starred"}

_TOKEN_FILE = "strava_tokens.json"


def _safe_float(val, default=0.0) -> float:
    """Coerce a value to float, returning default if None."""
    return float(val) if val is not None else default


def _safe_int(val, default=0) -> int:
    """Coerce a value to int, returning default if None."""
    return int(val) if val is not None else default


def _token_path() -> Path | None:
    d = _memory_dir()
    if d is None:
        return None
    return d / _TOKEN_FILE


def _load_tokens() -> dict | None:
    p = _token_path()
    if p is None or not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8", errors="replace"))


def _save_tokens(tokens: dict) -> None:
    p = _token_path()
    if p is None:
        raise RuntimeError("TARS_MEMORY_DIR not configured")
    p.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    os.chmod(p, 0o600)


def _get_client():
    """Return an authenticated stravalib Client, refreshing tokens if needed."""
    os.environ.setdefault("SILENCE_TOKEN_WARNINGS", "true")
    import stravalib

    tokens = _load_tokens()
    if tokens is None:
        raise RuntimeError(
            "Strava not authenticated — run `uv run tars strava-auth` first"
        )

    if tokens.get("expires_at", 0) < time.time():
        client_id = os.environ.get("TARS_STRAVA_CLIENT_ID", "").strip()
        client_secret = os.environ.get("TARS_STRAVA_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            raise RuntimeError(
                "TARS_STRAVA_CLIENT_ID and TARS_STRAVA_CLIENT_SECRET required for token refresh"
            )
        client = stravalib.Client()
        fresh = client.refresh_access_token(
            client_id=int(client_id),
            client_secret=client_secret,
            refresh_token=tokens["refresh_token"],
        )
        tokens = {
            "access_token": fresh["access_token"],
            "refresh_token": fresh["refresh_token"],
            "expires_at": fresh["expires_at"],
        }
        _save_tokens(tokens)

    client = stravalib.Client(access_token=tokens["access_token"])
    return client


def strava_auth_flow(client_id: str, client_secret: str) -> None:
    """One-time OAuth setup: print URL, accept code, save tokens."""
    import stravalib

    client = stravalib.Client()
    url = client.authorization_url(
        client_id=int(client_id),
        redirect_uri="http://localhost",
        scope=["read_all", "activity:read_all", "profile:read_all"],
    )
    print(f"\n  Open this URL in your browser:\n\n  {url}\n")
    print("  After authorizing, paste the 'code' parameter from the redirect URL:")

    try:
        code = input("  code> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  cancelled")
        return

    if not code:
        print("  no code provided, aborting")
        return

    token_response = client.exchange_code_for_token(
        client_id=int(client_id),
        client_secret=client_secret,
        code=code,
    )
    tokens = {
        "access_token": token_response["access_token"],
        "refresh_token": token_response["refresh_token"],
        "expires_at": token_response["expires_at"],
    }
    _save_tokens(tokens)
    print(f"  tokens saved to {_token_path()}")


_PERIOD_RE = re.compile(r"^(\d+)([dwmy])$")


def _parse_period(period: str) -> tuple[datetime, datetime] | str:
    """Parse a user-friendly period string into (after, before) UTC datetimes.

    Returns an error string on invalid input.
    """
    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if period == "this-week":
        after = today - timedelta(days=today.weekday())
        return (after, now)
    if period == "last-week":
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=7)
        return (start, end)
    if period == "this-month":
        after = today.replace(day=1)
        return (after, now)
    if period == "last-month":
        first = today.replace(day=1)
        prev_last = first - timedelta(days=1)
        after = prev_last.replace(day=1)
        return (after, first)
    if period in ("this-year", "ytd"):
        after = today.replace(month=1, day=1)
        return (after, now)

    m = _PERIOD_RE.match(period)
    if not m:
        return f"invalid period: {period!r} — use e.g. 7d, 3m, this-week, ytd"

    n = int(m.group(1))
    unit = m.group(2)
    if unit == "d":
        after = now - timedelta(days=n)
    elif unit == "w":
        after = now - timedelta(weeks=n)
    elif unit == "m":
        month = today.month - n
        year = today.year
        while month < 1:
            month += 12
            year -= 1
        after = today.replace(year=year, month=month, day=1)
    elif unit == "y":
        after = today.replace(year=today.year - n, month=1, day=1)
    else:
        return f"invalid period unit: {unit!r}"

    return (after, now)


def _type_str(atype) -> str:
    """Extract activity type string, unwrapping stravalib RootModel if needed."""
    if atype is None:
        return ""
    if hasattr(atype, "root"):
        return str(atype.root)
    return str(atype)


def _activity_to_dict(a) -> dict:
    """Extract activity summary fields into a plain dict."""
    distance_m = _safe_float(a.distance)
    distance_km = round(distance_m / 1000, 2)
    moving_secs = _safe_int(a.moving_time)
    elapsed_secs = _safe_int(a.elapsed_time)
    moving_min = round(moving_secs / 60, 1)
    elapsed_min = round(elapsed_secs / 60, 1)
    elev_m = round(_safe_float(a.total_elevation_gain), 1) if a.total_elevation_gain is not None else None
    avg_hr = _safe_float(a.average_heartrate) if a.average_heartrate is not None else None
    max_hr = _safe_float(a.max_heartrate) if a.max_heartrate is not None else None

    result = {
        "id": a.id,
        "name": a.name,
        "type": _type_str(a.type) if a.type else None,
        "distance_km": distance_km,
        "moving_time_min": moving_min,
        "elapsed_time_min": elapsed_min,
        "elevation_gain_m": elev_m,
        "average_heartrate": avg_hr,
        "max_heartrate": max_hr,
        "start_date": a.start_date_local.isoformat() if a.start_date_local else None,
        "suffer_score": a.suffer_score,
    }

    wt = getattr(a, "workout_type", None)
    if wt is not None and wt != 0:
        label = _WORKOUT_TYPE_LABELS.get(wt)
        if label:
            result["workout_type"] = label

    cad = getattr(a, "average_cadence", None)
    if cad is not None:
        result["average_cadence"] = round(float(cad), 1)

    watts = getattr(a, "average_watts", None)
    if watts is not None:
        result["average_watts"] = round(float(watts), 0)
    w_avg = getattr(a, "weighted_average_watts", None)
    if w_avg is not None:
        result["weighted_average_watts"] = int(w_avg)

    activity_type = _type_str(a.type) if a.type else ""
    if activity_type in _PACE_TYPES:
        if distance_km > 0 and moving_secs > 0:
            result["pace_min_per_km"] = round(moving_secs / 60 / distance_km, 2)
    elif activity_type in _SPEED_TYPES:
        if moving_secs > 0:
            result["speed_kmh"] = round(distance_km / (moving_secs / 3600), 1)

    return result


def _lap_to_dict(lap) -> dict:
    distance_m = _safe_float(lap.distance)
    moving_secs = _safe_int(lap.moving_time)
    return {
        "name": lap.name,
        "distance_km": round(distance_m / 1000, 2),
        "moving_time_min": round(moving_secs / 60, 1),
        "average_heartrate": _safe_float(lap.average_heartrate) if lap.average_heartrate is not None else None,
        "max_heartrate": _safe_float(lap.max_heartrate) if lap.max_heartrate is not None else None,
    }


def _split_to_dict(split) -> dict:
    distance_m = _safe_float(split.distance)
    moving_secs = _safe_int(split.moving_time)
    avg_hr = getattr(split, "average_heartrate", None)
    elev_diff = getattr(split, "elevation_difference", None)
    return {
        "split": split.split,
        "distance_km": round(distance_m / 1000, 2),
        "moving_time_min": round(moving_secs / 60, 1),
        "average_heartrate": _safe_float(avg_hr) if avg_hr is not None else None,
        "elevation_difference_m": _safe_float(elev_diff) if elev_diff is not None else None,
    }


def _totals_to_dict(totals) -> dict:
    if totals is None:
        return {}
    distance_m = _safe_float(totals.distance)
    moving_secs = _safe_int(totals.moving_time)
    elev_m = _safe_float(totals.elevation_gain)
    return {
        "count": totals.count,
        "distance_km": round(distance_m / 1000, 1),
        "moving_time_hours": round(moving_secs / 3600, 1),
        "elevation_gain_m": round(elev_m, 0),
    }


def _run_strava_tool(name: str, args: dict) -> str:
    """Dispatch strava tool calls. Returns JSON string."""
    try:
        client = _get_client()
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    try:
        if name == "strava_activities":
            return _handle_activities(client, args)
        elif name == "strava_user":
            return _handle_user(client, args)
        elif name == "strava_summary":
            return _handle_summary(client, args)
        elif name == "strava_compare":
            return _handle_compare(client, args)
        elif name == "strava_analysis":
            return _handle_analysis(client, args)
        elif name == "strava_routes":
            return _handle_routes(client, args)
        else:
            return json.dumps({"error": f"unknown strava tool: {name}"})
    finally:
        if hasattr(client, "protocol") and hasattr(client.protocol, "rsession"):
            client.protocol.rsession.close()


def _handle_activities(client, args: dict) -> str:
    try:
        activity_id = args.get("id")
        if activity_id is not None:
            a = client.get_activity(int(activity_id))
            result = _activity_to_dict(a)
            # DetailedActivity-only fields (not on SummaryActivity)
            if getattr(a, "calories", None) is not None:
                result["calories"] = int(a.calories)
            desc = getattr(a, "description", None)
            if desc:
                result["description"] = desc
            if getattr(a, "perceived_exertion", None) is not None:
                result["perceived_exertion"] = a.perceived_exertion
            if hasattr(a, "laps") and a.laps:
                result["laps"] = [_lap_to_dict(lap) for lap in a.laps]
            if hasattr(a, "splits_metric") and a.splits_metric:
                result["splits"] = [_split_to_dict(s) for s in a.splits_metric]
            return json.dumps(result)

        limit = max(1, min(100, int(args.get("limit", 20))))
        sort = args.get("sort", "recent")
        if sort not in _VALID_SORT:
            return json.dumps({"error": f"invalid sort: {sort!r} — use 'recent' or 'oldest'"})

        activity_type = args.get("type")
        if activity_type is not None and activity_type not in _VALID_ACTIVITY_TYPES:
            return json.dumps({"error": f"invalid activity type: {activity_type!r}"})

        kwargs = {"limit": limit}
        period = args.get("period")
        if period:
            parsed = _parse_period(period)
            if isinstance(parsed, str):
                return json.dumps({"error": parsed})
            kwargs["after"] = parsed[0]
            kwargs["before"] = parsed[1]

        activities = list(client.get_activities(**kwargs))

        if activity_type:
            activities = [a for a in activities if _type_str(a.type) == activity_type]

        if sort == "oldest":
            activities.reverse()

        return json.dumps([_activity_to_dict(a) for a in activities])

    except Exception as e:
        return json.dumps({"error": f"Strava API error: {e}"})


def _handle_user(client, args: dict) -> str:
    try:
        include = args.get("include", ["profile", "stats"])
        if isinstance(include, str):
            include = [include]
        invalid = set(include) - _VALID_USER_SECTIONS
        if invalid:
            return json.dumps({"error": f"invalid sections: {invalid!r} — use profile, stats, zones, gear"})

        result = {}

        needs_athlete = {"profile", "stats", "gear"} & set(include)
        athlete = client.get_athlete() if needs_athlete else None

        if "profile" in include:
            result["profile"] = {
                "name": f"{athlete.firstname} {athlete.lastname}".strip(),
                "city": athlete.city,
                "state": athlete.state,
                "country": athlete.country,
                "weight_kg": athlete.weight,
                "premium": athlete.premium,
            }

        if "stats" in include:
            stats = client.get_athlete_stats(athlete.id)
            result["stats"] = {
                "ytd_run": _totals_to_dict(stats.ytd_run_totals),
                "ytd_ride": _totals_to_dict(stats.ytd_ride_totals),
                "ytd_swim": _totals_to_dict(stats.ytd_swim_totals),
                "all_run": _totals_to_dict(stats.all_run_totals),
                "all_ride": _totals_to_dict(stats.all_ride_totals),
                "all_swim": _totals_to_dict(stats.all_swim_totals),
                "recent_run": _totals_to_dict(stats.recent_run_totals),
                "recent_ride": _totals_to_dict(stats.recent_ride_totals),
                "recent_swim": _totals_to_dict(stats.recent_swim_totals),
            }

        if "zones" in include:
            zones = client.get_athlete_zones()
            zone_list = []
            if hasattr(zones, "heart_rate") and zones.heart_rate:
                hr = zones.heart_rate
                hr_zones = getattr(hr, "zones", None) or getattr(hr, "root", None)
                if hasattr(hr_zones, "root"):
                    hr_zones = hr_zones.root
                if hr_zones:
                    for z in hr_zones:
                        if isinstance(z, (tuple, list)):
                            zone_list.append({"min": z[0], "max": z[1]})
                        elif hasattr(z, "min"):
                            zone_list.append({"min": z.min, "max": z.max})
                        else:
                            zone_list.append({"min": z.root[0], "max": z.root[1]})
            result["zones"] = {"heart_rate": zone_list}

        if "gear" in include:
            gear = []
            if hasattr(athlete, "bikes") and athlete.bikes:
                for b in athlete.bikes:
                    gear.append({
                        "type": "bike",
                        "name": b.name,
                        "distance_km": round(_safe_float(b.distance) / 1000, 0),
                    })
            if hasattr(athlete, "shoes") and athlete.shoes:
                for s in athlete.shoes:
                    gear.append({
                        "type": "shoe",
                        "name": s.name,
                        "distance_km": round(_safe_float(s.distance) / 1000, 0),
                    })
            result["gear"] = gear

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"Strava API error: {e}"})


def _default_comparison_period(period_a_str: str, parsed_a: tuple[datetime, datetime]) -> tuple[datetime, datetime] | str:
    """Derive the comparison period from period_a. Returns (after, before) or error string."""
    after_a, before_a = parsed_a

    if period_a_str == "this-week":
        return _parse_period("last-week")
    if period_a_str == "last-week":
        after = after_a - timedelta(days=7)
        before = after_a
        return (after, before)
    if period_a_str == "this-month":
        return _parse_period("last-month")
    if period_a_str == "last-month":
        prev_last = after_a - timedelta(days=1)
        after = prev_last.replace(day=1)
        return (after, after_a)
    if period_a_str in ("this-year", "ytd"):
        after = after_a.replace(year=after_a.year - 1)
        before = before_a.replace(year=before_a.year - 1)
        return (after, before)

    m = _PERIOD_RE.match(period_a_str)
    if m:
        span = before_a - after_a
        after = after_a - span
        before = after_a
        return (after, before)

    return f"cannot auto-derive comparison for {period_a_str!r}"


def _compute_delta(a: dict, b: dict) -> dict:
    """Compute absolute and percentage change between two summary dicts (a - b)."""
    delta = {}
    for key in a:
        va = a[key]
        vb = b.get(key)
        if not isinstance(va, (int, float)) or vb is None or not isinstance(vb, (int, float)):
            continue
        change = round(va - vb, 2)
        entry: dict = {"change": change}
        if vb != 0:
            entry["pct"] = round(change / abs(vb) * 100, 1)
        delta[key] = entry
    return delta


def _summarise_group(atype: str, activities: list) -> dict:
    """Compute aggregate stats for a list of same-type activities."""
    total_dist_m = 0.0
    total_moving_secs = 0
    total_elev_m = 0.0
    hr_sum = 0.0
    hr_count = 0
    cad_sum = 0.0
    cad_count = 0
    suffer_total = 0
    suffer_count = 0

    for a in activities:
        total_dist_m += _safe_float(a.distance)
        total_moving_secs += _safe_int(a.moving_time)
        total_elev_m += _safe_float(a.total_elevation_gain)

        if a.average_heartrate is not None:
            hr_sum += float(a.average_heartrate)
            hr_count += 1
        cad = getattr(a, "average_cadence", None)
        if cad is not None:
            cad_sum += float(cad)
            cad_count += 1
        ss = a.suffer_score
        if ss is not None:
            suffer_total += int(ss)
            suffer_count += 1

    count = len(activities)
    dist_km = total_dist_m / 1000
    total_dist_km = round(dist_km, 1)
    total_hours = round(total_moving_secs / 3600, 1)

    out: dict = {
        "count": count,
        "total_distance_km": total_dist_km,
        "total_time_hours": total_hours,
        "total_elevation_m": int(round(total_elev_m, 0)),
    }

    if atype in _PACE_TYPES and dist_km > 0 and total_moving_secs > 0:
        out["avg_pace_min_per_km"] = round(total_moving_secs / 60 / dist_km, 2)
    elif atype in _SPEED_TYPES and total_moving_secs > 0:
        out["avg_speed_kmh"] = round(dist_km / (total_moving_secs / 3600), 1)

    if hr_count > 0:
        out["avg_heartrate"] = round(hr_sum / hr_count, 1)
    if cad_count > 0:
        out["avg_cadence"] = round(cad_sum / cad_count, 1)
    if suffer_count > 0:
        out["avg_suffer_score"] = round(suffer_total / suffer_count, 1)
        out["total_suffer_score"] = suffer_total

    return out


def _format_dates(after: datetime, before: datetime) -> dict:
    """Return date range as ISO strings."""
    return {"after": after.strftime("%Y-%m-%d"), "before": before.strftime("%Y-%m-%d")}


def _overall_totals(activities: list) -> dict:
    """Cross-type sum of distance/time/elevation + avg_distance_km."""
    total_dist_m = 0.0
    total_moving_secs = 0
    total_elev_m = 0.0
    for a in activities:
        total_dist_m += _safe_float(a.distance)
        total_moving_secs += _safe_int(a.moving_time)
        total_elev_m += _safe_float(a.total_elevation_gain)
    count = len(activities)
    total_dist_km = round(total_dist_m / 1000, 1)
    total_hours = round(total_moving_secs / 3600, 1)
    avg_dist = round(total_dist_km / count, 1) if count > 0 else 0.0
    return {
        "total_distance_km": total_dist_km,
        "total_time_hours": total_hours,
        "total_elevation_m": int(round(total_elev_m, 0)),
        "avg_distance_km": avg_dist,
    }


def _compare_label(period_str: str) -> str:
    """Map a period string to its natural comparison label."""
    _MAP = {
        "this-week": "last-week",
        "last-week": "2-weeks-ago",
        "this-month": "last-month",
        "last-month": "2-months-ago",
        "this-year": "last-year",
        "ytd": "last-year-ytd",
    }
    label = _MAP.get(period_str)
    if label:
        return label
    m = _PERIOD_RE.match(period_str)
    if m:
        return f"prior {period_str}"
    return period_str


def _handle_summary(client, args: dict) -> str:
    """Aggregate activities for a period into per-type summaries."""
    try:
        period = args.get("period", "this-month")
        parsed = _parse_period(period)
        if isinstance(parsed, str):
            return json.dumps({"error": parsed})
        after, before = parsed

        activity_type = args.get("type")
        if activity_type is not None and activity_type not in _VALID_ACTIVITY_TYPES:
            return json.dumps({"error": f"invalid activity type: {activity_type!r}"})

        activities = list(client.get_activities(after=after, before=before, limit=200))

        if activity_type:
            activities = [a for a in activities if _type_str(a.type) == activity_type]

        if not activities:
            return json.dumps({"period": period, "count": 0, "by_type": {}})

        groups: dict[str, list] = defaultdict(list)
        for a in activities:
            groups[_type_str(a.type)].append(a)

        result: dict = {"period": period, "count": len(activities), "by_type": {}}
        for gtype, acts in groups.items():
            result["by_type"][gtype] = _summarise_group(gtype, acts)

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"Strava API error: {e}"})


def _handle_compare(client, args: dict) -> str:
    """Compare activity summaries across two periods."""
    try:
        period_a_str = args.get("period_a", "this-month")
        parsed_a = _parse_period(period_a_str)
        if isinstance(parsed_a, str):
            return json.dumps({"error": parsed_a})

        period_b_str = args.get("period_b")
        if period_b_str:
            parsed_b = _parse_period(period_b_str)
            if isinstance(parsed_b, str):
                return json.dumps({"error": parsed_b})
        else:
            parsed_b = _default_comparison_period(period_a_str, parsed_a)
            if isinstance(parsed_b, str):
                return json.dumps({"error": parsed_b})
            period_b_str = "auto"

        activity_type = args.get("type")
        if activity_type is not None and activity_type not in _VALID_ACTIVITY_TYPES:
            return json.dumps({"error": f"invalid activity type: {activity_type!r}"})

        acts_a = list(client.get_activities(after=parsed_a[0], before=parsed_a[1], limit=200))
        acts_b = list(client.get_activities(after=parsed_b[0], before=parsed_b[1], limit=200))

        if activity_type:
            acts_a = [a for a in acts_a if _type_str(a.type) == activity_type]
            acts_b = [a for a in acts_b if _type_str(a.type) == activity_type]

        groups_a: dict[str, list] = defaultdict(list)
        for a in acts_a:
            groups_a[_type_str(a.type)].append(a)

        groups_b: dict[str, list] = defaultdict(list)
        for a in acts_b:
            groups_b[_type_str(a.type)].append(a)

        all_types = sorted(set(groups_a) | set(groups_b))

        by_type = {}
        for atype in all_types:
            entry: dict = {}
            if atype in groups_a:
                entry["period_a"] = _summarise_group(atype, groups_a[atype])
            if atype in groups_b:
                entry["period_b"] = _summarise_group(atype, groups_b[atype])
            if "period_a" in entry and "period_b" in entry:
                entry["delta"] = _compute_delta(entry["period_a"], entry["period_b"])
            by_type[atype] = entry

        return json.dumps({
            "period_a": period_a_str,
            "period_b": period_b_str,
            "count_a": len(acts_a),
            "count_b": len(acts_b),
            "by_type": by_type,
        })

    except Exception as e:
        return json.dumps({"error": f"Strava API error: {e}"})


def _handle_analysis(client, args: dict) -> str:
    """Analyse training for a period with automatic trend comparison."""
    try:
        period_str = args.get("period", "this-week")
        parsed = _parse_period(period_str)
        if isinstance(parsed, str):
            return json.dumps({"error": parsed})
        after, before = parsed

        activity_type = args.get("type")
        if activity_type is not None and activity_type not in _VALID_ACTIVITY_TYPES:
            return json.dumps({"error": f"invalid activity type: {activity_type!r}"})

        compare_period_str = args.get("compare_period")
        if compare_period_str:
            parsed_b = _parse_period(compare_period_str)
            if isinstance(parsed_b, str):
                return json.dumps({"error": parsed_b})
            compare_label = compare_period_str
        else:
            parsed_b = _default_comparison_period(period_str, parsed)
            if isinstance(parsed_b, str):
                return json.dumps({"error": parsed_b})
            compare_label = _compare_label(period_str)
        after_b, before_b = parsed_b

        activities = list(client.get_activities(after=after, before=before, limit=200))
        compare_activities = list(client.get_activities(after=after_b, before=before_b, limit=200))

        if activity_type:
            activities = [a for a in activities if _type_str(a.type) == activity_type]
            compare_activities = [a for a in compare_activities if _type_str(a.type) == activity_type]

        # Group by type
        groups: dict[str, list] = defaultdict(list)
        for a in activities:
            groups[_type_str(a.type)].append(a)

        compare_groups: dict[str, list] = defaultdict(list)
        for a in compare_activities:
            compare_groups[_type_str(a.type)].append(a)

        by_type = {}
        for gtype, acts in groups.items():
            by_type[gtype] = _summarise_group(gtype, acts)

        compare_by_type = {}
        for gtype, acts in compare_groups.items():
            compare_by_type[gtype] = _summarise_group(gtype, acts)

        overall = _overall_totals(activities)
        compare_overall = _overall_totals(compare_activities)

        overall_delta = _compute_delta(overall, compare_overall)

        # Per-type deltas only for types present in both periods
        by_type_delta = {}
        for gtype in set(by_type) & set(compare_by_type):
            by_type_delta[gtype] = _compute_delta(by_type[gtype], compare_by_type[gtype])

        return json.dumps({
            "period": period_str,
            "period_dates": _format_dates(after, before),
            "count": len(activities),
            "overall": overall,
            "by_type": by_type,
            "compare_period": compare_label,
            "compare_period_dates": _format_dates(after_b, before_b),
            "compare_count": len(compare_activities),
            "compare_overall": compare_overall,
            "compare_by_type": compare_by_type,
            "overall_delta": overall_delta,
            "by_type_delta": by_type_delta,
        })

    except Exception as e:
        return json.dumps({"error": f"Strava API error: {e}"})


def _route_to_dict(route, include_segments: bool = False) -> dict:
    """Extract route fields into a plain dict."""
    distance_m = _safe_float(getattr(route, "distance", None))
    elev_m = _safe_float(getattr(route, "elevation_gain", None))
    est_time = getattr(route, "estimated_moving_time", None)

    result: dict = {
        "id": route.id,
        "name": getattr(route, "name", None),
        "type": _ROUTE_TYPE.get(getattr(route, "type", None)),
        "sub_type": _ROUTE_SUB_TYPE.get(getattr(route, "sub_type", None)),
        "distance_km": round(distance_m / 1000, 1),
        "elevation_gain_m": round(elev_m, 0),
        "estimated_time_min": round(_safe_int(est_time) / 60, 0) if est_time is not None else None,
        "starred": getattr(route, "starred", None),
        "private": getattr(route, "private", None),
    }

    desc = getattr(route, "description", None)
    if desc:
        result["description"] = desc

    if include_segments and hasattr(route, "segments") and route.segments:
        result["segments"] = [_segment_to_dict(s) for s in route.segments]

    return result


def _segment_to_dict(seg) -> dict:
    """Extract segment fields into a plain dict."""
    distance_m = _safe_float(getattr(seg, "distance", None))
    result: dict = {
        "id": seg.id,
        "name": getattr(seg, "name", None),
        "activity_type": _type_str(getattr(seg, "activity_type", None)),
        "distance_km": round(distance_m / 1000, 1),
        "average_grade": _safe_float(getattr(seg, "average_grade", None)),
        "maximum_grade": _safe_float(getattr(seg, "maximum_grade", None)),
        "elevation_high_m": _safe_float(getattr(seg, "elevation_high", None)),
        "elevation_low_m": _safe_float(getattr(seg, "elevation_low", None)),
        "climb_category": _safe_int(getattr(seg, "climb_category", None)),
    }

    for attr in ("city", "state", "country"):
        val = getattr(seg, attr, None)
        if val:
            result[attr] = val

    pr = getattr(seg, "athlete_pr_effort", None)
    if pr is not None:
        pr_time = getattr(pr, "elapsed_time", None) or getattr(pr, "moving_time", None)
        if pr_time is not None:
            result["pr"] = {"time_sec": _safe_int(pr_time)}

    return result


def _handle_routes(client, args: dict) -> str:
    """Handle route listing, detail, and starred segments."""
    try:
        action = args.get("action", "list")
        if action not in _VALID_ROUTE_ACTIONS:
            return json.dumps({"error": f"invalid action: {action!r} — use 'list', 'detail', or 'starred'"})

        limit = max(1, min(50, int(args.get("limit", 20))))

        if action == "detail":
            route_id = args.get("id")
            if route_id is None:
                return json.dumps({"error": "id is required for action 'detail'"})
            route = client.get_route(int(route_id))
            return json.dumps(_route_to_dict(route, include_segments=True))

        if action == "starred":
            segs = list(client.get_starred_segments(limit=limit))
            return json.dumps({"segments": [_segment_to_dict(s) for s in segs]})

        # list
        routes = list(client.get_routes(limit=limit))
        return json.dumps({"routes": [_route_to_dict(r) for r in routes]})

    except Exception as e:
        return json.dumps({"error": f"Strava API error: {e}"})
