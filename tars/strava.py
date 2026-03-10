"""Strava integration — OAuth token lifecycle and tool dispatch."""

import json
import os
import re
import time
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

_TOKEN_FILE = "strava_tokens.json"


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
    distance_m = float(a.distance) if a.distance is not None else 0.0
    distance_km = round(distance_m / 1000, 2)
    moving_secs = int(a.moving_time) if a.moving_time is not None else 0
    elapsed_secs = int(a.elapsed_time) if a.elapsed_time is not None else 0
    moving_min = round(moving_secs / 60, 1)
    elapsed_min = round(elapsed_secs / 60, 1)
    elev_m = round(float(a.total_elevation_gain), 1) if a.total_elevation_gain is not None else None

    result = {
        "id": a.id,
        "name": a.name,
        "type": _type_str(a.type) if a.type else None,
        "distance_km": distance_km,
        "moving_time_min": moving_min,
        "elapsed_time_min": elapsed_min,
        "elevation_gain_m": elev_m,
        "average_heartrate": a.average_heartrate,
        "max_heartrate": a.max_heartrate,
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
    if activity_type in ("Run", "Walk", "Hike"):
        if distance_km > 0 and moving_secs > 0:
            result["pace_min_per_km"] = round(moving_secs / 60 / distance_km, 2)
    elif activity_type in ("Ride", "VirtualRide", "EBikeRide"):
        if moving_secs > 0:
            result["speed_kmh"] = round(distance_km / (moving_secs / 3600), 1)

    return result


def _lap_to_dict(lap) -> dict:
    distance_m = float(lap.distance) if lap.distance is not None else 0.0
    moving_secs = int(lap.moving_time) if lap.moving_time is not None else 0
    return {
        "name": lap.name,
        "distance_km": round(distance_m / 1000, 2),
        "moving_time_min": round(moving_secs / 60, 1),
        "average_heartrate": lap.average_heartrate,
        "max_heartrate": lap.max_heartrate,
    }


def _split_to_dict(split) -> dict:
    distance_m = float(split.distance) if split.distance is not None else 0.0
    moving_secs = int(split.moving_time) if split.moving_time is not None else 0
    return {
        "split": split.split,
        "distance_km": round(distance_m / 1000, 2),
        "moving_time_min": round(moving_secs / 60, 1),
        "average_heartrate": split.average_heartrate if hasattr(split, "average_heartrate") else None,
        "elevation_difference_m": split.elevation_difference,
    }


def _totals_to_dict(totals) -> dict:
    if totals is None:
        return {}
    distance_m = float(totals.distance) if totals.distance is not None else 0.0
    moving_secs = int(totals.moving_time) if totals.moving_time is not None else 0
    elev_m = float(totals.elevation_gain) if totals.elevation_gain is not None else 0.0
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

        if "profile" in include:
            athlete = client.get_athlete()
            result["profile"] = {
                "name": f"{athlete.firstname} {athlete.lastname}".strip(),
                "city": athlete.city,
                "state": athlete.state,
                "country": athlete.country,
                "weight_kg": athlete.weight,
                "premium": athlete.premium,
            }
            athlete_id = athlete.id
        else:
            athlete_id = None

        if "stats" in include:
            if athlete_id is None:
                athlete = client.get_athlete()
                athlete_id = athlete.id
            stats = client.get_athlete_stats(athlete_id)
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
            if athlete_id is None:
                athlete = client.get_athlete()
                athlete_id = athlete.id
            else:
                if "profile" not in include:
                    athlete = client.get_athlete()
            gear = []
            if hasattr(athlete, "bikes") and athlete.bikes:
                for b in athlete.bikes:
                    dist_m = float(b.distance) if b.distance is not None else 0.0
                    gear.append({
                        "type": "bike",
                        "name": b.name,
                        "distance_km": round(dist_m / 1000, 0),
                    })
            if hasattr(athlete, "shoes") and athlete.shoes:
                for s in athlete.shoes:
                    dist_m = float(s.distance) if s.distance is not None else 0.0
                    gear.append({
                        "type": "shoe",
                        "name": s.name,
                        "distance_km": round(dist_m / 1000, 0),
                    })
            result["gear"] = gear

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"Strava API error: {e}"})


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
        total_dist_m += float(a.distance) if a.distance is not None else 0.0
        total_moving_secs += int(a.moving_time) if a.moving_time is not None else 0
        total_elev_m += float(a.total_elevation_gain) if a.total_elevation_gain is not None else 0.0

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
    total_dist_km = round(total_dist_m / 1000, 1)
    total_hours = round(total_moving_secs / 3600, 1)

    out: dict = {
        "count": count,
        "total_distance_km": total_dist_km,
        "total_time_hours": total_hours,
        "total_elevation_m": int(round(total_elev_m, 0)),
    }

    if atype in ("Run", "Walk", "Hike") and total_dist_km > 0 and total_moving_secs > 0:
        out["avg_pace_min_per_km"] = round(total_moving_secs / 60 / total_dist_km, 2)
    elif atype in ("Ride", "VirtualRide", "EBikeRide") and total_moving_secs > 0:
        out["avg_speed_kmh"] = round(total_dist_km / (total_moving_secs / 3600), 1)

    if hr_count > 0:
        out["avg_heartrate"] = round(hr_sum / hr_count, 1)
    if cad_count > 0:
        out["avg_cadence"] = round(cad_sum / cad_count, 1)
    if suffer_count > 0:
        out["avg_suffer_score"] = round(suffer_total / suffer_count, 1)
        out["total_suffer_score"] = suffer_total

    return out


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
            return json.dumps({"summary": {}, "period": period, "count": 0})

        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)
        for a in activities:
            groups[_type_str(a.type)].append(a)

        result: dict = {"period": period, "count": len(activities), "by_type": {}}
        for gtype, acts in groups.items():
            result["by_type"][gtype] = _summarise_group(gtype, acts)

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"Strava API error: {e}"})
