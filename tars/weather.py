import json
import os
import urllib.request


WMO_WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def _fetch_weather(lat: float, lon: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,precipitation,weather_code,wind_speed_10m"
        f"&hourly=temperature_2m,precipitation_probability,precipitation"
        f"&forecast_days=1&timezone=auto"
    )
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def _resolve_coords(args: dict) -> tuple[float | None, float | None]:
    lat = args["lat"] if "lat" in args and args["lat"] is not None else None
    lon = args["lon"] if "lon" in args and args["lon"] is not None else None

    if lat is None:
        env_lat = os.environ.get("DEFAULT_LAT")
        lat = float(env_lat) if env_lat not in (None, "") else None
    if lon is None:
        env_lon = os.environ.get("DEFAULT_LON")
        lon = float(env_lon) if env_lon not in (None, "") else None

    return lat, lon


def _run_weather_tool(name: str, args: dict) -> str:
    lat, lon = _resolve_coords(args)
    if lat is None or lon is None:
        return json.dumps({"error": "No location provided and DEFAULT_LAT/DEFAULT_LON not set in .env"})
    try:
        data = _fetch_weather(lat, lon)
    except Exception as e:
        return json.dumps({"error": f"Weather API request failed: {e}"})

    if name == "weather_now":
        current = data.get("current", {})
        hourly = data.get("hourly", {})
        # Next 6 hours of precipitation data
        precip_probs = hourly.get("precipitation_probability", [])[:6]
        precip_amounts = hourly.get("precipitation", [])[:6]
        temps = hourly.get("temperature_2m", [])[:6]
        times = hourly.get("time", [])[:6]
        series_len = min(
            len(times),
            len(temps),
            len(precip_probs),
            len(precip_amounts),
        )
        weather_code = current.get("weather_code", 0)
        return json.dumps({
            "current": {
                "temperature_c": current.get("temperature_2m"),
                "precipitation_mm": current.get("precipitation"),
                "conditions": WMO_WEATHER_CODES.get(weather_code, f"Code {weather_code}"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
            },
            "next_hours": [
                {
                    "time": times[i] if i < len(times) else None,
                    "temp_c": temps[i] if i < len(temps) else None,
                    "precip_prob_pct": precip_probs[i] if i < len(precip_probs) else None,
                    "precip_mm": precip_amounts[i] if i < len(precip_amounts) else None,
                }
                for i in range(series_len)
            ],
            "location": {"lat": lat, "lon": lon},
        })

    # weather_forecast — full day hourly
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precip_probs = hourly.get("precipitation_probability", [])
    precip_amounts = hourly.get("precipitation", [])
    series_len = min(len(times), len(temps), len(precip_probs), len(precip_amounts))
    return json.dumps({
        "hourly": [
            {
                "time": times[i],
                "temp_c": temps[i] if i < len(temps) else None,
                "precip_prob_pct": precip_probs[i] if i < len(precip_probs) else None,
                "precip_mm": precip_amounts[i] if i < len(precip_amounts) else None,
            }
            for i in range(series_len)
        ],
        "location": {"lat": lat, "lon": lon},
    })
