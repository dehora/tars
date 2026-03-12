import json
import unittest

from tars.format import (
    format_memory_recall,
    format_strava_activities,
    format_strava_zones,
    format_todoist_action,
    format_todoist_list,
    format_tool_result,
    format_web_read,
    format_weather_forecast,
    format_weather_now,
    sparkline,
)


class TodoistListTests(unittest.TestCase):
    def test_formats_tasks(self) -> None:
        raw = json.dumps({
            "results": [
                {
                    "content": "Buy milk",
                    "priority": 4,
                    "due": {"string": "today"},
                    "duration": None,
                },
                {
                    "content": "Read book",
                    "priority": 1,
                    "due": {"string": "tomorrow"},
                    "duration": {"amount": 30, "unit": "minute"},
                },
            ],
            "nextCursor": None,
        })
        out = format_todoist_list(raw)
        self.assertIn("1. [p4] Buy milk (due: today)", out)
        self.assertIn("2. [p1] Read book (due: tomorrow, 30m)", out)

    def test_empty_results(self) -> None:
        raw = json.dumps({"results": [], "nextCursor": None})
        self.assertEqual(format_todoist_list(raw), "no tasks")

    def test_error(self) -> None:
        raw = json.dumps({"error": "td not found"})
        self.assertEqual(format_todoist_list(raw), "td not found")

    def test_passthrough_non_json(self) -> None:
        self.assertEqual(format_todoist_list("not json"), "not json")


class TodoistActionTests(unittest.TestCase):
    def test_ok(self) -> None:
        self.assertEqual(format_todoist_action('{"ok": true}'), "done")

    def test_error(self) -> None:
        self.assertEqual(format_todoist_action('{"error": "fail"}'), "fail")


class WeatherNowTests(unittest.TestCase):
    def test_formats_current(self) -> None:
        raw = json.dumps({
            "current": {
                "temperature_c": 10.0,
                "precipitation_mm": 0.0,
                "conditions": "Clear sky",
                "wind_speed_kmh": 15.0,
            },
            "next_hours": [
                {"time": "2026-02-20T14:00", "temp_c": 10, "precip_prob_pct": 0, "precip_mm": 0},
                {"time": "2026-02-20T15:00", "temp_c": 11, "precip_prob_pct": 50, "precip_mm": 0.1},
            ],
            "location": {"lat": 53.0, "lon": -6.0},
        })
        out = format_weather_now(raw)
        self.assertIn("10.0\u00b0C", out)
        self.assertIn("clear sky", out)
        self.assertIn("wind 15.0 km/h", out)
        self.assertIn("next:", out)
        self.assertIn("50%", out)

    def test_error(self) -> None:
        raw = json.dumps({"error": "no location"})
        self.assertEqual(format_weather_now(raw), "no location")


class WeatherForecastTests(unittest.TestCase):
    def test_formats_hourly(self) -> None:
        hours = [
            {"time": f"2026-02-20T{h:02d}:00", "temp_c": 8 + h % 5, "precip_prob_pct": h * 4, "precip_mm": 0}
            for h in range(24)
        ]
        raw = json.dumps({"hourly": hours, "location": {"lat": 53.0, "lon": -6.0}})
        out = format_weather_forecast(raw)
        self.assertIn("00:00", out)
        # Should have multiple lines
        self.assertGreater(len(out.splitlines()), 1)

    def test_empty(self) -> None:
        raw = json.dumps({"hourly": []})
        self.assertEqual(format_weather_forecast(raw), "no forecast data")


class MemoryRecallTests(unittest.TestCase):
    def test_formats_sections(self) -> None:
        raw = json.dumps({"semantic": "- dog: Perry\n- city: Dublin", "procedural": "- be kind"})
        out = format_memory_recall(raw)
        self.assertIn("[semantic]", out)
        self.assertIn("- dog: Perry", out)
        self.assertIn("[procedural]", out)
        self.assertIn("- be kind", out)


class WebReadTests(unittest.TestCase):
    def test_formats_content(self) -> None:
        raw = json.dumps({"url": "https://example.com", "content": "Hello world", "truncated": False})
        result = format_web_read(raw)
        self.assertIn("[https://example.com]", result)
        self.assertIn("Hello world", result)
        self.assertNotIn("truncated", result)

    def test_truncated(self) -> None:
        raw = json.dumps({"url": "https://example.com", "content": "text", "truncated": True})
        result = format_web_read(raw)
        self.assertIn("(content truncated)", result)

    def test_error(self) -> None:
        raw = json.dumps({"error": "fetch failed"})
        self.assertEqual(format_web_read(raw), "fetch failed")


class FormatToolResultTests(unittest.TestCase):
    def test_known_tool(self) -> None:
        raw = json.dumps({"results": [], "nextCursor": None})
        self.assertEqual(format_tool_result("todoist_today", raw), "no tasks")

    def test_unknown_tool_passthrough(self) -> None:
        self.assertEqual(format_tool_result("unknown_tool", "raw data"), "raw data")


class SparklineTests(unittest.TestCase):
    def test_basic(self) -> None:
        result = sparkline([1, 2, 3, 4, 5])
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], "▁")
        self.assertEqual(result[-1], "█")

    def test_equal_values(self) -> None:
        result = sparkline([5, 5, 5])
        self.assertEqual(result, "▄▄▄")

    def test_single_value(self) -> None:
        self.assertEqual(sparkline([42]), "")

    def test_empty(self) -> None:
        self.assertEqual(sparkline([]), "")

    def test_nones_filtered(self) -> None:
        result = sparkline([1, None, 3])
        self.assertEqual(len(result), 2)

    def test_two_values(self) -> None:
        result = sparkline([0, 10])
        self.assertEqual(result, "▁█")

    def test_invert(self) -> None:
        result = sparkline([0, 10], invert=True)
        self.assertEqual(result, "█▁")

    def test_invert_pace(self) -> None:
        # Lower pace = faster = taller bar
        result = sparkline([6.0, 5.5, 5.0], invert=True)
        self.assertEqual(result[0], "▁")
        self.assertEqual(result[-1], "█")


class StravaActivitiesSparklineTests(unittest.TestCase):
    def test_activities_with_pace_sparkline(self) -> None:
        activities = [
            {"name": f"Run {i}", "type": "Run", "distance_km": 5, "moving_time_min": 25,
             "pace_min_per_km": pace, "average_heartrate": hr, "start_date": f"2026-03-0{i}"}
            for i, (pace, hr) in enumerate([(5.5, 140), (5.3, 145), (5.0, 150)], 1)
        ]
        result = format_strava_activities(json.dumps(activities))
        self.assertIn("pace:", result)
        self.assertIn("hr:", result)
        # Sparkline should be on the last line
        last_line = result.strip().split("\n")[-1]
        self.assertIn("▁", last_line)

    def test_single_activity_no_sparkline(self) -> None:
        activities = [
            {"name": "Run", "type": "Run", "distance_km": 5, "moving_time_min": 25,
             "pace_min_per_km": 5.5, "start_date": "2026-03-01"}
        ]
        result = format_strava_activities(json.dumps(activities))
        self.assertNotIn("▁", result)


class SplitPaceNormalizationTests(unittest.TestCase):
    def test_partial_split_same_pace_as_full(self) -> None:
        """A 0.2km split in 1min should yield the same pace bar as a 1km split in 5min."""
        from tars.format import _format_single_activity
        data = {
            "name": "Test Run", "type": "Run", "distance_km": 1.2, "moving_time_min": 6,
            "pace_min_per_km": 5.0, "start_date": "2026-03-01",
            "splits": [
                {"split": 1, "distance_km": 1.0, "moving_time_min": 5.0},
                {"split": 2, "distance_km": 0.2, "moving_time_min": 1.0},
            ],
        }
        result = _format_single_activity(data)
        # Both splits have pace 5.0 min/km → equal bars (mid-block since sparkline returns same char for equal)
        self.assertIn("pace:", result)
        # The sparkline chars should be identical since pace is the same
        pace_line = [l for l in result.splitlines() if "pace:" in l][0]
        spark_chars = pace_line.split("pace:")[1].strip().split()[0]
        self.assertEqual(spark_chars[0], spark_chars[1])


class ActivityListSparklineGapTests(unittest.TestCase):
    def test_elevation_sparkline_present(self) -> None:
        activities = [
            {"name": f"Run {i}", "type": "Run", "distance_km": 5, "moving_time_min": 25,
             "pace_min_per_km": 5.0, "average_heartrate": 140, "elevation_gain_m": elev,
             "start_date": f"2026-03-0{i}"}
            for i, elev in enumerate([50, 100, 75], 1)
        ]
        result = format_strava_activities(json.dumps(activities))
        self.assertIn("elev:", result)

    def test_mixed_type_no_pace_sparkline(self) -> None:
        activities = [
            {"name": "Run", "type": "Run", "distance_km": 5, "moving_time_min": 25,
             "pace_min_per_km": 5.0, "start_date": "2026-03-01"},
            {"name": "Ride", "type": "Ride", "distance_km": 20, "moving_time_min": 60,
             "speed_kmh": 20.0, "start_date": "2026-03-02"},
        ]
        result = format_strava_activities(json.dumps(activities))
        self.assertNotIn("pace:", result)


class ActivityIDTests(unittest.TestCase):
    def test_activity_id_in_output(self) -> None:
        activities = [
            {"id": 12345, "name": "Run", "type": "Run", "distance_km": 5,
             "moving_time_min": 25, "start_date": "2026-03-01"}
        ]
        result = format_strava_activities(json.dumps(activities))
        self.assertIn("(id:12345)", result)


class TodoistIDTests(unittest.TestCase):
    def test_todoist_id_in_output(self) -> None:
        raw = json.dumps({
            "results": [
                {"id": "abc123", "content": "Buy milk", "priority": 1, "due": None}
            ]
        })
        result = format_todoist_list(raw)
        self.assertIn("(id:abc123)", result)


class StravaZonesFormatTests(unittest.TestCase):
    def test_renders_chart(self) -> None:
        raw = json.dumps({
            "period": "4w",
            "classification": "Threshold-Heavy",
            "zone_pct": {"low": 72, "mod": 22, "high": 6},
            "total_hours": 6.2,
            "activities_analysed": 10,
            "activities_skipped": {"no_hr": 4, "too_short": 0, "over_cap": 0},
            "zone_boundaries": {"low_max": 145, "mod_max": 170},
            "per_activity": [],
        })
        out = format_strava_zones(raw)
        self.assertIn("Training Zones (4w)", out)
        self.assertIn("10 activities", out)
        self.assertIn("6.2h", out)
        self.assertIn("Low", out)
        self.assertIn("Mod", out)
        self.assertIn("High", out)
        self.assertIn("Threshold-Heavy", out)
        self.assertIn("\u2588", out)
        self.assertIn("<145", out)
        self.assertIn("145-169", out)
        self.assertIn(">=170", out)

    def test_error_passthrough(self) -> None:
        raw = json.dumps({"error": "HR zones not configured"})
        self.assertEqual(format_strava_zones(raw), "HR zones not configured")

    def test_non_json_passthrough(self) -> None:
        self.assertEqual(format_strava_zones("not json"), "not json")

    def test_tool_result_dispatch(self) -> None:
        raw = json.dumps({
            "period": "4w",
            "classification": "Polarised",
            "zone_pct": {"low": 80, "mod": 5, "high": 15},
            "total_hours": 8.0,
            "activities_analysed": 12,
            "activities_skipped": {"no_hr": 0, "too_short": 0, "over_cap": 0},
            "zone_boundaries": {"low_max": 145, "mod_max": 170},
            "per_activity": [],
        })
        out = format_tool_result("strava_zones", raw)
        self.assertIn("Polarised", out)


if __name__ == "__main__":
    unittest.main()
