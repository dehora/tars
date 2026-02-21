import json
import unittest

from tars.format import (
    format_memory_recall,
    format_todoist_action,
    format_todoist_list,
    format_tool_result,
    format_web_read,
    format_weather_forecast,
    format_weather_now,
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


if __name__ == "__main__":
    unittest.main()
