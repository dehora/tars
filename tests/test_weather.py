import json
import os
import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import weather


class WeatherToolTests(unittest.TestCase):
    def test_run_weather_tool_accepts_zero_coords(self) -> None:
        payload = {
            "current": {
                "temperature_2m": 12.0,
                "precipitation": 0.0,
                "weather_code": 0,
                "wind_speed_10m": 5.0,
            },
            "hourly": {
                "time": ["2025-01-01T00:00"],
                "temperature_2m": [12.0],
                "precipitation_probability": [5],
                "precipitation": [0.0],
            },
        }
        with mock.patch.object(weather, "_fetch_weather", return_value=payload):
            result = json.loads(weather._run_weather_tool("weather_now", {"lat": 0, "lon": -1}))
        self.assertNotIn("error", result)
        self.assertEqual(result["location"]["lat"], 0)
        self.assertEqual(result["location"]["lon"], -1)

    def test_missing_coords_returns_error(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            result = json.loads(weather._run_weather_tool("weather_now", {}))
        self.assertIn("error", result)

    def test_api_failure_returns_error(self) -> None:
        with mock.patch.object(weather, "_fetch_weather", side_effect=Exception("boom")):
            result = json.loads(weather._run_weather_tool("weather_now", {"lat": 10, "lon": 20}))
        self.assertIn("Weather API request failed", result.get("error", ""))

    def test_forecast_uses_minimum_hourly_length(self) -> None:
        payload = {
            "hourly": {
                "time": ["t1", "t2", "t3"],
                "temperature_2m": [1.0, 2.0],
                "precipitation_probability": [10],
                "precipitation": [0.1, 0.2],
            },
        }
        with mock.patch.object(weather, "_fetch_weather", return_value=payload):
            result = json.loads(weather._run_weather_tool("weather_forecast", {"lat": 10, "lon": 20}))
        self.assertEqual(len(result["hourly"]), 1)


if __name__ == "__main__":
    unittest.main()
