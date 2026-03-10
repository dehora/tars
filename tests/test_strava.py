import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import strava


def _mock_activity(**overrides):
    """Create a mock activity with sensible defaults."""
    defaults = {
        "id": 123,
        "name": "Morning Run",
        "type": "Run",
        "distance": 5000.0,
        "moving_time": 1500,
        "elapsed_time": 1600,
        "total_elevation_gain": 50.0,
        "average_heartrate": 145.0,
        "max_heartrate": 165.0,
        "start_date_local": datetime(2026, 3, 1, 8, 0, 0),
        "suffer_score": 42,
        "sport_type": "Run",
    }
    defaults.update(overrides)
    a = mock.Mock()
    for k, v in defaults.items():
        setattr(a, k, v)
    return a


def _mock_athlete(**overrides):
    defaults = {
        "id": 1,
        "firstname": "Test",
        "lastname": "User",
        "city": "Dublin",
        "state": "Leinster",
        "country": "Ireland",
        "weight": 75.0,
        "premium": True,
        "bikes": [],
        "shoes": [],
    }
    defaults.update(overrides)
    a = mock.Mock()
    for k, v in defaults.items():
        setattr(a, k, v)
    return a


def _mock_totals(count=10, distance=100000.0, moving_time=36000, elevation_gain=500.0):
    t = mock.Mock()
    t.count = count
    t.distance = distance
    t.moving_time = moving_time
    t.elevation_gain = elevation_gain
    return t


class ParsePeriodTests(unittest.TestCase):
    def test_days(self):
        result = strava._parse_period("7d")
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertAlmostEqual(
            (before - after).total_seconds(), 7 * 86400, delta=5
        )

    def test_weeks(self):
        result = strava._parse_period("2w")
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertAlmostEqual(
            (before - after).total_seconds(), 14 * 86400, delta=5
        )

    def test_months(self):
        result = strava._parse_period("3m")
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertGreater(before, after)

    def test_years(self):
        result = strava._parse_period("1y")
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertGreater(before, after)

    def test_this_week(self):
        result = strava._parse_period("this-week")
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertEqual(after.weekday(), 0)
        self.assertLessEqual(after, before)

    def test_last_week(self):
        result = strava._parse_period("last-week")
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertEqual(after.weekday(), 0)
        self.assertEqual((before - after).days, 7)

    def test_this_month(self):
        result = strava._parse_period("this-month")
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertEqual(after.day, 1)

    def test_last_month(self):
        result = strava._parse_period("last-month")
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertEqual(after.day, 1)
        self.assertEqual(before.day, 1)

    def test_this_year_and_ytd(self):
        for p in ("this-year", "ytd"):
            result = strava._parse_period(p)
            self.assertIsInstance(result, tuple)
            after, _ = result
            self.assertEqual(after.month, 1)
            self.assertEqual(after.day, 1)

    def test_invalid_period(self):
        result = strava._parse_period("garbage")
        self.assertIsInstance(result, str)
        self.assertIn("invalid period", result)

    def test_invalid_empty(self):
        result = strava._parse_period("")
        self.assertIsInstance(result, str)


class ActivitiesToolTests(unittest.TestCase):
    def setUp(self):
        self.client = mock.Mock()

    @mock.patch.object(strava, "_get_client")
    def test_recent_activities(self, mock_get):
        mock_get.return_value = self.client
        activities = [_mock_activity(id=1), _mock_activity(id=2)]
        self.client.get_activities.return_value = activities

        result = json.loads(strava._run_strava_tool("strava_activities", {}))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], 1)
        self.client.get_activities.assert_called_once_with(limit=20)

    @mock.patch.object(strava, "_get_client")
    def test_activities_with_period(self, mock_get):
        mock_get.return_value = self.client
        self.client.get_activities.return_value = []

        result = json.loads(strava._run_strava_tool("strava_activities", {"period": "7d"}))
        self.assertIsInstance(result, list)
        call_kwargs = self.client.get_activities.call_args[1]
        self.assertIn("after", call_kwargs)
        self.assertIn("before", call_kwargs)

    @mock.patch.object(strava, "_get_client")
    def test_activities_type_filter(self, mock_get):
        mock_get.return_value = self.client
        run = _mock_activity(id=1, type="Run")
        ride = _mock_activity(id=2, type="Ride")
        self.client.get_activities.return_value = [run, ride]

        result = json.loads(strava._run_strava_tool("strava_activities", {"type": "Run"}))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "Run")

    @mock.patch.object(strava, "_get_client")
    def test_activities_invalid_type(self, mock_get):
        mock_get.return_value = self.client
        result = json.loads(strava._run_strava_tool("strava_activities", {"type": "InvalidSport"}))
        self.assertIn("error", result)

    @mock.patch.object(strava, "_get_client")
    def test_activities_invalid_sort(self, mock_get):
        mock_get.return_value = self.client
        result = json.loads(strava._run_strava_tool("strava_activities", {"sort": "random"}))
        self.assertIn("error", result)

    @mock.patch.object(strava, "_get_client")
    def test_activity_by_id(self, mock_get):
        mock_get.return_value = self.client
        detail = _mock_activity(id=99)
        lap = mock.Mock()
        lap.name = "Lap 1"
        lap.distance = 1000.0
        lap.moving_time = 300
        lap.average_heartrate = 140.0
        lap.max_heartrate = 160.0
        detail.laps = [lap]
        split = mock.Mock()
        split.split = 1
        split.distance = 1000.0
        split.moving_time = 300
        split.average_heartrate = 140.0
        split.elevation_difference = 5.0
        detail.splits_metric = [split]
        self.client.get_activity.return_value = detail

        result = json.loads(strava._run_strava_tool("strava_activities", {"id": 99}))
        self.assertEqual(result["id"], 99)
        self.assertIn("laps", result)
        self.assertIn("splits", result)
        self.client.get_activity.assert_called_once_with(99)

    @mock.patch.object(strava, "_get_client")
    def test_limit_clamped(self, mock_get):
        mock_get.return_value = self.client
        self.client.get_activities.return_value = []

        strava._run_strava_tool("strava_activities", {"limit": 999})
        call_kwargs = self.client.get_activities.call_args[1]
        self.assertEqual(call_kwargs["limit"], 100)

        strava._run_strava_tool("strava_activities", {"limit": -5})
        call_kwargs = self.client.get_activities.call_args[1]
        self.assertEqual(call_kwargs["limit"], 1)

    @mock.patch.object(strava, "_get_client")
    def test_activities_oldest_sort(self, mock_get):
        mock_get.return_value = self.client
        a1 = _mock_activity(id=1)
        a2 = _mock_activity(id=2)
        self.client.get_activities.return_value = [a1, a2]

        result = json.loads(strava._run_strava_tool("strava_activities", {"sort": "oldest"}))
        self.assertEqual(result[0]["id"], 2)
        self.assertEqual(result[1]["id"], 1)

    @mock.patch.object(strava, "_get_client")
    def test_invalid_period_returns_error(self, mock_get):
        mock_get.return_value = self.client
        result = json.loads(strava._run_strava_tool("strava_activities", {"period": "garbage"}))
        self.assertIn("error", result)

    @mock.patch.object(strava, "_get_client")
    def test_pace_for_run(self, mock_get):
        mock_get.return_value = self.client
        run = _mock_activity(type="Run", distance=5000.0, moving_time=1500)
        self.client.get_activities.return_value = [run]

        result = json.loads(strava._run_strava_tool("strava_activities", {}))
        self.assertIn("pace_min_per_km", result[0])
        self.assertAlmostEqual(result[0]["pace_min_per_km"], 5.0, places=1)

    @mock.patch.object(strava, "_get_client")
    def test_speed_for_ride(self, mock_get):
        mock_get.return_value = self.client
        ride = _mock_activity(type="Ride", distance=20000.0, moving_time=3600)
        self.client.get_activities.return_value = [ride]

        result = json.loads(strava._run_strava_tool("strava_activities", {}))
        self.assertIn("speed_kmh", result[0])
        self.assertAlmostEqual(result[0]["speed_kmh"], 20.0, places=0)


class UserToolTests(unittest.TestCase):
    @mock.patch.object(strava, "_get_client")
    def test_profile_and_stats(self, mock_get):
        client = mock.Mock()
        mock_get.return_value = client
        client.get_athlete.return_value = _mock_athlete()
        stats = mock.Mock()
        stats.ytd_run_totals = _mock_totals(count=50, distance=200000.0)
        stats.ytd_ride_totals = _mock_totals(count=0)
        stats.ytd_swim_totals = _mock_totals(count=0)
        stats.all_run_totals = _mock_totals(count=200, distance=800000.0)
        stats.all_ride_totals = _mock_totals(count=0)
        stats.all_swim_totals = _mock_totals(count=0)
        stats.recent_run_totals = _mock_totals(count=5)
        stats.recent_ride_totals = _mock_totals(count=0)
        stats.recent_swim_totals = _mock_totals(count=0)
        client.get_athlete_stats.return_value = stats

        result = json.loads(strava._run_strava_tool("strava_user", {}))
        self.assertIn("profile", result)
        self.assertEqual(result["profile"]["name"], "Test User")
        self.assertIn("stats", result)
        self.assertEqual(result["stats"]["ytd_run"]["count"], 50)

    @mock.patch.object(strava, "_get_client")
    def test_zones(self, mock_get):
        client = mock.Mock()
        mock_get.return_value = client
        zones = mock.Mock()
        hr = mock.Mock()
        hr.zones = [
            mock.Mock(min=0, max=120),
            mock.Mock(min=120, max=150),
            mock.Mock(min=150, max=170),
            mock.Mock(min=170, max=185),
            mock.Mock(min=185, max=-1),
        ]
        zones.heart_rate = hr
        client.get_athlete_zones.return_value = zones

        result = json.loads(strava._run_strava_tool("strava_user", {"include": ["zones"]}))
        self.assertIn("zones", result)
        self.assertEqual(len(result["zones"]["heart_rate"]), 5)

    @mock.patch.object(strava, "_get_client")
    def test_gear(self, mock_get):
        client = mock.Mock()
        mock_get.return_value = client
        bike = mock.Mock(name="Road Bike", distance=5000000.0)
        bike.name = "Road Bike"
        shoe = mock.Mock(name="Running Shoes", distance=500000.0)
        shoe.name = "Running Shoes"
        athlete = _mock_athlete(bikes=[bike], shoes=[shoe])
        client.get_athlete.return_value = athlete

        result = json.loads(strava._run_strava_tool("strava_user", {"include": ["gear"]}))
        self.assertIn("gear", result)
        self.assertEqual(len(result["gear"]), 2)

    @mock.patch.object(strava, "_get_client")
    def test_invalid_section(self, mock_get):
        client = mock.Mock()
        mock_get.return_value = client
        result = json.loads(strava._run_strava_tool("strava_user", {"include": ["invalid"]}))
        self.assertIn("error", result)


class TokenTests(unittest.TestCase):
    def test_missing_tokens_returns_error(self):
        with mock.patch.object(strava, "_load_tokens", return_value=None):
            result = json.loads(strava._run_strava_tool("strava_activities", {}))
        self.assertIn("error", result)
        self.assertIn("strava-auth", result["error"])

    def test_token_refresh_on_expiry(self):
        expired_tokens = {
            "access_token": "old",
            "refresh_token": "refresh123",
            "expires_at": int(time.time()) - 100,
        }
        fresh_response = {
            "access_token": "new",
            "refresh_token": "refresh456",
            "expires_at": int(time.time()) + 3600,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            token_file = Path(tmpdir) / "strava_tokens.json"
            token_file.write_text(json.dumps(expired_tokens))

            mock_stravalib = mock.Mock()
            mock_client_instance = mock.Mock()
            mock_stravalib.Client.return_value = mock_client_instance
            mock_client_instance.refresh_access_token.return_value = fresh_response

            with mock.patch.object(strava, "_token_path", return_value=token_file), \
                 mock.patch.object(strava, "_load_tokens", return_value=expired_tokens), \
                 mock.patch.dict(os.environ, {
                     "TARS_STRAVA_CLIENT_ID": "12345",
                     "TARS_STRAVA_CLIENT_SECRET": "secret",
                 }), \
                 mock.patch.dict(sys.modules, {"stravalib": mock_stravalib}):
                client = strava._get_client()

            mock_client_instance.refresh_access_token.assert_called_once_with(
                client_id=12345,
                client_secret="secret",
                refresh_token="refresh123",
            )
            saved = json.loads(token_file.read_text())
            self.assertEqual(saved["access_token"], "new")
            self.assertEqual(saved["refresh_token"], "refresh456")

    def test_save_tokens_sets_permissions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            token_file = Path(tmpdir) / "strava_tokens.json"
            with mock.patch.object(strava, "_token_path", return_value=token_file):
                strava._save_tokens({"access_token": "test", "refresh_token": "r", "expires_at": 0})
            self.assertEqual(oct(token_file.stat().st_mode & 0o777), oct(0o600))


class UnknownToolTests(unittest.TestCase):
    @mock.patch.object(strava, "_get_client")
    def test_unknown_tool_returns_error(self, mock_get):
        mock_get.return_value = mock.Mock()
        result = json.loads(strava._run_strava_tool("strava_unknown", {}))
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
