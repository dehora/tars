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
        "average_cadence": None,
        "workout_type": None,
        "average_watts": None,
        "weighted_average_watts": None,
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
        detail.calories = None
        detail.description = None
        detail.perceived_exertion = None
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
    def test_zones_tuple_format(self, mock_get):
        """Stravalib v2 returns zones as tuples (min, max) instead of objects."""
        client = mock.Mock()
        mock_get.return_value = client
        zones = mock.Mock()
        hr = mock.Mock()
        hr.zones = [(0, 120), (120, 150), (150, 170), (170, 185), (185, -1)]
        zones.heart_rate = hr
        client.get_athlete_zones.return_value = zones

        result = json.loads(strava._run_strava_tool("strava_user", {"include": ["zones"]}))
        self.assertIn("zones", result)
        hr_zones = result["zones"]["heart_rate"]
        self.assertEqual(len(hr_zones), 5)
        self.assertEqual(hr_zones[0], {"min": 0, "max": 120})
        self.assertEqual(hr_zones[4], {"min": 185, "max": -1})

    @mock.patch.object(strava, "_get_client")
    def test_zones_root_model_wrapped(self, mock_get):
        """hr has no .zones attr, falls back to hr.root."""
        client = mock.Mock()
        mock_get.return_value = client
        zones = mock.Mock()
        hr = mock.Mock(spec=[])
        hr.root = [(0, 120), (120, 150)]
        zones.heart_rate = hr
        client.get_athlete_zones.return_value = zones

        result = json.loads(strava._run_strava_tool("strava_user", {"include": ["zones"]}))
        hr_zones = result["zones"]["heart_rate"]
        self.assertEqual(len(hr_zones), 2)
        self.assertEqual(hr_zones[0], {"min": 0, "max": 120})

    @mock.patch.object(strava, "_get_client")
    def test_zones_stravalib_v2_model(self, mock_get):
        """Stravalib v2: hr.zones is a ZoneRanges RootModel wrapping ZoneRange objects."""
        client = mock.Mock()
        mock_get.return_value = client
        zones = mock.Mock()
        hr = mock.Mock()
        zone_ranges = mock.Mock(spec=["root"])
        zone_ranges.root = [
            mock.Mock(spec=["min", "max"], min=0, max=120),
            mock.Mock(spec=["min", "max"], min=120, max=150),
            mock.Mock(spec=["min", "max"], min=150, max=170),
        ]
        hr.zones = zone_ranges
        zones.heart_rate = hr
        client.get_athlete_zones.return_value = zones

        result = json.loads(strava._run_strava_tool("strava_user", {"include": ["zones"]}))
        hr_zones = result["zones"]["heart_rate"]
        self.assertEqual(len(hr_zones), 3)
        self.assertEqual(hr_zones[0], {"min": 0, "max": 120})
        self.assertEqual(hr_zones[2], {"min": 150, "max": 170})

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


class TypeStrTests(unittest.TestCase):
    def test_plain_string(self):
        self.assertEqual(strava._type_str("Run"), "Run")

    def test_none(self):
        self.assertEqual(strava._type_str(None), "")

    def test_root_model(self):
        rm = mock.Mock()
        rm.root = "WeightTraining"
        self.assertEqual(strava._type_str(rm), "WeightTraining")

    def test_type_filter_with_root_model(self):
        """Type filter works when stravalib wraps types in RootModel."""
        rm_run = mock.Mock()
        rm_run.root = "Run"
        rm_ride = mock.Mock()
        rm_ride.root = "Ride"
        run = _mock_activity(id=1, type=rm_run)
        ride = _mock_activity(id=2, type=rm_ride)
        result_run = strava._activity_to_dict(run)
        result_ride = strava._activity_to_dict(ride)
        self.assertEqual(result_run["type"], "Run")
        self.assertEqual(result_ride["type"], "Ride")


class ActivityEnrichmentTests(unittest.TestCase):
    def test_cadence_included(self):
        a = _mock_activity(average_cadence=172.5)
        result = strava._activity_to_dict(a)
        self.assertEqual(result["average_cadence"], 172.5)

    def test_cadence_omitted_when_none(self):
        a = _mock_activity(average_cadence=None)
        result = strava._activity_to_dict(a)
        self.assertNotIn("average_cadence", result)

    def test_workout_type_long_run(self):
        a = _mock_activity(workout_type=2)
        result = strava._activity_to_dict(a)
        self.assertEqual(result["workout_type"], "long_run")

    def test_workout_type_race(self):
        a = _mock_activity(workout_type=1)
        result = strava._activity_to_dict(a)
        self.assertEqual(result["workout_type"], "race")

    def test_workout_type_default_omitted(self):
        a = _mock_activity(workout_type=0)
        result = strava._activity_to_dict(a)
        self.assertNotIn("workout_type", result)

    def test_workout_type_none_omitted(self):
        a = _mock_activity(workout_type=None)
        result = strava._activity_to_dict(a)
        self.assertNotIn("workout_type", result)

    def test_watts_included(self):
        a = _mock_activity(type="Ride", average_watts=200.0, weighted_average_watts=210)
        result = strava._activity_to_dict(a)
        self.assertEqual(result["average_watts"], 200.0)
        self.assertEqual(result["weighted_average_watts"], 210)

    @mock.patch.object(strava, "_get_client")
    def test_detail_includes_calories_description_rpe(self, mock_get):
        client = mock.Mock()
        mock_get.return_value = client
        detail = _mock_activity(id=99)
        detail.calories = 450.0
        detail.description = "Easy morning run"
        detail.perceived_exertion = 6
        detail.laps = []
        detail.splits_metric = []
        client.get_activity.return_value = detail

        result = json.loads(strava._run_strava_tool("strava_activities", {"id": 99}))
        self.assertEqual(result["calories"], 450)
        self.assertEqual(result["description"], "Easy morning run")
        self.assertEqual(result["perceived_exertion"], 6)

    @mock.patch.object(strava, "_get_client")
    def test_detail_omits_none_calories(self, mock_get):
        client = mock.Mock()
        mock_get.return_value = client
        detail = _mock_activity(id=99)
        detail.calories = None
        detail.description = ""
        detail.perceived_exertion = None
        detail.laps = []
        detail.splits_metric = []
        client.get_activity.return_value = detail

        result = json.loads(strava._run_strava_tool("strava_activities", {"id": 99}))
        self.assertNotIn("calories", result)
        self.assertNotIn("description", result)
        self.assertNotIn("perceived_exertion", result)


class SummaryToolTests(unittest.TestCase):
    def setUp(self):
        self.client = mock.Mock()

    @mock.patch.object(strava, "_get_client")
    def test_summary_basic(self, mock_get):
        mock_get.return_value = self.client
        acts = [
            _mock_activity(type="Run", distance=5000.0, moving_time=1500,
                           total_elevation_gain=50.0, average_heartrate=145.0,
                           average_cadence=172.0, suffer_score=40),
            _mock_activity(type="Run", distance=8000.0, moving_time=2400,
                           total_elevation_gain=80.0, average_heartrate=150.0,
                           average_cadence=174.0, suffer_score=60),
        ]
        self.client.get_activities.return_value = acts

        result = json.loads(strava._run_strava_tool("strava_summary", {"period": "this-month"}))
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["period"], "this-month")
        self.assertIn("Run", result["by_type"])
        run = result["by_type"]["Run"]
        self.assertEqual(run["count"], 2)
        self.assertAlmostEqual(run["total_distance_km"], 13.0, places=0)
        self.assertIn("avg_pace_min_per_km", run)
        self.assertIn("avg_heartrate", run)
        self.assertIn("avg_cadence", run)
        self.assertIn("avg_suffer_score", run)
        self.assertEqual(run["total_suffer_score"], 100)

    @mock.patch.object(strava, "_get_client")
    def test_summary_groups_by_type(self, mock_get):
        mock_get.return_value = self.client
        acts = [
            _mock_activity(type="Run", distance=5000.0, moving_time=1500),
            _mock_activity(type="WeightTraining", distance=0.0, moving_time=1800,
                           total_elevation_gain=0.0, average_heartrate=95.0),
        ]
        self.client.get_activities.return_value = acts

        result = json.loads(strava._run_strava_tool("strava_summary", {"period": "this-month"}))
        self.assertEqual(result["count"], 2)
        self.assertIn("Run", result["by_type"])
        self.assertIn("WeightTraining", result["by_type"])

    @mock.patch.object(strava, "_get_client")
    def test_summary_type_filter(self, mock_get):
        mock_get.return_value = self.client
        run = _mock_activity(type="Run", distance=5000.0, moving_time=1500)
        ride = _mock_activity(type="Ride", distance=20000.0, moving_time=3600)
        self.client.get_activities.return_value = [run, ride]

        result = json.loads(strava._run_strava_tool("strava_summary", {"period": "this-month", "type": "Run"}))
        self.assertEqual(result["count"], 1)
        self.assertIn("Run", result["by_type"])
        self.assertNotIn("Ride", result["by_type"])

    @mock.patch.object(strava, "_get_client")
    def test_summary_empty_period(self, mock_get):
        mock_get.return_value = self.client
        self.client.get_activities.return_value = []

        result = json.loads(strava._run_strava_tool("strava_summary", {}))
        self.assertEqual(result["count"], 0)

    @mock.patch.object(strava, "_get_client")
    def test_summary_invalid_period(self, mock_get):
        mock_get.return_value = self.client
        result = json.loads(strava._run_strava_tool("strava_summary", {"period": "garbage"}))
        self.assertIn("error", result)

    @mock.patch.object(strava, "_get_client")
    def test_summary_invalid_type(self, mock_get):
        mock_get.return_value = self.client
        result = json.loads(strava._run_strava_tool("strava_summary", {"type": "InvalidSport"}))
        self.assertIn("error", result)

    @mock.patch.object(strava, "_get_client")
    def test_summary_ride_speed(self, mock_get):
        mock_get.return_value = self.client
        ride = _mock_activity(type="Ride", distance=20000.0, moving_time=3600,
                              total_elevation_gain=100.0)
        self.client.get_activities.return_value = [ride]

        result = json.loads(strava._run_strava_tool("strava_summary", {"period": "this-month"}))
        self.assertIn("avg_speed_kmh", result["by_type"]["Ride"])
        self.assertAlmostEqual(result["by_type"]["Ride"]["avg_speed_kmh"], 20.0, places=0)


class FormatSummaryTests(unittest.TestCase):
    def test_format_summary_basic(self):
        from tars.format import format_strava_summary
        raw = json.dumps({
            "period": "this-month",
            "count": 2,
            "by_type": {
                "Run": {
                    "count": 2,
                    "total_distance_km": 13.0,
                    "total_time_hours": 1.1,
                    "total_elevation_m": 130,
                    "avg_pace_min_per_km": 5.0,
                    "avg_heartrate": 147.5,
                    "avg_cadence": 173.0,
                    "avg_suffer_score": 50.0,
                    "total_suffer_score": 100,
                }
            },
        })
        out = format_strava_summary(raw)
        self.assertIn("this-month", out)
        self.assertIn("Run x2", out)
        self.assertIn("13.0km", out)
        self.assertIn("hr:147", out)
        self.assertIn("cad:173", out)
        self.assertIn("effort:50", out)

    def test_format_summary_empty(self):
        from tars.format import format_strava_summary
        raw = json.dumps({"period": "this-month", "count": 0, "by_type": {}})
        out = format_strava_summary(raw)
        self.assertIn("no activities", out)


class UnknownToolTests(unittest.TestCase):
    @mock.patch.object(strava, "_get_client")
    def test_unknown_tool_returns_error(self, mock_get):
        mock_get.return_value = mock.Mock()
        result = json.loads(strava._run_strava_tool("strava_unknown", {}))
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
