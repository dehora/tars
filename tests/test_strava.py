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
    def test_today(self):
        result = strava._parse_period("today")
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertEqual(after.hour, 0)
        self.assertEqual(after.minute, 0)
        self.assertGreater(before, after)

    def test_yesterday(self):
        result = strava._parse_period("yesterday")
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertAlmostEqual(
            (before - after).total_seconds(), 86400, delta=5
        )
        self.assertEqual(before.hour, 0)

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

    def test_single_date(self):
        result = strava._parse_period("2025-10-28")
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertEqual(after.year, 2025)
        self.assertEqual(after.month, 10)
        self.assertEqual(after.day, 28)
        self.assertGreater(before, after)

    def test_date_range(self):
        result = strava._parse_period("2025-10-28_2026-03-15")
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertEqual(after.year, 2025)
        self.assertEqual(after.month, 10)
        self.assertEqual(after.day, 28)
        self.assertEqual(before.year, 2026)
        self.assertEqual(before.month, 3)
        self.assertEqual(before.day, 15)
        self.assertEqual(before.hour, 23)
        self.assertEqual(before.minute, 59)

    def test_date_range_start_after_end(self):
        result = strava._parse_period("2026-03-15_2025-10-28")
        self.assertIsInstance(result, str)
        self.assertIn("invalid date range", result)

    def test_malformed_date_range(self):
        result = strava._parse_period("2025-13-01_2025-14-01")
        self.assertIsInstance(result, str)
        self.assertIn("invalid date range", result)

    def test_malformed_single_date_falls_through(self):
        result = strava._parse_period("2025-13-01")
        self.assertIsInstance(result, str)
        self.assertIn("invalid period", result)


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
        self.client.get_activities.return_value = iter([run, ride])

        result = json.loads(strava._run_strava_tool("strava_activities", {"type": "Run"}))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "Run")

    @mock.patch.object(strava, "_get_client")
    def test_activities_type_filter_root_model(self, mock_get):
        """Type filter works when stravalib wraps types in RootModel on the list path."""
        mock_get.return_value = self.client
        rm_run = mock.Mock()
        rm_run.root = "Run"
        rm_ride = mock.Mock()
        rm_ride.root = "Ride"
        run = _mock_activity(id=1, type=rm_run)
        ride = _mock_activity(id=2, type=rm_ride)
        self.client.get_activities.return_value = iter([run, ride])

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
    def test_type_filter_overfetches(self, mock_get):
        """When type filter is set, fetch more from API so limit applies after filtering."""
        mock_get.return_value = self.client
        runs = [_mock_activity(id=i, type="Run") for i in range(3)]
        rides = [_mock_activity(id=i + 10, type="Ride") for i in range(7)]
        self.client.get_activities.return_value = rides + runs

        result = json.loads(strava._run_strava_tool("strava_activities", {"type": "Run", "limit": 2}))
        self.assertEqual(len(result), 2)
        self.assertTrue(all(a["type"] == "Run" for a in result))
        fetch_limit = self.client.get_activities.call_args[1]["limit"]
        self.assertGreater(fetch_limit, 2)

    @mock.patch.object(strava, "_get_client")
    def test_type_filter_respects_limit(self, mock_get):
        """Result count is capped to the requested limit after type filtering."""
        mock_get.return_value = self.client
        runs = [_mock_activity(id=i, type="Run") for i in range(10)]
        self.client.get_activities.return_value = runs

        result = json.loads(strava._run_strava_tool("strava_activities", {"type": "Run", "limit": 3}))
        self.assertEqual(len(result), 3)

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
    def test_stats_and_gear_single_athlete_fetch(self, mock_get):
        """include=['stats', 'gear'] fetches athlete only once."""
        client = mock.Mock()
        mock_get.return_value = client
        bike = mock.Mock(distance=1000000.0)
        bike.name = "Bike"
        athlete = _mock_athlete(bikes=[bike])
        client.get_athlete.return_value = athlete
        stats = mock.Mock()
        stats.ytd_run_totals = _mock_totals(count=5)
        stats.ytd_ride_totals = _mock_totals(count=0)
        stats.ytd_swim_totals = _mock_totals(count=0)
        stats.all_run_totals = _mock_totals(count=0)
        stats.all_ride_totals = _mock_totals(count=0)
        stats.all_swim_totals = _mock_totals(count=0)
        stats.recent_run_totals = _mock_totals(count=0)
        stats.recent_ride_totals = _mock_totals(count=0)
        stats.recent_swim_totals = _mock_totals(count=0)
        client.get_athlete_stats.return_value = stats

        result = json.loads(strava._run_strava_tool("strava_user", {"include": ["stats", "gear"]}))
        self.assertIn("stats", result)
        self.assertIn("gear", result)
        self.assertEqual(len(result["gear"]), 1)
        client.get_athlete.assert_called_once()

    @mock.patch.object(strava, "_get_client")
    def test_include_as_string(self, mock_get):
        """include param as a plain string is coerced to list."""
        client = mock.Mock()
        mock_get.return_value = client
        zones = mock.Mock()
        hr = mock.Mock()
        hr.zones = [mock.Mock(min=0, max=120)]
        zones.heart_rate = hr
        client.get_athlete_zones.return_value = zones

        result = json.loads(strava._run_strava_tool("strava_user", {"include": "zones"}))
        self.assertIn("zones", result)
        self.assertEqual(len(result["zones"]["heart_rate"]), 1)

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
        self.assertEqual(result["by_type"], {})

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


def _mock_route(**overrides):
    """Create a mock route with sensible defaults."""
    defaults = {
        "id": 1001,
        "name": "Morning Loop",
        "type": 2,  # Run
        "sub_type": 4,  # trail
        "distance": 10200.0,
        "elevation_gain": 250.0,
        "estimated_moving_time": 3900,
        "starred": True,
        "private": False,
        "description": None,
        "segments": [],
    }
    defaults.update(overrides)
    r = mock.Mock()
    for k, v in defaults.items():
        setattr(r, k, v)
    return r


def _mock_segment(**overrides):
    """Create a mock segment with sensible defaults."""
    defaults = {
        "id": 5001,
        "name": "Hill Climb",
        "activity_type": "Run",
        "distance": 1500.0,
        "average_grade": 6.5,
        "maximum_grade": 12.0,
        "elevation_high": 320.0,
        "elevation_low": 220.0,
        "climb_category": 3,
        "city": "Dublin",
        "state": "Leinster",
        "country": "Ireland",
        "athlete_pr_effort": None,
    }
    defaults.update(overrides)
    s = mock.Mock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


class DefaultComparisonPeriodTests(unittest.TestCase):
    def test_today(self):
        parsed_a = strava._parse_period("today")
        result = strava._default_comparison_period("today", parsed_a)
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertAlmostEqual(
            (before - after).total_seconds(), 86400, delta=5
        )

    def test_yesterday(self):
        parsed_a = strava._parse_period("yesterday")
        result = strava._default_comparison_period("yesterday", parsed_a)
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertAlmostEqual(
            (before - after).total_seconds(), 86400, delta=5
        )

    def test_this_week(self):
        parsed_a = strava._parse_period("this-week")
        result = strava._default_comparison_period("this-week", parsed_a)
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertEqual(after.weekday(), 0)
        self.assertEqual((before - after).days, 7)

    def test_this_month(self):
        parsed_a = strava._parse_period("this-month")
        result = strava._default_comparison_period("this-month", parsed_a)
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertEqual(after.day, 1)
        self.assertEqual(before.day, 1)

    def test_last_week(self):
        parsed_a = strava._parse_period("last-week")
        result = strava._default_comparison_period("last-week", parsed_a)
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertEqual((before - after).days, 7)

    def test_last_month(self):
        parsed_a = strava._parse_period("last-month")
        result = strava._default_comparison_period("last-month", parsed_a)
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertEqual(after.day, 1)

    def test_this_year(self):
        parsed_a = strava._parse_period("this-year")
        result = strava._default_comparison_period("this-year", parsed_a)
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertEqual(after.year, parsed_a[0].year - 1)

    def test_ytd(self):
        parsed_a = strava._parse_period("ytd")
        result = strava._default_comparison_period("ytd", parsed_a)
        self.assertIsInstance(result, tuple)

    def test_numeric_7d(self):
        parsed_a = strava._parse_period("7d")
        result = strava._default_comparison_period("7d", parsed_a)
        self.assertIsInstance(result, tuple)
        after, before = result
        span = before - after
        self.assertAlmostEqual(span.total_seconds(), 7 * 86400, delta=5)

    def test_numeric_3m(self):
        parsed_a = strava._parse_period("3m")
        result = strava._default_comparison_period("3m", parsed_a)
        self.assertIsInstance(result, tuple)
        after, before = result
        self.assertLess(after, before)

    def test_invalid_returns_error(self):
        result = strava._default_comparison_period("garbage", (datetime.now(timezone.utc), datetime.now(timezone.utc)))
        self.assertIsInstance(result, str)
        self.assertIn("cannot auto-derive", result)


class ComputeDeltaTests(unittest.TestCase):
    def test_basic_delta(self):
        a = {"count": 10, "total_distance_km": 50.0}
        b = {"count": 8, "total_distance_km": 40.0}
        delta = strava._compute_delta(a, b)
        self.assertEqual(delta["count"]["change"], 2)
        self.assertAlmostEqual(delta["count"]["pct"], 25.0)
        self.assertAlmostEqual(delta["total_distance_km"]["change"], 10.0)
        self.assertAlmostEqual(delta["total_distance_km"]["pct"], 25.0)

    def test_zero_denominator(self):
        a = {"count": 5}
        b = {"count": 0}
        delta = strava._compute_delta(a, b)
        self.assertEqual(delta["count"]["change"], 5)
        self.assertNotIn("pct", delta["count"])

    def test_negative_change(self):
        a = {"total_distance_km": 30.0}
        b = {"total_distance_km": 50.0}
        delta = strava._compute_delta(a, b)
        self.assertAlmostEqual(delta["total_distance_km"]["change"], -20.0)
        self.assertAlmostEqual(delta["total_distance_km"]["pct"], -40.0)

    def test_non_numeric_skipped(self):
        a = {"count": 5, "label": "Run"}
        b = {"count": 3, "label": "Run"}
        delta = strava._compute_delta(a, b)
        self.assertIn("count", delta)
        self.assertNotIn("label", delta)

    def test_missing_key_in_b(self):
        a = {"count": 5, "extra": 10}
        b = {"count": 3}
        delta = strava._compute_delta(a, b)
        self.assertIn("count", delta)
        self.assertNotIn("extra", delta)


class CompareToolTests(unittest.TestCase):
    def setUp(self):
        self.client = mock.Mock()

    @mock.patch.object(strava, "_get_client")
    def test_basic_comparison(self, mock_get):
        mock_get.return_value = self.client
        acts_a = [_mock_activity(type="Run", distance=5000.0, moving_time=1500)]
        acts_b = [_mock_activity(type="Run", distance=4000.0, moving_time=1300)]
        self.client.get_activities.side_effect = [acts_a, acts_b]

        result = json.loads(strava._run_strava_tool("strava_compare", {"period_a": "this-month"}))
        self.assertEqual(result["period_a"], "this-month")
        self.assertEqual(result["period_b"], "auto")
        self.assertEqual(result["count_a"], 1)
        self.assertEqual(result["count_b"], 1)
        self.assertIn("Run", result["by_type"])
        self.assertIn("delta", result["by_type"]["Run"])

    @mock.patch.object(strava, "_get_client")
    def test_explicit_period_b(self, mock_get):
        mock_get.return_value = self.client
        self.client.get_activities.side_effect = [[], []]

        result = json.loads(strava._run_strava_tool("strava_compare", {
            "period_a": "this-month", "period_b": "last-month"
        }))
        self.assertEqual(result["period_b"], "last-month")

    @mock.patch.object(strava, "_get_client")
    def test_type_filter(self, mock_get):
        mock_get.return_value = self.client
        run = _mock_activity(type="Run", distance=5000.0, moving_time=1500)
        ride = _mock_activity(type="Ride", distance=20000.0, moving_time=3600)
        self.client.get_activities.side_effect = [[run, ride], [run]]

        result = json.loads(strava._run_strava_tool("strava_compare", {
            "period_a": "this-month", "type": "Run"
        }))
        self.assertEqual(result["count_a"], 1)
        self.assertIn("Run", result["by_type"])
        self.assertNotIn("Ride", result["by_type"])

    @mock.patch.object(strava, "_get_client")
    def test_invalid_type(self, mock_get):
        mock_get.return_value = self.client
        result = json.loads(strava._run_strava_tool("strava_compare", {
            "period_a": "this-month", "type": "InvalidSport"
        }))
        self.assertIn("error", result)

    @mock.patch.object(strava, "_get_client")
    def test_invalid_period(self, mock_get):
        mock_get.return_value = self.client
        result = json.loads(strava._run_strava_tool("strava_compare", {"period_a": "garbage"}))
        self.assertIn("error", result)

    @mock.patch.object(strava, "_get_client")
    def test_invalid_period_b(self, mock_get):
        mock_get.return_value = self.client
        result = json.loads(strava._run_strava_tool("strava_compare", {
            "period_a": "this-month", "period_b": "garbage"
        }))
        self.assertIn("error", result)

    @mock.patch.object(strava, "_get_client")
    def test_empty_periods(self, mock_get):
        mock_get.return_value = self.client
        self.client.get_activities.side_effect = [[], []]

        result = json.loads(strava._run_strava_tool("strava_compare", {"period_a": "this-month"}))
        self.assertEqual(result["count_a"], 0)
        self.assertEqual(result["count_b"], 0)
        self.assertEqual(result["by_type"], {})

    @mock.patch.object(strava, "_get_client")
    def test_one_sided_comparison(self, mock_get):
        mock_get.return_value = self.client
        acts_a = [_mock_activity(type="Run", distance=5000.0, moving_time=1500)]
        self.client.get_activities.side_effect = [acts_a, []]

        result = json.loads(strava._run_strava_tool("strava_compare", {"period_a": "this-month"}))
        self.assertEqual(result["count_a"], 1)
        self.assertEqual(result["count_b"], 0)
        run_entry = result["by_type"]["Run"]
        self.assertIn("period_a", run_entry)
        self.assertNotIn("period_b", run_entry)
        self.assertNotIn("delta", run_entry)

    @mock.patch.object(strava, "_get_client")
    def test_delta_keys_present(self, mock_get):
        mock_get.return_value = self.client
        acts_a = [_mock_activity(type="Run", distance=5000.0, moving_time=1500,
                                 average_heartrate=150.0)]
        acts_b = [_mock_activity(type="Run", distance=4000.0, moving_time=1300,
                                 average_heartrate=145.0)]
        self.client.get_activities.side_effect = [acts_a, acts_b]

        result = json.loads(strava._run_strava_tool("strava_compare", {"period_a": "this-month"}))
        delta = result["by_type"]["Run"]["delta"]
        self.assertIn("count", delta)
        self.assertIn("total_distance_km", delta)
        self.assertIn("total_time_hours", delta)
        self.assertIn("avg_heartrate", delta)


class RoutesToolTests(unittest.TestCase):
    def setUp(self):
        self.client = mock.Mock()

    @mock.patch.object(strava, "_get_client")
    def test_list_routes(self, mock_get):
        mock_get.return_value = self.client
        self.client.get_routes.return_value = [_mock_route(), _mock_route(id=1002, name="Evening Loop")]

        result = json.loads(strava._run_strava_tool("strava_routes", {"action": "list"}))
        self.assertIn("routes", result)
        self.assertEqual(len(result["routes"]), 2)
        self.assertEqual(result["routes"][0]["id"], 1001)
        self.assertEqual(result["routes"][0]["type"], "Run")
        self.assertEqual(result["routes"][0]["sub_type"], "trail")

    @mock.patch.object(strava, "_get_client")
    def test_route_detail(self, mock_get):
        mock_get.return_value = self.client
        seg = _mock_segment()
        route = _mock_route(segments=[seg])
        self.client.get_route.return_value = route

        result = json.loads(strava._run_strava_tool("strava_routes", {"action": "detail", "id": 1001}))
        self.assertEqual(result["id"], 1001)
        self.assertIn("segments", result)
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["name"], "Hill Climb")

    @mock.patch.object(strava, "_get_client")
    def test_starred_segments(self, mock_get):
        mock_get.return_value = self.client
        seg = _mock_segment()
        self.client.get_starred_segments.return_value = [seg]

        result = json.loads(strava._run_strava_tool("strava_routes", {"action": "starred"}))
        self.assertIn("segments", result)
        self.assertEqual(len(result["segments"]), 1)

    @mock.patch.object(strava, "_get_client")
    def test_limit_clamped(self, mock_get):
        mock_get.return_value = self.client
        self.client.get_routes.return_value = []

        strava._run_strava_tool("strava_routes", {"action": "list", "limit": 999})
        self.client.get_routes.assert_called_once_with(limit=50)

        self.client.get_routes.reset_mock()
        strava._run_strava_tool("strava_routes", {"action": "list", "limit": -5})
        self.client.get_routes.assert_called_once_with(limit=1)

    @mock.patch.object(strava, "_get_client")
    def test_missing_id_error(self, mock_get):
        mock_get.return_value = self.client
        result = json.loads(strava._run_strava_tool("strava_routes", {"action": "detail"}))
        self.assertIn("error", result)
        self.assertIn("id is required", result["error"])

    @mock.patch.object(strava, "_get_client")
    def test_invalid_action(self, mock_get):
        mock_get.return_value = self.client
        result = json.loads(strava._run_strava_tool("strava_routes", {"action": "delete"}))
        self.assertIn("error", result)

    @mock.patch.object(strava, "_get_client")
    def test_type_sub_type_mapping(self, mock_get):
        mock_get.return_value = self.client
        route = _mock_route(type=1, sub_type=2)  # Ride / MTB
        self.client.get_routes.return_value = [route]

        result = json.loads(strava._run_strava_tool("strava_routes", {"action": "list"}))
        self.assertEqual(result["routes"][0]["type"], "Ride")
        self.assertEqual(result["routes"][0]["sub_type"], "MTB")

    @mock.patch.object(strava, "_get_client")
    def test_segment_pr_handling(self, mock_get):
        mock_get.return_value = self.client
        pr = mock.Mock()
        pr.elapsed_time = 360
        pr.moving_time = 350
        seg = _mock_segment(athlete_pr_effort=pr)
        self.client.get_starred_segments.return_value = [seg]

        result = json.loads(strava._run_strava_tool("strava_routes", {"action": "starred"}))
        self.assertIn("pr", result["segments"][0])
        self.assertEqual(result["segments"][0]["pr"]["time_sec"], 360)


class FormatCompareTests(unittest.TestCase):
    def test_basic_output(self):
        from tars.format import format_strava_compare
        raw = json.dumps({
            "period_a": "this-month", "period_b": "last-month",
            "count_a": 10, "count_b": 8,
            "by_type": {
                "Run": {
                    "period_a": {"count": 10, "total_distance_km": 50.0, "total_time_hours": 5.0,
                                 "total_elevation_m": 500, "avg_pace_min_per_km": 5.0, "avg_heartrate": 150.0},
                    "period_b": {"count": 8, "total_distance_km": 40.0, "total_time_hours": 4.0,
                                 "total_elevation_m": 400, "avg_pace_min_per_km": 5.25, "avg_heartrate": 148.0},
                    "delta": {
                        "count": {"change": 2, "pct": 25.0},
                        "total_distance_km": {"change": 10.0, "pct": 25.0},
                        "total_time_hours": {"change": 1.0, "pct": 25.0},
                        "total_elevation_m": {"change": 100, "pct": 25.0},
                        "avg_pace_min_per_km": {"change": -0.25, "pct": -4.8},
                        "avg_heartrate": {"change": 2.0, "pct": 1.4},
                    },
                }
            },
        })
        out = format_strava_compare(raw)
        self.assertIn("this-month vs last-month", out)
        self.assertIn("Run: 10 vs 8", out)
        self.assertIn("distance:", out)
        self.assertIn("faster", out)

    def test_empty_periods(self):
        from tars.format import format_strava_compare
        raw = json.dumps({
            "period_a": "this-month", "period_b": "last-month",
            "count_a": 0, "count_b": 0,
            "by_type": {},
        })
        out = format_strava_compare(raw)
        self.assertIn("no activities", out)

    def test_error_passthrough(self):
        from tars.format import format_strava_compare
        raw = json.dumps({"error": "Strava API error: timeout"})
        out = format_strava_compare(raw)
        self.assertIn("timeout", out)


class FormatRoutesTests(unittest.TestCase):
    def test_route_list(self):
        from tars.format import format_strava_routes
        raw = json.dumps({"routes": [
            {"id": 1001, "name": "Morning Loop", "type": "Run", "sub_type": "trail",
             "distance_km": 10.2, "elevation_gain_m": 250, "estimated_time_min": 65,
             "starred": True},
        ]})
        out = format_strava_routes(raw)
        self.assertIn("[Run/trail]", out)
        self.assertIn("Morning Loop", out)
        self.assertIn("10.2km", out)
        self.assertIn("id:1001", out)
        self.assertIn("*", out)

    def test_route_detail_with_segments(self):
        from tars.format import format_strava_routes
        raw = json.dumps({
            "id": 1001, "name": "Morning Loop", "type": "Run", "sub_type": "trail",
            "distance_km": 10.2, "elevation_gain_m": 250, "estimated_time_min": 65,
            "starred": True, "segments": [
                {"id": 5001, "name": "Hill Climb", "distance_km": 1.5,
                 "average_grade": 6.5, "climb_category": 3,
                 "pr": {"time_sec": 360}},
            ],
        })
        out = format_strava_routes(raw)
        self.assertIn("[Run/trail]", out)
        self.assertIn("segments:", out)
        self.assertIn("Hill Climb", out)
        self.assertIn("cat 3", out)
        self.assertIn("PR 6:00", out)

    def test_starred_segments(self):
        from tars.format import format_strava_routes
        raw = json.dumps({"segments": [
            {"id": 5001, "name": "Hill Climb", "distance_km": 1.5,
             "average_grade": 6.5, "climb_category": 3, "city": "Dublin",
             "pr": {"time_sec": 360}},
        ]})
        out = format_strava_routes(raw)
        self.assertIn("Hill Climb", out)
        self.assertIn("Dublin", out)
        self.assertIn("PR 6:00", out)

    def test_empty_list(self):
        from tars.format import format_strava_routes
        raw = json.dumps({"routes": []})
        out = format_strava_routes(raw)
        self.assertIn("no routes", out)

    def test_empty_segments(self):
        from tars.format import format_strava_routes
        raw = json.dumps({"segments": []})
        out = format_strava_routes(raw)
        self.assertIn("no starred segments", out)

    def test_error_passthrough(self):
        from tars.format import format_strava_routes
        raw = json.dumps({"error": "Strava API error: timeout"})
        out = format_strava_routes(raw)
        self.assertIn("timeout", out)


class SummariseGroupPrecisionTests(unittest.TestCase):
    def test_short_distance_pace_not_distorted_by_rounding(self):
        """A 450m run (rounds to 0.5km) should compute pace from 0.45km, not 0.5km."""
        a = _mock_activity(type="Run", distance=450.0, moving_time=180,
                           total_elevation_gain=0.0, average_heartrate=None,
                           suffer_score=None)
        result = strava._summarise_group("Run", [a])
        # 180s / 0.45km = 6.67 min/km (correct)
        # 180s / 0.5km  = 6.00 min/km (wrong, from pre-rounded value)
        self.assertAlmostEqual(result["avg_pace_min_per_km"], 6.67, places=1)

    def test_short_distance_speed_not_distorted_by_rounding(self):
        """A 450m ride (rounds to 0.5km) should compute speed from 0.45km."""
        a = _mock_activity(type="Ride", distance=450.0, moving_time=60,
                           total_elevation_gain=0.0, average_heartrate=None,
                           suffer_score=None)
        result = strava._summarise_group("Ride", [a])
        # 0.45km / (60/3600)h = 27.0 km/h (correct)
        # 0.5km  / (60/3600)h = 30.0 km/h (wrong)
        self.assertAlmostEqual(result["avg_speed_kmh"], 27.0, places=0)


class OverallTotalsTests(unittest.TestCase):
    def test_empty_list_returns_zeros(self):
        result = strava._overall_totals([])
        self.assertEqual(result["total_distance_km"], 0.0)
        self.assertEqual(result["total_time_hours"], 0.0)
        self.assertEqual(result["total_elevation_m"], 0)
        self.assertEqual(result["avg_distance_km"], 0.0)

    def test_single_activity(self):
        a = _mock_activity(distance=10000.0, moving_time=3600, total_elevation_gain=100.0)
        result = strava._overall_totals([a])
        self.assertEqual(result["total_distance_km"], 10.0)
        self.assertEqual(result["total_time_hours"], 1.0)
        self.assertEqual(result["total_elevation_m"], 100)
        self.assertEqual(result["avg_distance_km"], 10.0)

    def test_mixed_types_aggregate(self):
        run = _mock_activity(type="Run", distance=5000.0, moving_time=1500, total_elevation_gain=50.0)
        walk = _mock_activity(type="Walk", distance=3000.0, moving_time=1800, total_elevation_gain=20.0)
        result = strava._overall_totals([run, walk])
        self.assertEqual(result["total_distance_km"], 8.0)
        self.assertEqual(result["avg_distance_km"], 4.0)

    def test_avg_distance_div_by_count(self):
        a1 = _mock_activity(distance=6000.0, moving_time=1800, total_elevation_gain=0.0)
        a2 = _mock_activity(distance=4000.0, moving_time=1200, total_elevation_gain=0.0)
        result = strava._overall_totals([a1, a2])
        self.assertEqual(result["avg_distance_km"], 5.0)


class AnalysisToolTests(unittest.TestCase):
    def setUp(self):
        self.client = mock.Mock()

    @mock.patch.object(strava, "_get_client")
    def test_basic_output_keys(self, mock_get):
        mock_get.return_value = self.client
        activities = [_mock_activity(id=1), _mock_activity(id=2)]
        self.client.get_activities.return_value = activities

        result = json.loads(strava._run_strava_tool("strava_analysis", {"period": "this-week"}))
        for key in ("period", "period_dates", "count", "overall", "by_type",
                     "compare_period", "compare_period_dates", "compare_count",
                     "compare_overall", "compare_by_type", "overall_delta", "by_type_delta"):
            self.assertIn(key, result, f"missing key: {key}")

    @mock.patch.object(strava, "_get_client")
    def test_overall_totals_match_activities(self, mock_get):
        mock_get.return_value = self.client
        a1 = _mock_activity(distance=5000.0, moving_time=1500, total_elevation_gain=50.0)
        a2 = _mock_activity(distance=3000.0, moving_time=900, total_elevation_gain=30.0)
        self.client.get_activities.return_value = [a1, a2]

        result = json.loads(strava._run_strava_tool("strava_analysis", {"period": "this-week"}))
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["overall"]["total_distance_km"], 8.0)

    @mock.patch.object(strava, "_get_client")
    def test_auto_derived_compare_period(self, mock_get):
        mock_get.return_value = self.client
        self.client.get_activities.return_value = []

        result = json.loads(strava._run_strava_tool("strava_analysis", {"period": "this-week"}))
        self.assertEqual(result["compare_period"], "last-week")

    @mock.patch.object(strava, "_get_client")
    def test_explicit_compare_period(self, mock_get):
        mock_get.return_value = self.client
        self.client.get_activities.return_value = []

        result = json.loads(strava._run_strava_tool("strava_analysis", {
            "period": "this-week", "compare_period": "last-month"
        }))
        self.assertEqual(result["compare_period"], "last-month")

    @mock.patch.object(strava, "_get_client")
    def test_type_filter_applied_both_periods(self, mock_get):
        mock_get.return_value = self.client
        run = _mock_activity(type="Run", distance=5000.0)
        ride = _mock_activity(type="Ride", distance=10000.0)
        self.client.get_activities.return_value = [run, ride]

        result = json.loads(strava._run_strava_tool("strava_analysis", {
            "period": "this-week", "type": "Run"
        }))
        self.assertEqual(result["count"], 1)
        self.assertIn("Run", result["by_type"])
        self.assertNotIn("Ride", result["by_type"])

    @mock.patch.object(strava, "_get_client")
    def test_invalid_period_returns_error(self, mock_get):
        mock_get.return_value = self.client
        result = json.loads(strava._run_strava_tool("strava_analysis", {"period": "garbage"}))
        self.assertIn("error", result)

    @mock.patch.object(strava, "_get_client")
    def test_invalid_type_returns_error(self, mock_get):
        mock_get.return_value = self.client
        result = json.loads(strava._run_strava_tool("strava_analysis", {"type": "InvalidSport"}))
        self.assertIn("error", result)

    @mock.patch.object(strava, "_get_client")
    def test_empty_period_returns_zero_overall(self, mock_get):
        mock_get.return_value = self.client
        self.client.get_activities.return_value = []

        result = json.loads(strava._run_strava_tool("strava_analysis", {"period": "this-week"}))
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["overall"]["total_distance_km"], 0.0)
        self.assertEqual(result["by_type"], {})

    @mock.patch.object(strava, "_get_client")
    def test_by_type_delta_only_intersection(self, mock_get):
        mock_get.return_value = self.client
        run = _mock_activity(type="Run", distance=5000.0)
        walk = _mock_activity(type="Walk", distance=3000.0)

        call_count = [0]
        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [run, walk]
            return [run]

        self.client.get_activities.side_effect = side_effect

        result = json.loads(strava._run_strava_tool("strava_analysis", {"period": "this-week"}))
        self.assertIn("Run", result["by_type_delta"])
        self.assertNotIn("Walk", result["by_type_delta"])

    @mock.patch.object(strava, "_get_client")
    def test_default_period_is_this_week(self, mock_get):
        mock_get.return_value = self.client
        self.client.get_activities.return_value = []

        result = json.loads(strava._run_strava_tool("strava_analysis", {}))
        self.assertEqual(result["period"], "this-week")


class FormatAnalysisTests(unittest.TestCase):
    def _make_analysis_json(self, **overrides):
        data = {
            "period": "this-week",
            "period_dates": {"after": "2026-03-02", "before": "2026-03-08"},
            "count": 3,
            "overall": {
                "total_distance_km": 15.0, "total_time_hours": 1.5,
                "total_elevation_m": 100, "avg_distance_km": 5.0,
            },
            "by_type": {
                "Run": {
                    "count": 2, "total_distance_km": 10.0, "total_time_hours": 1.0,
                    "total_elevation_m": 80, "avg_pace_min_per_km": 6.0, "avg_heartrate": 148.0,
                },
                "Walk": {
                    "count": 1, "total_distance_km": 5.0, "total_time_hours": 0.5,
                    "total_elevation_m": 20,
                },
            },
            "compare_period": "last-week",
            "compare_period_dates": {"after": "2026-02-23", "before": "2026-03-01"},
            "compare_count": 2,
            "compare_overall": {
                "total_distance_km": 12.0, "total_time_hours": 1.2,
                "total_elevation_m": 80, "avg_distance_km": 6.0,
            },
            "compare_by_type": {
                "Run": {
                    "count": 2, "total_distance_km": 12.0, "total_time_hours": 1.2,
                    "total_elevation_m": 80, "avg_pace_min_per_km": 6.1, "avg_heartrate": 151.0,
                },
            },
            "overall_delta": {
                "total_distance_km": {"change": 3.0, "pct": 25.0},
                "total_time_hours": {"change": 0.3, "pct": 25.0},
                "total_elevation_m": {"change": 20, "pct": 25.0},
            },
            "by_type_delta": {
                "Run": {
                    "total_distance_km": {"change": -2.0, "pct": -16.7},
                },
            },
        }
        data.update(overrides)
        return json.dumps(data)

    def test_period_header_with_totals(self):
        from tars.format import format_strava_analysis
        out = format_strava_analysis(self._make_analysis_json())
        self.assertIn("this-week", out)
        self.assertIn("15.0km", out)
        self.assertIn("3 activities", out)

    def test_per_type_lines(self):
        from tars.format import format_strava_analysis
        out = format_strava_analysis(self._make_analysis_json())
        self.assertIn("Run x2", out)
        self.assertIn("Walk x1", out)

    def test_compare_period_present(self):
        from tars.format import format_strava_analysis
        out = format_strava_analysis(self._make_analysis_json())
        self.assertIn("last-week", out)
        self.assertIn("2 activities", out)

    def test_changes_line_with_deltas(self):
        from tars.format import format_strava_analysis
        out = format_strava_analysis(self._make_analysis_json())
        self.assertIn("Changes:", out)
        self.assertIn("distance", out)

    def test_error_passthrough(self):
        from tars.format import format_strava_analysis
        out = format_strava_analysis(json.dumps({"error": "something broke"}))
        self.assertEqual(out, "something broke")

    def test_dates_displayed(self):
        from tars.format import format_strava_analysis
        out = format_strava_analysis(self._make_analysis_json())
        self.assertIn("2026-03-02", out)
        self.assertIn("2026-03-08", out)

    def test_empty_data_no_activities(self):
        from tars.format import format_strava_analysis
        raw = json.dumps({
            "period": "this-week", "period_dates": {}, "count": 0,
            "overall": {"total_distance_km": 0, "total_time_hours": 0, "total_elevation_m": 0, "avg_distance_km": 0},
            "by_type": {},
            "compare_period": "last-week", "compare_period_dates": {}, "compare_count": 0,
            "compare_overall": {"total_distance_km": 0, "total_time_hours": 0, "total_elevation_m": 0, "avg_distance_km": 0},
            "compare_by_type": {},
            "overall_delta": {}, "by_type_delta": {},
        })
        out = format_strava_analysis(raw)
        self.assertIn("no activities", out)


class CompareLabelTests(unittest.TestCase):
    def test_today(self):
        self.assertEqual(strava._compare_label("today"), "yesterday")

    def test_yesterday(self):
        self.assertEqual(strava._compare_label("yesterday"), "2-days-ago")

    def test_this_week(self):
        self.assertEqual(strava._compare_label("this-week"), "last-week")

    def test_numeric_period(self):
        self.assertEqual(strava._compare_label("7d"), "prior 7d")


def _strict_get_activities(activities):
    """Return a mock get_activities that only accepts before/after/limit kwargs."""
    def _get_activities(*, before=None, after=None, limit=None):
        return iter(activities)
    return _get_activities


class TypeFilteredFetchTests(unittest.TestCase):
    def setUp(self):
        self.client = mock.Mock()

    @mock.patch.object(strava, "_get_client")
    def test_filters_matching_type(self, mock_get):
        mock_get.return_value = self.client
        all_activities = [
            _mock_activity(type="Ride", distance=10000.0),
            _mock_activity(type="Run", distance=5000.0),
            _mock_activity(type="Ride", distance=15000.0),
        ]
        self.client.get_activities = _strict_get_activities(all_activities)

        result = json.loads(strava._run_strava_tool("strava_activities", {
            "type": "Run", "limit": 5
        }))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "Run")

    @mock.patch.object(strava, "_get_client")
    def test_stops_at_limit(self, mock_get):
        mock_get.return_value = self.client
        all_activities = [_mock_activity(type="Run", distance=5000.0)] * 10
        self.client.get_activities = _strict_get_activities(all_activities)

        result = json.loads(strava._run_strava_tool("strava_activities", {
            "type": "Run", "limit": 3
        }))
        self.assertEqual(len(result), 3)

    @mock.patch.object(strava, "_get_client")
    def test_no_matches_returns_empty(self, mock_get):
        mock_get.return_value = self.client
        all_activities = [_mock_activity(type="Ride", distance=10000.0)] * 5
        self.client.get_activities = _strict_get_activities(all_activities)

        result = json.loads(strava._run_strava_tool("strava_activities", {
            "type": "Run", "limit": 5
        }))
        self.assertEqual(len(result), 0)

    @mock.patch.object(strava, "_get_client")
    def test_rejects_page_kwarg(self, mock_get):
        """Ensure get_activities is not called with page= (stravalib doesn't support it)."""
        mock_get.return_value = self.client
        self.client.get_activities = _strict_get_activities([])

        # Should not raise — if page= were passed, _strict_get_activities would TypeError
        result = json.loads(strava._run_strava_tool("strava_activities", {
            "type": "Run", "limit": 5
        }))
        self.assertEqual(len(result), 0)

    @mock.patch.object(strava, "_get_client")
    def test_oldest_sort_returns_oldest_matches(self, mock_get):
        """sort=oldest with type filter should return the oldest matching activities,
        not just reverse the first N matches from newest-first iteration."""
        mock_get.return_value = self.client
        # API yields newest-first: ids 5,4,3,2,1
        all_runs = [
            _mock_activity(id=5, type="Run", distance=5000.0,
                           start_date_local=datetime(2026, 3, 5, 8, 0)),
            _mock_activity(id=4, type="Run", distance=5000.0,
                           start_date_local=datetime(2026, 3, 4, 8, 0)),
            _mock_activity(id=3, type="Run", distance=5000.0,
                           start_date_local=datetime(2026, 3, 3, 8, 0)),
            _mock_activity(id=2, type="Run", distance=5000.0,
                           start_date_local=datetime(2026, 3, 2, 8, 0)),
            _mock_activity(id=1, type="Run", distance=5000.0,
                           start_date_local=datetime(2026, 3, 1, 8, 0)),
        ]
        self.client.get_activities = _strict_get_activities(all_runs)

        result = json.loads(strava._run_strava_tool("strava_activities", {
            "type": "Run", "sort": "oldest", "limit": 2
        }))
        self.assertEqual(len(result), 2)
        # Should be the oldest 2 (ids 1, 2), not the newest 2 reversed
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[1]["id"], 2)


class UnknownToolTests(unittest.TestCase):
    @mock.patch.object(strava, "_get_client")
    def test_unknown_tool_returns_error(self, mock_get):
        mock_get.return_value = mock.Mock()
        result = json.loads(strava._run_strava_tool("strava_unknown", {}))
        self.assertIn("error", result)


def _mock_hr_zones_5():
    """Return a mock zones object with 5 HR zones."""
    zones = mock.Mock()
    hr = mock.Mock()
    z_list = [
        mock.Mock(min=0, max=120),
        mock.Mock(min=120, max=145),
        mock.Mock(min=145, max=160),
        mock.Mock(min=160, max=175),
        mock.Mock(min=175, max=-1),
    ]
    for z in z_list:
        z.root = None
    hr.zones = z_list
    hr.root = None
    zones.heart_rate = hr
    return zones


def _mock_stream(hr_data, time_data):
    """Create a mock stream response dict."""
    hr_stream = mock.Mock()
    hr_stream.data = hr_data
    time_stream = mock.Mock()
    time_stream.data = time_data
    return {"heartrate": hr_stream, "time": time_stream}


class GetThreeZoneBoundariesTests(unittest.TestCase):
    def test_returns_boundaries(self):
        client = mock.Mock()
        client.get_athlete_zones.return_value = _mock_hr_zones_5()
        result = strava._get_3_zone_boundaries(client)
        self.assertEqual(result, (145, 175))

    def test_returns_error_no_zones(self):
        client = mock.Mock()
        zones = mock.Mock()
        zones.heart_rate = None
        client.get_athlete_zones.return_value = zones
        result = strava._get_3_zone_boundaries(client)
        self.assertIsInstance(result, str)
        self.assertIn("not configured", result)

    def test_returns_error_too_few_zones(self):
        client = mock.Mock()
        zones = mock.Mock()
        hr = mock.Mock()
        hr.zones = [mock.Mock(min=0, max=120)]
        hr.root = None
        zones.heart_rate = hr
        client.get_athlete_zones.return_value = zones
        result = strava._get_3_zone_boundaries(client)
        self.assertIsInstance(result, str)


class ClassifyZonesTests(unittest.TestCase):
    def test_polarised(self):
        self.assertEqual(strava._classify_zones(80, 5, 15), "Polarised")

    def test_pyramidal(self):
        self.assertEqual(strava._classify_zones(72, 18, 10), "Pyramidal")

    def test_threshold_heavy(self):
        self.assertEqual(strava._classify_zones(50, 30, 20), "Threshold-Heavy")

    def test_unstructured(self):
        self.assertEqual(strava._classify_zones(60, 15, 25), "Unstructured")


class HandleZonesTests(unittest.TestCase):
    def setUp(self):
        self.client = mock.Mock()
        self.client.get_athlete_zones.return_value = _mock_hr_zones_5()

    @mock.patch.object(strava, "_get_client")
    def test_basic_zone_analysis(self, mock_get):
        mock_get.return_value = self.client
        activities = [
            _mock_activity(id=1, moving_time=3600, average_heartrate=140),
            _mock_activity(id=2, moving_time=3600, average_heartrate=150),
            _mock_activity(id=3, moving_time=3600, average_heartrate=170),
        ]
        self.client.get_activities = _strict_get_activities(activities)
        self.client.get_activity_streams.return_value = _mock_stream(
            [130] * 61, list(range(0, 3660, 60))
        )

        result = json.loads(strava._run_strava_tool("strava_zones", {"period": "4w"}))
        self.assertIn("classification", result)
        self.assertIn("zone_pct", result)
        self.assertEqual(result["activities_analysed"], 3)
        self.assertIn("zone_boundaries", result)
        self.assertEqual(result["zone_boundaries"]["low_max"], 145)
        self.assertEqual(result["zone_boundaries"]["mod_max"], 175)

    @mock.patch.object(strava, "_get_client")
    def test_skips_no_hr(self, mock_get):
        mock_get.return_value = self.client
        activities = [
            _mock_activity(id=1, moving_time=3600, average_heartrate=None),
            _mock_activity(id=2, moving_time=3600, average_heartrate=140),
        ]
        self.client.get_activities = _strict_get_activities(activities)
        self.client.get_activity_streams.return_value = _mock_stream(
            [130] * 61, list(range(0, 3660, 60))
        )

        result = json.loads(strava._run_strava_tool("strava_zones", {"period": "4w"}))
        self.assertEqual(result["activities_analysed"], 1)
        self.assertEqual(result["activities_skipped"]["no_hr"], 1)

    @mock.patch.object(strava, "_get_client")
    def test_skips_short_activities(self, mock_get):
        mock_get.return_value = self.client
        activities = [
            _mock_activity(id=1, moving_time=600, average_heartrate=140),
            _mock_activity(id=2, moving_time=3600, average_heartrate=140),
        ]
        self.client.get_activities = _strict_get_activities(activities)
        self.client.get_activity_streams.return_value = _mock_stream(
            [130] * 61, list(range(0, 3660, 60))
        )

        result = json.loads(strava._run_strava_tool("strava_zones", {"period": "4w"}))
        self.assertEqual(result["activities_analysed"], 1)
        self.assertEqual(result["activities_skipped"]["too_short"], 1)

    @mock.patch.object(strava, "_get_client")
    def test_type_filter(self, mock_get):
        mock_get.return_value = self.client
        activities = [
            _mock_activity(id=1, type="Ride", moving_time=3600, average_heartrate=140),
            _mock_activity(id=2, type="Run", moving_time=3600, average_heartrate=140),
        ]
        self.client.get_activities = _strict_get_activities(activities)
        self.client.get_activity_streams.return_value = _mock_stream(
            [130] * 61, list(range(0, 3660, 60))
        )

        result = json.loads(strava._run_strava_tool("strava_zones", {"period": "4w", "type": "Run"}))
        self.assertEqual(result["activities_analysed"], 1)

    @mock.patch.object(strava, "_get_client")
    def test_stream_fetch_cap(self, mock_get):
        mock_get.return_value = self.client
        activities = [
            _mock_activity(id=i, moving_time=3600, average_heartrate=140)
            for i in range(60)
        ]
        self.client.get_activities = _strict_get_activities(activities)
        self.client.get_activity_streams.return_value = _mock_stream(
            [130] * 61, list(range(0, 3660, 60))
        )

        result = json.loads(strava._run_strava_tool("strava_zones", {"period": "4w"}))
        self.assertEqual(self.client.get_activity_streams.call_count, 50)
        self.assertEqual(result["activities_skipped"]["over_cap"], 10)

    @mock.patch.object(strava, "_get_client")
    def test_stream_types_requested(self, mock_get):
        mock_get.return_value = self.client
        activities = [_mock_activity(id=1, moving_time=3600, average_heartrate=140)]
        self.client.get_activities = _strict_get_activities(activities)
        self.client.get_activity_streams.return_value = _mock_stream(
            [130] * 61, list(range(0, 3660, 60))
        )

        strava._run_strava_tool("strava_zones", {"period": "4w"})
        self.client.get_activity_streams.assert_called_with(
            1, types=["heartrate", "time"]
        )

    @mock.patch.object(strava, "_get_client")
    def test_per_activity_detail(self, mock_get):
        mock_get.return_value = self.client
        activities = [_mock_activity(id=1, moving_time=3600, average_heartrate=140)]
        self.client.get_activities = _strict_get_activities(activities)
        hr = [130, 150, 180]
        time_data = [0, 1200, 2400]
        self.client.get_activity_streams.return_value = _mock_stream(hr, time_data)

        result = json.loads(strava._run_strava_tool("strava_zones", {"period": "4w"}))
        self.assertEqual(len(result["per_activity"]), 1)
        pa = result["per_activity"][0]
        self.assertIn("low_pct", pa)
        self.assertIn("mod_pct", pa)
        self.assertIn("high_pct", pa)

    @mock.patch.object(strava, "_get_client")
    def test_zones_unavailable_error(self, mock_get):
        mock_get.return_value = self.client
        zones = mock.Mock()
        zones.heart_rate = None
        self.client.get_athlete_zones.return_value = zones

        result = json.loads(strava._run_strava_tool("strava_zones", {"period": "4w"}))
        self.assertIn("error", result)

    @mock.patch.object(strava, "_get_client")
    def test_boundary_hr_exact(self, mock_get):
        """HR exactly on boundary: low_max=145 → hr=145 is moderate, not low."""
        mock_get.return_value = self.client
        activities = [_mock_activity(id=1, moving_time=3600, average_heartrate=145)]
        self.client.get_activities = _strict_get_activities(activities)
        # 3 data points: hr exactly at low_max (145) and mod_max (175)
        hr = [100, 145, 175]
        time_data = [0, 1200, 2400]
        self.client.get_activity_streams.return_value = _mock_stream(hr, time_data)

        result = json.loads(strava._run_strava_tool("strava_zones", {"period": "4w"}))
        pa = result["per_activity"][0]
        # hr=145 at i=1: 145 is NOT < 145 (low_max), so it's moderate
        # hr=175 at i=2: 175 is NOT < 175 (mod_max), so it's high
        self.assertGreater(pa["mod_pct"], 0)
        self.assertGreater(pa["high_pct"], 0)

    @mock.patch.object(strava, "_get_client")
    def test_stream_api_errors_surface(self, mock_get):
        """Repeated non-data API errors should surface, not silently skip."""
        mock_get.return_value = self.client
        activities = [
            _mock_activity(id=i, moving_time=3600, average_heartrate=140)
            for i in range(5)
        ]
        self.client.get_activities = _strict_get_activities(activities)
        self.client.get_activity_streams.side_effect = RuntimeError("rate limited")

        result = json.loads(strava._run_strava_tool("strava_zones", {"period": "4w"}))
        self.assertIn("error", result)
        self.assertIn("API error", result["error"])

    @mock.patch.object(strava, "_get_client")
    def test_stream_data_errors_skipped(self, mock_get):
        """KeyError/ValueError from missing stream data should skip, not error."""
        mock_get.return_value = self.client
        activities = [
            _mock_activity(id=1, moving_time=3600, average_heartrate=140),
            _mock_activity(id=2, moving_time=3600, average_heartrate=140),
        ]
        self.client.get_activities = _strict_get_activities(activities)
        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise KeyError("heartrate")
            return _mock_stream([130] * 61, list(range(0, 3660, 60)))

        self.client.get_activity_streams.side_effect = _side_effect

        result = json.loads(strava._run_strava_tool("strava_zones", {"period": "4w"}))
        self.assertEqual(result["activities_analysed"], 1)
        self.assertEqual(result["activities_skipped"]["no_hr"], 1)


if __name__ == "__main__":
    unittest.main()
