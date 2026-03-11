"""Tests for the daily brief module."""

import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars.brief import build_brief_sections, format_brief_cli, format_brief_text


class BuildBriefSectionsTests(unittest.TestCase):
    @mock.patch("tars.brief._load_pinned", return_value="- watching Severance S2\n")
    @mock.patch("tars.brief.run_tool", return_value='{"tasks": []}')
    def test_pinned_included_when_present(self, mock_run, mock_pinned) -> None:
        sections = build_brief_sections()
        labels = [s[0] for s in sections]
        self.assertIn("pinned", labels)
        self.assertEqual(labels[0], "pinned")
        pinned_content = next(c for l, c in sections if l == "pinned")
        self.assertIn("watching Severance S2", pinned_content)

    @mock.patch("tars.brief._load_pinned", return_value="")
    @mock.patch("tars.brief.run_tool", return_value='{"tasks": []}')
    def test_pinned_omitted_when_empty(self, mock_run, mock_pinned) -> None:
        sections = build_brief_sections()
        labels = [s[0] for s in sections]
        self.assertNotIn("pinned", labels)

    @mock.patch("tars.brief._load_pinned", return_value="   \n  \n")
    @mock.patch("tars.brief.run_tool", return_value='{"tasks": []}')
    def test_pinned_omitted_when_whitespace_only(self, mock_run, mock_pinned) -> None:
        sections = build_brief_sections()
        labels = [s[0] for s in sections]
        self.assertNotIn("pinned", labels)


    @mock.patch("tars.brief._load_pinned", side_effect=OSError("disk error"))
    @mock.patch("tars.brief.run_tool", return_value='{"tasks": []}')
    def test_pinned_load_failure_degrades(self, mock_run, mock_pinned) -> None:
        sections = build_brief_sections()
        labels = [s[0] for s in sections]
        self.assertNotIn("pinned", labels)
        self.assertIn("tasks", labels)

    @mock.patch("tars.brief._load_tokens", return_value={"access_token": "tok"})
    @mock.patch("tars.brief._load_pinned", return_value="")
    @mock.patch("tars.brief.run_tool", return_value='{"activities": []}')
    def test_strava_included_when_tokens_exist(self, mock_run, mock_pinned, mock_tokens) -> None:
        sections = build_brief_sections()
        labels = [s[0] for s in sections]
        self.assertIn("strava", labels)

    @mock.patch("tars.brief._load_tokens", return_value=None)
    @mock.patch("tars.brief._load_pinned", return_value="")
    @mock.patch("tars.brief.run_tool", return_value='{"tasks": []}')
    def test_strava_omitted_when_no_tokens(self, mock_run, mock_pinned, mock_tokens) -> None:
        sections = build_brief_sections()
        labels = [s[0] for s in sections]
        self.assertNotIn("strava", labels)

    @mock.patch("tars.brief._load_tokens", return_value={"access_token": "tok"})
    @mock.patch("tars.brief._load_pinned", return_value="")
    @mock.patch("tars.brief.run_tool", side_effect=Exception("strava API error"))
    def test_strava_error_degrades_gracefully(self, mock_run, mock_pinned, mock_tokens) -> None:
        sections = build_brief_sections()
        strava = [c for l, c in sections if l == "strava"]
        self.assertEqual(len(strava), 1)
        self.assertIn("unavailable", strava[0])


class FormatBriefTextTests(unittest.TestCase):
    def test_pinned_section_formatted(self) -> None:
        sections = [("pinned", "- watching Severance S2"), ("tasks", "no tasks")]
        result = format_brief_text(sections)
        self.assertIn("[pinned]", result)
        self.assertIn("watching Severance S2", result)
        self.assertIn("[tasks]", result)


class BriefStravaWeekdayTests(unittest.TestCase):
    @mock.patch("tars.brief._load_tokens", return_value={"access_token": "tok"})
    @mock.patch("tars.brief._load_pinned", return_value="")
    @mock.patch("tars.brief.run_tool", return_value='{}')
    @mock.patch("tars.brief.datetime")
    def test_monday_calls_strava_analysis(self, mock_dt, mock_run, mock_pinned, mock_tokens) -> None:
        from datetime import datetime as real_dt, timezone as real_tz
        mock_dt.now.return_value = real_dt(2026, 3, 9, 8, 0, 0, tzinfo=real_tz.utc)  # Monday
        mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
        build_brief_sections()
        strava_calls = [c for c in mock_run.call_args_list if "strava" in c[0][0]]
        self.assertEqual(len(strava_calls), 1)
        self.assertEqual(strava_calls[0][0][0], "strava_analysis")
        self.assertEqual(strava_calls[0][0][1], {"period": "last-week"})

    @mock.patch("tars.brief._load_tokens", return_value={"access_token": "tok"})
    @mock.patch("tars.brief._load_pinned", return_value="")
    @mock.patch("tars.brief.run_tool", return_value='{}')
    @mock.patch("tars.brief.datetime")
    def test_non_monday_calls_strava_activities(self, mock_dt, mock_run, mock_pinned, mock_tokens) -> None:
        from datetime import datetime as real_dt, timezone as real_tz
        mock_dt.now.return_value = real_dt(2026, 3, 11, 8, 0, 0, tzinfo=real_tz.utc)  # Wednesday
        mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
        build_brief_sections()
        strava_calls = [c for c in mock_run.call_args_list if "strava" in c[0][0]]
        self.assertEqual(len(strava_calls), 1)
        self.assertEqual(strava_calls[0][0][0], "strava_activities")
        self.assertEqual(strava_calls[0][0][1], {"period": "1d", "limit": 5})


class FormatBriefCliTests(unittest.TestCase):
    def test_pinned_section_formatted(self) -> None:
        sections = [("pinned", "- item one"), ("tasks", "no tasks")]
        result = format_brief_cli(sections)
        self.assertIn("pinned", result)
        self.assertIn("item one", result)


if __name__ == "__main__":
    unittest.main()
