"""Tests for the daily brief module."""

import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars.brief import build_brief_sections, build_daily_context, build_review_sections, format_brief_cli, format_brief_text


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


class BuildDailyContextTests(unittest.TestCase):
    @mock.patch("tars.brief.run_tool", return_value='{"tasks": [{"content": "Buy milk"}]}')
    def test_returns_tasks_and_weather(self, mock_run) -> None:
        result = build_daily_context()
        self.assertIn("[tasks]", result)
        self.assertIn("[weather]", result)
        self.assertEqual(mock_run.call_count, 2)
        tool_names = [c[0][0] for c in mock_run.call_args_list]
        self.assertEqual(tool_names, ["todoist_today", "weather_now"])

    @mock.patch("tars.brief.run_tool", side_effect=Exception("api down"))
    def test_all_failures_returns_empty(self, mock_run) -> None:
        result = build_daily_context()
        self.assertEqual(result, "")

    @mock.patch("tars.brief.run_tool")
    def test_partial_failure_returns_available(self, mock_run) -> None:
        def side_effect(name, args, **kwargs):
            if name == "todoist_today":
                return '{"tasks": []}'
            raise Exception("weather down")
        mock_run.side_effect = side_effect
        result = build_daily_context()
        self.assertIn("[tasks]", result)
        self.assertNotIn("[weather]", result)


class FormatBriefCliTests(unittest.TestCase):
    def test_pinned_section_formatted(self) -> None:
        sections = [("pinned", "- item one"), ("tasks", "no tasks")]
        result = format_brief_cli(sections)
        self.assertIn("pinned", result)
        self.assertIn("item one", result)


class BuildReviewSectionsTests(unittest.TestCase):
    @mock.patch("tars.commands._dispatch_review", return_value="suggested rules:\n  - rule one")
    @mock.patch("tars.commands._dispatch_tidy", return_value="proposed removals:\n  [semantic] stale entry")
    def test_returns_tidy_and_review_sections(self, mock_tidy, mock_review) -> None:
        sections = build_review_sections("claude", "sonnet")
        labels = [s[0] for s in sections]
        self.assertEqual(labels, ["tidy", "review"])
        self.assertIn("proposed removals", sections[0][1])
        self.assertIn("suggested rules", sections[1][1])

    @mock.patch("tars.commands._dispatch_review", return_value="nothing to review")
    @mock.patch("tars.commands._dispatch_tidy", side_effect=Exception("model error"))
    def test_tidy_failure_degrades_gracefully(self, mock_tidy, mock_review) -> None:
        sections = build_review_sections("claude", "sonnet")
        labels = [s[0] for s in sections]
        self.assertIn("tidy", labels)
        self.assertIn("review", labels)
        tidy_content = next(c for l, c in sections if l == "tidy")
        self.assertIn("unavailable", tidy_content)

    @mock.patch("tars.commands._dispatch_review", side_effect=Exception("api down"))
    @mock.patch("tars.commands._dispatch_tidy", return_value="memory looks clean")
    def test_review_failure_degrades_gracefully(self, mock_tidy, mock_review) -> None:
        sections = build_review_sections("claude", "sonnet")
        review_content = next(c for l, c in sections if l == "review")
        self.assertIn("unavailable", review_content)

    @mock.patch("tars.commands._dispatch_review", return_value="nothing to review")
    @mock.patch("tars.commands._dispatch_tidy", return_value="memory looks clean")
    def test_clean_memory_returns_both_sections(self, mock_tidy, mock_review) -> None:
        sections = build_review_sections("claude", "sonnet")
        self.assertEqual(len(sections), 2)
        self.assertIn("memory looks clean", sections[0][1])
        self.assertIn("nothing to review", sections[1][1])

    @mock.patch("tars.commands._dispatch_review", return_value="suggested rules:\n  - rule one")
    @mock.patch("tars.commands._dispatch_tidy", return_value="proposed removals:\n  [semantic] stale entry")
    def test_formatted_output(self, mock_tidy, mock_review) -> None:
        sections = build_review_sections("claude", "sonnet")
        text = format_brief_text(sections)
        self.assertIn("[tidy]", text)
        self.assertIn("[review]", text)
        self.assertIn("proposed removals", text)
        self.assertIn("suggested rules", text)


if __name__ == "__main__":
    unittest.main()
