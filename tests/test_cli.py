import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars.cli import (
    _completer,
    _handle_review,
    _handle_slash_search,
    _handle_slash_tool,
    _parse_todoist_add,
)


class ParseTodoistAddTests(unittest.TestCase):
    def test_simple(self) -> None:
        args = _parse_todoist_add(["buy", "eggs"])
        self.assertEqual(args["content"], "buy eggs")

    def test_single_word_due(self) -> None:
        args = _parse_todoist_add(["buy", "eggs", "--due", "today"])
        self.assertEqual(args["content"], "buy eggs")
        self.assertEqual(args["due"], "today")

    def test_multi_word_due(self) -> None:
        args = _parse_todoist_add(["buy", "eggs", "--due", "tomorrow", "3pm"])
        self.assertEqual(args["content"], "buy eggs")
        self.assertEqual(args["due"], "tomorrow 3pm")

    def test_due_with_date_string(self) -> None:
        args = _parse_todoist_add(["buy", "eggs", "--due", "next", "monday"])
        self.assertEqual(args["due"], "next monday")
        self.assertEqual(args["content"], "buy eggs")

    def test_all_flags(self) -> None:
        args = _parse_todoist_add([
            "buy", "eggs", "--due", "tomorrow", "3pm",
            "--project", "Groceries", "--priority", "4",
        ])
        self.assertEqual(args["content"], "buy eggs")
        self.assertEqual(args["due"], "tomorrow 3pm")
        self.assertEqual(args["project"], "Groceries")
        self.assertEqual(args["priority"], 4)

    def test_priority_is_int(self) -> None:
        args = _parse_todoist_add(["task", "--priority", "2"])
        self.assertEqual(args["priority"], 2)

    def test_priority_invalid_defaults_to_1(self) -> None:
        args = _parse_todoist_add(["task", "--priority", "high"])
        self.assertEqual(args["priority"], 1)

    def test_flags_before_content_gives_empty_content(self) -> None:
        args = _parse_todoist_add(["--due", "tomorrow", "buy", "eggs"])
        # Greedy parsing: "buy" and "eggs" consumed by --due
        self.assertEqual(args["content"], "")


class HandleReviewTests(unittest.TestCase):
    def test_nothing_to_review(self) -> None:
        with mock.patch("tars.cli.load_feedback", return_value=("", "")):
            with mock.patch("builtins.print") as mock_print:
                _handle_review("ollama", "test-model")
        mock_print.assert_any_call("  nothing to review")

    def test_review_applies_rules(self) -> None:
        corrections = "# Corrections\n\n## 2026-01-01T00:00:00\n- input: weather\n- got: nonsense\n"
        rewards = ""
        model_reply = "- route weather queries to weather_now\n- check memory before adding duplicates\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("tars.cli.load_feedback", return_value=(corrections, rewards)):
                with mock.patch("tars.cli.chat", return_value=model_reply):
                    with mock.patch("builtins.input", return_value="y"):
                        with mock.patch("tars.cli._memory_file", return_value=Path(tmpdir) / "Procedural.md"):
                            with mock.patch("tars.cli.archive_feedback") as mock_archive:
                                with mock.patch("builtins.print"):
                                    _handle_review("ollama", "test-model")
            # Rules should be written to Procedural.md
            text = (Path(tmpdir) / "Procedural.md").read_text()
            self.assertIn("- route weather queries to weather_now", text)
            self.assertIn("- check memory before adding duplicates", text)
            mock_archive.assert_called_once()

    def test_review_declined(self) -> None:
        corrections = "# Corrections\n\n## 2026-01-01T00:00:00\n- input: x\n- got: y\n"
        model_reply = "- some rule\n"
        with mock.patch("tars.cli.load_feedback", return_value=(corrections, "")):
            with mock.patch("tars.cli.chat", return_value=model_reply):
                with mock.patch("builtins.input", return_value="n"):
                    with mock.patch("tars.cli.archive_feedback") as mock_archive:
                        with mock.patch("builtins.print") as mock_print:
                            _handle_review("ollama", "test-model")
        mock_archive.assert_not_called()
        mock_print.assert_any_call("  skipped")

    def test_review_no_actionable_rules(self) -> None:
        corrections = "# Corrections\n\n## 2026-01-01T00:00:00\n- input: x\n- got: y\n"
        model_reply = "No clear patterns found in the corrections."
        with mock.patch("tars.cli.load_feedback", return_value=(corrections, "")):
            with mock.patch("tars.cli.chat", return_value=model_reply):
                with mock.patch("builtins.print") as mock_print:
                    _handle_review("ollama", "test-model")
        mock_print.assert_any_call("  no actionable rules found")

    def test_review_no_memory_dir(self) -> None:
        corrections = "# Corrections\n\n## 2026-01-01T00:00:00\n- input: x\n- got: y\n"
        model_reply = "- a rule\n"
        with mock.patch("tars.cli.load_feedback", return_value=(corrections, "")):
            with mock.patch("tars.cli.chat", return_value=model_reply):
                with mock.patch("builtins.input", return_value="y"):
                    with mock.patch("tars.cli._memory_file", return_value=None):
                        with mock.patch("builtins.print") as mock_print:
                            _handle_review("ollama", "test-model")
        mock_print.assert_any_call("  no memory dir configured")


class HandleSlashToolTests(unittest.TestCase):
    def test_weather_dispatches(self) -> None:
        with mock.patch("tars.cli._print_tool") as m:
            result = _handle_slash_tool("/weather")
        self.assertTrue(result)
        m.assert_called_once_with("weather_now", {})

    def test_forecast_dispatches(self) -> None:
        with mock.patch("tars.cli._print_tool") as m:
            result = _handle_slash_tool("/forecast")
        self.assertTrue(result)
        m.assert_called_once_with("weather_forecast", {})

    def test_todoist_today_dispatches(self) -> None:
        with mock.patch("tars.cli._print_tool") as m:
            result = _handle_slash_tool("/todoist today")
        self.assertTrue(result)
        m.assert_called_once_with("todoist_today", {})

    def test_unknown_returns_false(self) -> None:
        result = _handle_slash_tool("/unknown")
        self.assertFalse(result)

    def test_empty_content_prints_usage(self) -> None:
        with mock.patch("builtins.print") as m:
            result = _handle_slash_tool("/todoist add --due tomorrow")
        self.assertTrue(result)
        m.assert_any_call("  usage: /todoist add <text> [--due D] [--project P] [--priority N]")

    def test_upcoming_bad_days_prints_usage(self) -> None:
        with mock.patch("builtins.print") as m:
            result = _handle_slash_tool("/todoist upcoming abc")
        self.assertTrue(result)
        m.assert_any_call("  usage: /todoist upcoming [days]")


class HandleSlashSearchTests(unittest.TestCase):
    def test_search_dispatches(self) -> None:
        with mock.patch("tars.cli.search", return_value=[]) as m:
            with mock.patch("tars.cli._print_search_results"):
                result = _handle_slash_search("/search weather")
        self.assertTrue(result)
        m.assert_called_once_with("weather", mode="hybrid", limit=10)

    def test_sgrep_dispatches(self) -> None:
        with mock.patch("tars.cli.search", return_value=[]) as m:
            with mock.patch("tars.cli._print_search_results"):
                result = _handle_slash_search("/sgrep test query")
        self.assertTrue(result)
        m.assert_called_once_with("test query", mode="fts", limit=10)

    def test_no_query_prints_usage(self) -> None:
        with mock.patch("builtins.print") as m:
            result = _handle_slash_search("/search")
        self.assertTrue(result)
        m.assert_any_call("  usage: /search <query>")

    def test_unknown_returns_false(self) -> None:
        result = _handle_slash_search("/notasearch foo")
        self.assertFalse(result)


class CompleterTests(unittest.TestCase):
    def test_matches_slash_commands(self) -> None:
        with mock.patch("readline.get_line_buffer", return_value="/we"):
            result = _completer("/we", 0)
        self.assertEqual(result, "/weather")

    def test_todoist_subcommands(self) -> None:
        with mock.patch("readline.get_line_buffer", return_value="/todoist a"):
            result = _completer("/todoist a", 0)
        self.assertEqual(result, "/todoist add ")

    def test_returns_none_past_end(self) -> None:
        with mock.patch("readline.get_line_buffer", return_value="/weather"):
            result = _completer("/weather", 99)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
