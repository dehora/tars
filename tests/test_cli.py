import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars.cli import _apply_review, _apply_tidy, _completer


class ApplyReviewTests(unittest.TestCase):
    def test_apply_review_writes_rules(self) -> None:
        result = (
            "reviewing 1 corrections, 0 rewards...\n\n"
            "suggested rules:\n"
            "  - route weather queries to weather_now\n"
            "  - check memory before adding duplicates\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("builtins.input", return_value="y"):
                with mock.patch("tars.cli._memory_file", return_value=Path(tmpdir) / "Procedural.md"):
                    with mock.patch("tars.cli.archive_feedback") as mock_archive:
                        with mock.patch("tars.cli.build_index") as mock_index:
                            with mock.patch("builtins.print"):
                                _apply_review(result)
            text = (Path(tmpdir) / "Procedural.md").read_text()
            self.assertIn("- route weather queries to weather_now", text)
            self.assertIn("- check memory before adding duplicates", text)
            mock_archive.assert_called_once()
            mock_index.assert_called_once()

    def test_apply_review_declined(self) -> None:
        result = "suggested rules:\n  - some rule\n"
        with mock.patch("builtins.input", return_value="n"):
            with mock.patch("tars.cli.archive_feedback") as mock_archive:
                with mock.patch("builtins.print") as mock_print:
                    _apply_review(result)
        mock_archive.assert_not_called()
        mock_print.assert_any_call("  skipped")

    def test_apply_review_no_rules(self) -> None:
        result = "no actionable rules found"
        with mock.patch("builtins.print"):
            _apply_review(result)
        # Should not prompt — no rules to apply

    def test_apply_review_no_memory_dir(self) -> None:
        result = "suggested rules:\n  - a rule\n"
        with mock.patch("builtins.input", return_value="y"):
            with mock.patch("tars.cli._memory_file", return_value=None):
                with mock.patch("builtins.print") as mock_print:
                    _apply_review(result)
        mock_print.assert_any_call("  no memory dir configured")

    def test_apply_review_reindex_failure(self) -> None:
        result = "suggested rules:\n  - a rule\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("builtins.input", return_value="y"):
                with mock.patch("tars.cli._memory_file", return_value=Path(tmpdir) / "Procedural.md"):
                    with mock.patch("tars.cli.archive_feedback"):
                        with mock.patch("tars.cli.build_index", side_effect=RuntimeError("no db")):
                            with mock.patch("builtins.print") as mock_print:
                                _apply_review(result)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("reindex failed", output)


class ApplyTidyTests(unittest.TestCase):
    def test_apply_tidy_removes_entries(self) -> None:
        result = "proposed removals:\n  [semantic] lorem ipsum\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "Memory.md"
            p.write_text("# Memory\n- real fact\n- lorem ipsum\n")
            with mock.patch("builtins.input", return_value="y"):
                with mock.patch("tars.cli._memory_file", return_value=p):
                    with mock.patch("builtins.print"):
                        _apply_tidy(result)
            text = p.read_text()
            self.assertIn("- real fact", text)
            self.assertNotIn("- lorem ipsum", text)

    def test_apply_tidy_declined(self) -> None:
        result = "proposed removals:\n  [semantic] junk\n"
        with mock.patch("builtins.input", return_value="n"):
            with mock.patch("builtins.print") as m:
                _apply_tidy(result)
        m.assert_any_call("  skipped")

    def test_apply_tidy_no_removals(self) -> None:
        result = "memory looks clean"
        with mock.patch("builtins.print"):
            _apply_tidy(result)
        # No prompt expected


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


class OneShotHintTests(unittest.TestCase):
    def test_hint_in_search_context_not_message(self) -> None:
        from tars.cli import main

        captured: dict = {}

        def fake_process_message(conv, message, session_file):
            captured["conv"] = conv
            captured["message"] = message
            return "ok"

        with (
            mock.patch("sys.argv", ["tars", "hello there"]),
            mock.patch("tars.cli.process_message", side_effect=fake_process_message),
            mock.patch("tars.cli._startup_index"),
            mock.patch("tars.cli._session_path", return_value=None),
            mock.patch("builtins.print"),
        ):
            main()

        self.assertIn("conv", captured)
        self.assertIn("message", captured)
        self.assertNotIn("one-shot", captured["message"])
        self.assertIn("one-shot", captured["conv"].search_context)

    def test_message_content_unchanged(self) -> None:
        from tars.cli import main

        captured: dict = {}

        def fake_process_message(conv, message, session_file):
            captured["message"] = message
            return "ok"

        with (
            mock.patch("sys.argv", ["tars", "what", "is", "the", "weather"]),
            mock.patch("tars.cli.process_message", side_effect=fake_process_message),
            mock.patch("tars.cli._startup_index"),
            mock.patch("tars.cli._session_path", return_value=None),
            mock.patch("builtins.print"),
        ):
            main()

        self.assertEqual(captured["message"], "what is the weather")


class HandleBriefTests(unittest.TestCase):
    def test_brief_returns_sections(self) -> None:
        from tars.commands import dispatch

        def fake_run_tool(name, args, *, quiet=False):
            if name == "todoist_today":
                return '{"results": []}'
            if name == "weather_now":
                return '{"current": {"temperature_c": 10, "conditions": "Clear", "wind_speed_kmh": 5, "precipitation_mm": 0}}'
            if name == "weather_forecast":
                return '{"hourly": []}'
            return '{}'

        with mock.patch("tars.brief.run_tool", side_effect=fake_run_tool):
            result = dispatch("/brief")
        self.assertIn("[tasks]", result)
        self.assertIn("[weather]", result)
        self.assertIn("[forecast]", result)

    def test_brief_handles_tool_failure(self) -> None:
        from tars.commands import dispatch

        def fake_run_tool(name, args, *, quiet=False):
            if name == "todoist_today":
                raise FileNotFoundError("td not found")
            if name == "weather_now":
                return '{"current": {"temperature_c": 10, "conditions": "Clear", "wind_speed_kmh": 5, "precipitation_mm": 0}}'
            return '{"hourly": []}'

        with mock.patch("tars.brief.run_tool", side_effect=fake_run_tool):
            result = dispatch("/brief")
        self.assertIn("unavailable", result)
        self.assertIn("[weather]", result)


if __name__ == "__main__":
    unittest.main()
