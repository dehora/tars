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
    _handle_brief,
    _handle_review,
    _handle_sessions,
    _handle_slash_search,
    _handle_tidy,
)


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
                                with mock.patch("tars.cli.build_index") as mock_index:
                                    with mock.patch("builtins.print"):
                                        _handle_review("ollama", "test-model")
            # Rules should be written to Procedural.md
            text = (Path(tmpdir) / "Procedural.md").read_text()
            self.assertIn("- route weather queries to weather_now", text)
            self.assertIn("- check memory before adding duplicates", text)
            mock_archive.assert_called_once()
            mock_index.assert_called_once()

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

    def test_review_triggers_reindex(self) -> None:
        corrections = "# Corrections\n\n## 2026-01-01T00:00:00\n- input: x\n- got: y\n"
        model_reply = "- a rule\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("tars.cli.load_feedback", return_value=(corrections, "")):
                with mock.patch("tars.cli.chat", return_value=model_reply):
                    with mock.patch("builtins.input", return_value="y"):
                        with mock.patch("tars.cli._memory_file", return_value=Path(tmpdir) / "Procedural.md"):
                            with mock.patch("tars.cli.archive_feedback"):
                                with mock.patch("tars.cli.build_index") as mock_index:
                                    with mock.patch("builtins.print") as mock_print:
                                        _handle_review("ollama", "test-model")
        mock_index.assert_called_once()
        mock_print.assert_any_call("  index updated")

    def test_review_reindex_failure_warns(self) -> None:
        corrections = "# Corrections\n\n## 2026-01-01T00:00:00\n- input: x\n- got: y\n"
        model_reply = "- a rule\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("tars.cli.load_feedback", return_value=(corrections, "")):
                with mock.patch("tars.cli.chat", return_value=model_reply):
                    with mock.patch("builtins.input", return_value="y"):
                        with mock.patch("tars.cli._memory_file", return_value=Path(tmpdir) / "Procedural.md"):
                            with mock.patch("tars.cli.archive_feedback"):
                                with mock.patch("tars.cli.build_index", side_effect=RuntimeError("no db")):
                                    with mock.patch("builtins.print") as mock_print:
                                        _handle_review("ollama", "test-model")
        # Should warn but not crash
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("reindex failed", output)


class HandleSessionsTests(unittest.TestCase):
    def test_sessions_lists(self) -> None:
        from tars.sessions import SessionInfo
        fake = [
            SessionInfo(path=Path("/a.md"), date="2026-02-20 15:45", title="Weather talk", filename="2026-02-20T15-45-00"),
        ]
        with mock.patch("tars.cli.list_sessions", return_value=fake):
            with mock.patch("builtins.print") as m:
                result = _handle_sessions("/sessions")
        self.assertTrue(result)
        output = " ".join(str(c) for c in m.call_args_list)
        self.assertIn("Weather talk", output)
        self.assertIn("2026-02-20 15:45", output)

    def test_sessions_empty(self) -> None:
        with mock.patch("tars.cli.list_sessions", return_value=[]):
            with mock.patch("builtins.print") as m:
                result = _handle_sessions("/sessions")
        self.assertTrue(result)
        m.assert_any_call("  no sessions found")

    def test_session_search_dispatches(self) -> None:
        from tars.search import SearchResult
        r = SearchResult(
            content="weather chat", score=0.8, file_path="/s.md",
            file_title="S", memory_type="episodic",
            start_line=1, end_line=5, chunk_rowid=1,
        )
        non_episodic = SearchResult(
            content="memory", score=0.9, file_path="/m.md",
            file_title="M", memory_type="semantic",
            start_line=1, end_line=3, chunk_rowid=2,
        )
        with mock.patch("tars.cli.search", return_value=[r, non_episodic]) as m:
            with mock.patch("tars.cli._print_search_results") as pr:
                result = _handle_sessions("/session weather")
        self.assertTrue(result)
        m.assert_called_once_with("weather", mode="hybrid", limit=10)
        # Should only pass episodic results
        passed_results = pr.call_args[0][0]
        self.assertEqual(len(passed_results), 1)
        self.assertEqual(passed_results[0].memory_type, "episodic")

    def test_session_search_no_query(self) -> None:
        with mock.patch("builtins.print") as m:
            result = _handle_sessions("/session")
        self.assertTrue(result)
        m.assert_any_call("  usage: /session <query>")

    def test_unrelated_returns_false(self) -> None:
        result = _handle_sessions("/something")
        self.assertFalse(result)


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


class HandleTidyTests(unittest.TestCase):
    def test_nothing_to_tidy(self) -> None:
        with mock.patch("tars.cli.load_memory_files", return_value={}):
            with mock.patch("builtins.print") as m:
                _handle_tidy("ollama", "test-model")
        m.assert_any_call("  nothing to tidy")

    def test_tidy_applies_removals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "Memory.md"
            p.write_text("# Memory\n- real fact\n- lorem ipsum\n")
            files = {"semantic": p.read_text()}
            model_reply = "- [semantic] lorem ipsum\n"
            with mock.patch("tars.cli.load_memory_files", return_value=files):
                with mock.patch("tars.cli.chat", return_value=model_reply):
                    with mock.patch("builtins.input", return_value="y"):
                        with mock.patch("tars.cli._memory_file", return_value=p):
                            with mock.patch("builtins.print"):
                                _handle_tidy("ollama", "test-model")
            text = p.read_text()
            self.assertIn("- real fact", text)
            self.assertNotIn("- lorem ipsum", text)

    def test_tidy_declined(self) -> None:
        files = {"semantic": "- fact\n- junk\n"}
        model_reply = "- [semantic] junk\n"
        with mock.patch("tars.cli.load_memory_files", return_value=files):
            with mock.patch("tars.cli.chat", return_value=model_reply):
                with mock.patch("builtins.input", return_value="n"):
                    with mock.patch("builtins.print") as m:
                        _handle_tidy("ollama", "test-model")
        m.assert_any_call("  skipped")


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
    def test_brief_formats_all_sections(self) -> None:
        def fake_run_tool(name, args, *, quiet=False):
            if name == "todoist_today":
                return '{"results": []}'
            if name == "weather_now":
                return '{"current": {"temperature_c": 10, "conditions": "Clear", "wind_speed_kmh": 5, "precipitation_mm": 0}}'
            if name == "weather_forecast":
                return '{"hourly": []}'
            return '{}'

        with mock.patch("tars.brief.run_tool", side_effect=fake_run_tool):
            with mock.patch("builtins.print") as m:
                _handle_brief()
        output = " ".join(str(c) for c in m.call_args_list)
        self.assertIn("[tasks]", output)
        self.assertIn("[weather]", output)
        self.assertIn("[forecast]", output)

    def test_brief_handles_tool_failure(self) -> None:
        def fake_run_tool(name, args, *, quiet=False):
            if name == "todoist_today":
                raise FileNotFoundError("td not found")
            if name == "weather_now":
                return '{"current": {"temperature_c": 10, "conditions": "Clear", "wind_speed_kmh": 5, "precipitation_mm": 0}}'
            return '{"hourly": []}'

        with mock.patch("tars.brief.run_tool", side_effect=fake_run_tool):
            with mock.patch("builtins.print") as m:
                _handle_brief()
        output = " ".join(str(c) for c in m.call_args_list)
        self.assertIn("unavailable", output)
        self.assertIn("[weather]", output)


if __name__ == "__main__":
    unittest.main()
