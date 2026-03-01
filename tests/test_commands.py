"""Tests for the shared command dispatch module."""

import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars.commands import _export_conversation, _format_error, _parse_todoist_add, command_names, dispatch


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


class DispatchTests(unittest.TestCase):
    def test_not_a_command(self) -> None:
        self.assertIsNone(dispatch("just a message"))

    def test_unrecognized_command(self) -> None:
        result = dispatch("/unknown stuff")
        self.assertIsNotNone(result)
        self.assertIn("Unknown command", result)

    @mock.patch("tars.commands.run_tool", return_value='{"ok": true}')
    def test_weather(self, mock_run) -> None:
        result = dispatch("/weather")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with("weather_now", {}, quiet=True)

    @mock.patch("tars.commands.run_tool", return_value='{"ok": true}')
    def test_forecast(self, mock_run) -> None:
        result = dispatch("/forecast")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with("weather_forecast", {}, quiet=True)

    @mock.patch("tars.commands.run_tool", return_value='{"ok": true}')
    def test_todoist_today(self, mock_run) -> None:
        result = dispatch("/todoist today")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with("todoist_today", {}, quiet=True)

    @mock.patch("tars.commands.run_tool", return_value='{"ok": true}')
    def test_todoist_upcoming(self, mock_run) -> None:
        result = dispatch("/todoist upcoming")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with("todoist_upcoming", {"days": 7}, quiet=True)

    @mock.patch("tars.commands.run_tool", return_value='{"ok": true}')
    def test_todoist_upcoming_with_days(self, mock_run) -> None:
        result = dispatch("/todoist upcoming 5")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with("todoist_upcoming", {"days": 5}, quiet=True)

    def test_todoist_upcoming_bad_days(self) -> None:
        result = dispatch("/todoist upcoming abc")
        self.assertIn("Usage", result)

    @mock.patch("tars.commands.run_tool", return_value='{"ok": true}')
    def test_todoist_complete(self, mock_run) -> None:
        result = dispatch("/todoist complete buy eggs")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with(
            "todoist_complete_task", {"ref": "buy eggs"}, quiet=True
        )

    @mock.patch("tars.commands.run_tool", return_value='{"ok": true}')
    def test_todoist_add_with_flags(self, mock_run) -> None:
        result = dispatch("/todoist add buy eggs --due tomorrow")
        self.assertIsNotNone(result)
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][0], "todoist_add_task")
        self.assertEqual(call_args[0][1]["content"], "buy eggs")
        self.assertEqual(call_args[0][1]["due"], "tomorrow")

    @mock.patch("tars.commands.run_tool", return_value='{"ok": true}')
    @mock.patch(
        "tars.commands._parse_todoist_natural",
        return_value={"content": "eggs", "project": "Groceries"},
    )
    def test_todoist_add_natural_language(self, mock_parse, mock_run) -> None:
        result = dispatch("/todoist add eggs to Groceries", "ollama", "llama3.1:8b")
        self.assertIsNotNone(result)
        mock_parse.assert_called_once_with("eggs to Groceries", "ollama", "llama3.1:8b")
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][1]["content"], "eggs")
        self.assertEqual(call_args[0][1]["project"], "Groceries")

    @mock.patch("tars.commands.run_tool", return_value='{"ok": true}')
    def test_todoist_add_flags_bypass_model(self, mock_run) -> None:
        result = dispatch(
            "/todoist add buy eggs --project Groceries", "ollama", "llama3.1:8b"
        )
        self.assertIsNotNone(result)
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][1]["content"], "buy eggs")
        self.assertEqual(call_args[0][1]["project"], "Groceries")

    def test_todoist_add_no_content(self) -> None:
        result = dispatch("/todoist add")
        self.assertIn("Usage", result)

    @mock.patch("tars.commands.run_tool", return_value='{"ok": true}')
    def test_note(self, mock_run) -> None:
        result = dispatch("/note interesting idea")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with(
            "note_daily", {"content": "interesting idea"}, quiet=True
        )

    def test_note_no_content(self) -> None:
        result = dispatch("/note")
        self.assertIn("Usage", result)

    @mock.patch("tars.commands.run_tool", return_value='{"ok": true}')
    def test_remember(self, mock_run) -> None:
        result = dispatch("/remember semantic important fact")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with(
            "memory_remember",
            {"section": "semantic", "content": "important fact"},
            quiet=True,
        )

    def test_remember_no_content(self) -> None:
        result = dispatch("/remember")
        self.assertIn("Usage", result)

    @mock.patch("tars.commands.run_tool", return_value='{"ok": true}')
    def test_memory(self, mock_run) -> None:
        result = dispatch("/memory")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with("memory_recall", {}, quiet=True)

    @mock.patch("tars.commands.run_tool", return_value='{"url": "https://example.com", "content": "hello"}')
    def test_read(self, mock_run) -> None:
        result = dispatch("/read https://example.com")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with(
            "web_read", {"url": "https://example.com"}, quiet=True
        )

    def test_read_no_url(self) -> None:
        result = dispatch("/read")
        self.assertIn("Usage", result)

    @mock.patch("tars.commands._dispatch_capture")
    def test_capture(self, mock_cap) -> None:
        mock_cap.return_value = "Captured: test"
        result = dispatch("/capture https://example.com", "ollama", "llama3.1:8b")
        self.assertEqual(result, "Captured: test")

    def test_capture_no_url(self) -> None:
        result = dispatch("/capture")
        self.assertIn("Usage", result)

    @mock.patch("tars.commands._dispatch_capture")
    def test_capture_raw_flag(self, mock_cap) -> None:
        mock_cap.return_value = "Raw captured"
        dispatch("/capture https://example.com --raw", "ollama", "llama3.1:8b")
        mock_cap.assert_called_once_with(
            ["/capture", "https://example.com", "--raw"], "ollama", "llama3.1:8b"
        )

    @mock.patch("tars.brief.build_brief_sections", return_value=[("tasks", "list")])
    @mock.patch("tars.brief.format_brief_text", return_value="brief output")
    def test_brief(self, mock_fmt, mock_sections) -> None:
        result = dispatch("/brief")
        self.assertEqual(result, "brief output")

    @mock.patch("tars.commands._dispatch_search", return_value="1. result")
    def test_search(self, mock_search) -> None:
        result = dispatch("/search weather")
        self.assertEqual(result, "1. result")

    def test_search_no_query(self) -> None:
        result = dispatch("/search")
        self.assertIn("Usage", result)

    @mock.patch("tars.commands._dispatch_find", return_value="1. note result")
    def test_find(self, mock_find) -> None:
        result = dispatch("/find weather")
        self.assertEqual(result, "1. note result")

    def test_find_no_query(self) -> None:
        result = dispatch("/find")
        self.assertIn("Usage", result)

    @mock.patch("tars.commands._dispatch_sessions", return_value="2026-02-20  Weather talk")
    def test_sessions(self, mock_sessions) -> None:
        result = dispatch("/sessions")
        self.assertIn("Weather talk", result)

    @mock.patch("tars.commands.run_tool", side_effect=Exception("boom"))
    def test_tool_error(self, mock_run) -> None:
        result = dispatch("/weather")
        self.assertIn("Error: boom", result)
        self.assertNotIn("Tool error", result)


class ExportTests(unittest.TestCase):
    def test_export_with_messages(self) -> None:
        from tars.conversation import Conversation

        conv = Conversation(id="test-1", provider="ollama", model="test")
        conv.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = _export_conversation(conv)
        self.assertIn("# Conversation test-1", result)
        self.assertIn("## user", result)
        self.assertIn("hello", result)
        self.assertIn("## assistant", result)
        self.assertIn("hi there", result)

    def test_export_no_conversation(self) -> None:
        result = _export_conversation(None)
        self.assertEqual(result, "No conversation to export.")

    def test_export_empty_conversation(self) -> None:
        from tars.conversation import Conversation

        conv = Conversation(id="test-2", provider="ollama", model="test")
        result = _export_conversation(conv)
        self.assertEqual(result, "No conversation to export.")


class CentralizedDispatchTests(unittest.TestCase):
    def test_dispatch_channel_guard_cli_only(self) -> None:
        result = dispatch("/review", context={"channel": "telegram"})
        self.assertIn("CLI", result)

    def test_dispatch_channel_guard_interactive(self) -> None:
        result = dispatch("/w", context={"channel": "telegram"})
        self.assertIn("CLI", result)

    def test_dispatch_unknown_slash_returns_error(self) -> None:
        result = dispatch("/notacommand")
        self.assertIsNotNone(result)
        self.assertIn("Unknown command: /notacommand", result)
        self.assertIn("/help", result)

    def test_dispatch_question_mark(self) -> None:
        result = dispatch("?")
        self.assertIsNotNone(result)
        self.assertIn("/todoist", result)
        self.assertIn("/help", result)

    @mock.patch("tars.commands._dispatch_stats", return_value="db: 1 MB")
    def test_dispatch_stats(self, mock_stats) -> None:
        result = dispatch("/stats")
        self.assertEqual(result, "db: 1 MB")

    def test_dispatch_model_with_config(self) -> None:
        from tars.config import ModelConfig
        config = ModelConfig(
            primary_provider="claude",
            primary_model="sonnet",
            remote_provider=None,
            remote_model=None,
        )
        result = dispatch("/model", context={"channel": "cli", "config": config})
        self.assertIn("primary: claude:sonnet", result)
        self.assertIn("remote: none", result)

    def test_dispatch_model_no_config(self) -> None:
        result = dispatch("/model", context={"channel": "cli"})
        self.assertIn("no model config", result)

    @mock.patch("tars.commands._dispatch_search", return_value="1. result")
    def test_dispatch_sgrep(self, mock_search) -> None:
        result = dispatch("/sgrep test query")
        self.assertEqual(result, "1. result")
        mock_search.assert_called_once_with("test query", mode="fts")

    @mock.patch("tars.commands._dispatch_search", return_value="1. result")
    def test_dispatch_svec(self, mock_search) -> None:
        result = dispatch("/svec test query")
        self.assertEqual(result, "1. result")
        mock_search.assert_called_once_with("test query", mode="vec")

    def test_dispatch_sgrep_no_query(self) -> None:
        result = dispatch("/sgrep")
        self.assertIn("Usage", result)

    def test_dispatch_svec_no_query(self) -> None:
        result = dispatch("/svec")
        self.assertIn("Usage", result)

    def test_dispatch_help(self) -> None:
        result = dispatch("/help")
        self.assertIn("/todoist", result)
        self.assertIn("/weather", result)
        self.assertIn("/search", result)
        self.assertIn("/help", result)

    def test_dispatch_clear(self) -> None:
        result = dispatch("/clear")
        self.assertEqual(result, "__clear__")

    @mock.patch("tars.commands._dispatch_session_search", return_value="1. session result")
    def test_dispatch_session_search(self, mock_search) -> None:
        result = dispatch("/session test")
        self.assertEqual(result, "1. session result")

    def test_dispatch_session_no_query(self) -> None:
        result = dispatch("/session")
        self.assertIn("Usage", result)

    @mock.patch("tars.commands._dispatch_schedule", return_value="no schedules installed")
    def test_dispatch_schedule(self, mock_sched) -> None:
        result = dispatch("/schedule")
        self.assertEqual(result, "no schedules installed")

    def test_command_names_complete(self) -> None:
        names = command_names()
        expected = {
            "/todoist", "/weather", "/forecast", "/memory", "/remember", "/note",
            "/read", "/capture", "/brief",
            "/search", "/sgrep", "/svec", "/find",
            "/sessions", "/session",
            "/w", "/r", "/review", "/tidy",
            "/mcp", "/stats", "/schedule", "/model",
            "/export", "/help", "/clear",
        }
        self.assertEqual(names, expected)

    def test_dispatch_feedback_w(self) -> None:
        from tars.conversation import Conversation
        conv = Conversation(id="test", provider="ollama", model="test")
        conv.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        with mock.patch("tars.commands.save_correction", return_value="feedback saved") as mock_save:
            result = dispatch("/w bad answer", conv=conv, context={"channel": "cli"})
        self.assertEqual(result, "feedback saved")
        mock_save.assert_called_once_with("hello", "hi there", "bad answer")

    def test_dispatch_feedback_no_messages(self) -> None:
        from tars.conversation import Conversation
        conv = Conversation(id="test", provider="ollama", model="test")
        result = dispatch("/w", conv=conv, context={"channel": "cli"})
        self.assertIn("nothing to flag", result)


class FormatErrorTests(unittest.TestCase):
    def test_connection_error(self) -> None:
        result = _format_error(ConnectionError("refused"))
        self.assertIn("Network error", result)

    def test_timeout_error(self) -> None:
        result = _format_error(TimeoutError("timed out"))
        self.assertIn("Network error", result)

    def test_os_error_network(self) -> None:
        result = _format_error(OSError("network unreachable"))
        self.assertIn("Network error", result)

    def test_file_not_found_not_network(self) -> None:
        result = _format_error(FileNotFoundError("missing.txt"))
        self.assertNotIn("Network error", result)
        self.assertIn("Error:", result)

    def test_api_key_error(self) -> None:
        result = _format_error(Exception("ANTHROPIC_API_KEY not set"))
        self.assertIn("Auth error", result)

    def test_api_key_error_lowercase(self) -> None:
        result = _format_error(Exception("Invalid api key provided"))
        self.assertIn("Auth error", result)

    def test_memory_dir_error(self) -> None:
        result = _format_error(Exception("TARS_MEMORY_DIR not configured"))
        self.assertIn("Memory not configured", result)

    def test_generic_error(self) -> None:
        result = _format_error(Exception("something broke"))
        self.assertEqual(result, "Error: something broke")


class SessionsDisplayTests(unittest.TestCase):
    def test_sessions_with_channel(self) -> None:
        from tars.sessions import SessionInfo
        from pathlib import Path

        mock_sessions = [
            SessionInfo(
                path=Path("/tmp/s1.md"), date="2026-03-01 10:00",
                title="Weather talk", filename="2026-03-01T10-00-00-cli",
                channel="cli",
            ),
            SessionInfo(
                path=Path("/tmp/s2.md"), date="2026-02-28 15:00",
                title="Web chat", filename="2026-02-28T15-00-00-web",
                channel="web",
            ),
        ]
        with mock.patch("tars.sessions.list_sessions", return_value=mock_sessions):
            from tars.commands import _dispatch_sessions
            result = _dispatch_sessions()
        self.assertIn("[cli]", result)
        self.assertIn("[web]", result)
        self.assertIn("Weather talk", result)

    def test_sessions_without_channel(self) -> None:
        from tars.sessions import SessionInfo
        from pathlib import Path

        mock_sessions = [
            SessionInfo(
                path=Path("/tmp/s1.md"), date="2026-03-01 10:00",
                title="Old session", filename="2026-03-01T10-00-00",
                channel="",
            ),
        ]
        with mock.patch("tars.sessions.list_sessions", return_value=mock_sessions):
            from tars.commands import _dispatch_sessions
            result = _dispatch_sessions()
        self.assertNotIn("[", result)
        self.assertIn("Old session", result)


if __name__ == "__main__":
    unittest.main()
