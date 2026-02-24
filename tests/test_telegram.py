"""Tests for the Telegram channel module."""

import os
import sys
import unittest
from unittest import mock

# Ensure ollama mock is in place before importing tars modules
if "ollama" not in sys.modules:
    sys.modules["ollama"] = mock.Mock()

from tars.telegram import (
    _KEYBOARD_ALIASES,
    _handle_slash_command,
    _telegram_config,
    _truncate,
)


class TestTelegramConfig(unittest.TestCase):
    def test_missing_config(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_telegram_config())

    def test_partial_config_token_only(self):
        with mock.patch.dict(
            os.environ, {"TARS_TELEGRAM_TOKEN": "tok123"}, clear=True
        ):
            self.assertIsNone(_telegram_config())

    def test_partial_config_allow_only(self):
        with mock.patch.dict(
            os.environ, {"TARS_TELEGRAM_ALLOW": "12345"}, clear=True
        ):
            self.assertIsNone(_telegram_config())

    def test_full_config(self):
        env = {
            "TARS_TELEGRAM_TOKEN": "7123456789:AAFtest",
            "TARS_TELEGRAM_ALLOW": "111, 222",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = _telegram_config()
            self.assertIsNotNone(cfg)
            self.assertEqual(cfg["token"], "7123456789:AAFtest")
            self.assertEqual(cfg["allow"], [111, 222])

    def test_single_user(self):
        env = {
            "TARS_TELEGRAM_TOKEN": "tok",
            "TARS_TELEGRAM_ALLOW": "42",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = _telegram_config()
            self.assertEqual(cfg["allow"], [42])

    def test_invalid_user_ids_skipped(self):
        env = {
            "TARS_TELEGRAM_TOKEN": "tok",
            "TARS_TELEGRAM_ALLOW": "abc, 123, , xyz",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = _telegram_config()
            self.assertIsNotNone(cfg)
            self.assertEqual(cfg["allow"], [123])

    def test_all_invalid_user_ids(self):
        env = {
            "TARS_TELEGRAM_TOKEN": "tok",
            "TARS_TELEGRAM_ALLOW": "abc, xyz",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(_telegram_config())

    def test_token_whitespace_stripped(self):
        env = {
            "TARS_TELEGRAM_TOKEN": "  tok123  ",
            "TARS_TELEGRAM_ALLOW": "42",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = _telegram_config()
            self.assertEqual(cfg["token"], "tok123")


class TestSlashCommand(unittest.TestCase):
    def test_not_a_command(self):
        self.assertIsNone(_handle_slash_command("just a message"))

    def test_unrecognized_command(self):
        self.assertIsNone(_handle_slash_command("/unknown stuff"))

    @mock.patch("tars.telegram.run_tool", return_value='{"ok": true}')
    def test_weather_command(self, mock_run):
        result = _handle_slash_command("/weather")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with("weather_now", {}, quiet=True)

    @mock.patch("tars.telegram.run_tool", return_value='{"ok": true}')
    def test_todoist_today(self, mock_run):
        result = _handle_slash_command("/todoist today")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with("todoist_today", {}, quiet=True)

    @mock.patch("tars.telegram.run_tool", return_value='{"ok": true}')
    def test_todoist_add(self, mock_run):
        result = _handle_slash_command("/todoist add buy eggs --due tomorrow")
        self.assertIsNotNone(result)
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][0], "todoist_add_task")
        self.assertEqual(call_args[0][1]["content"], "buy eggs")
        self.assertEqual(call_args[0][1]["due"], "tomorrow")

    @mock.patch("tars.telegram.run_tool", return_value='{"ok": true}')
    def test_note_command(self, mock_run):
        result = _handle_slash_command("/note check out that new KEF")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with(
            "note_daily", {"content": "check out that new KEF"}, quiet=True
        )

    @mock.patch("tars.telegram.run_tool", side_effect=Exception("boom"))
    def test_tool_error(self, mock_run):
        result = _handle_slash_command("/weather")
        self.assertIn("Tool error", result)

    def test_brief_command(self):
        with mock.patch("tars.telegram.build_brief_sections") as mock_brief:
            mock_brief.return_value = [("tasks", "task list")]
            with mock.patch("tars.telegram.format_brief_text", return_value="brief text"):
                result = _handle_slash_command("/brief")
                self.assertEqual(result, "brief text")

    @mock.patch("tars.telegram.run_tool", return_value='{"ok": true}')
    def test_bot_username_suffix_stripped(self, mock_run):
        """Commands with @botname suffix should still work."""
        result = _handle_slash_command("/weather@tars_bot")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with("weather_now", {}, quiet=True)


class TestKeyboardAliases(unittest.TestCase):
    def test_all_aliases_are_valid_commands(self):
        """All keyboard aliases should map to valid slash commands or known patterns."""
        for button, command in _KEYBOARD_ALIASES.items():
            self.assertTrue(
                command.startswith("/"),
                f"Alias '{button}' maps to '{command}' which doesn't start with /",
            )

    def test_expected_aliases_present(self):
        for key in ("Brief", "Weather", "Forecast", "Tasks", "Todoist",
                     "Note", "Remember", "Capture", "Search", "Sessions", "Find"):
            self.assertIn(key, _KEYBOARD_ALIASES)

    def test_tasks_maps_to_todoist_today(self):
        self.assertEqual(_KEYBOARD_ALIASES["Tasks"], "/todoist today")


class TestTruncate(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(_truncate("hello"), "hello")

    def test_long_text_truncated(self):
        text = "x" * 5000
        result = _truncate(text, limit=100)
        self.assertLessEqual(len(result), 100)
        self.assertTrue(result.endswith("...(truncated)"))

    def test_exact_limit_unchanged(self):
        text = "x" * 4096
        self.assertEqual(_truncate(text), text)


if __name__ == "__main__":
    unittest.main()
