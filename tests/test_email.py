"""Tests for the email channel module."""

import email
import os
import sys
import unittest
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest import mock

# Ensure ollama mock is in place before importing tars modules
if "ollama" not in sys.modules:
    sys.modules["ollama"] = mock.Mock()

from tars.email import (
    _email_config,
    _extract_body,
    _handle_slash_command,
    _is_allowed_sender,
    _strip_html,
    _thread_id,
)


class TestExtractBody(unittest.TestCase):
    def test_plain_text(self):
        msg = MIMEText("Hello, this is a test.", "plain", "utf-8")
        self.assertEqual(_extract_body(msg), "Hello, this is a test.")

    def test_multipart_prefers_plain(self):
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText("<b>HTML body</b>", "html", "utf-8"))
        msg.attach(MIMEText("Plain body", "plain", "utf-8"))
        self.assertEqual(_extract_body(msg), "Plain body")

    def test_multipart_fallback_html(self):
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText("<b>HTML only</b>", "html", "utf-8"))
        self.assertEqual(_extract_body(msg), "HTML only")

    def test_strips_quoted_replies(self):
        body = "My new message\n> Previous reply\n> More quoted\nAnother line"
        msg = MIMEText(body, "plain", "utf-8")
        result = _extract_body(msg)
        self.assertEqual(result, "My new message\nAnother line")

    def test_skips_attachment_parts(self):
        msg = MIMEMultipart("mixed")
        # Attachment text/plain part
        attachment = MIMEText("attachment content", "plain", "utf-8")
        attachment.add_header("Content-Disposition", "attachment", filename="data.txt")
        msg.attach(attachment)
        # Actual body
        msg.attach(MIMEText("real body", "plain", "utf-8"))
        self.assertEqual(_extract_body(msg), "real body")

    def test_empty_body(self):
        msg = MIMEText("", "plain", "utf-8")
        self.assertEqual(_extract_body(msg), "")


class TestStripHtml(unittest.TestCase):
    def test_removes_tags(self):
        self.assertEqual(_strip_html("<b>bold</b> text"), "bold text")

    def test_br_to_newline(self):
        self.assertEqual(_strip_html("line1<br>line2"), "line1\nline2")
        self.assertEqual(_strip_html("line1<br/>line2"), "line1\nline2")


class TestThreadId(unittest.TestCase):
    def test_from_references(self):
        msg = MIMEText("test", "plain")
        msg["Message-ID"] = "<msg2@example.com>"
        msg["References"] = "<root@example.com> <msg1@example.com>"
        self.assertEqual(_thread_id(msg), "<root@example.com>")

    def test_new_message(self):
        msg = MIMEText("test", "plain")
        msg["Message-ID"] = "<new@example.com>"
        self.assertEqual(_thread_id(msg), "<new@example.com>")

    def test_single_reference(self):
        msg = MIMEText("test", "plain")
        msg["Message-ID"] = "<msg2@example.com>"
        msg["References"] = "<root@example.com>"
        self.assertEqual(_thread_id(msg), "<root@example.com>")


class TestSenderFilter(unittest.TestCase):
    def test_allowed(self):
        msg = MIMEText("test", "plain")
        msg["From"] = "Bill <bill@example.com>"
        self.assertTrue(_is_allowed_sender(msg, ["bill@example.com"]))

    def test_blocked(self):
        msg = MIMEText("test", "plain")
        msg["From"] = "Spam <spam@evil.com>"
        self.assertFalse(_is_allowed_sender(msg, ["bill@example.com"]))

    def test_case_insensitive(self):
        msg = MIMEText("test", "plain")
        msg["From"] = "Bill <Bill@Example.COM>"
        self.assertTrue(_is_allowed_sender(msg, ["bill@example.com"]))

    def test_bare_address(self):
        msg = MIMEText("test", "plain")
        msg["From"] = "bill@example.com"
        self.assertTrue(_is_allowed_sender(msg, ["bill@example.com"]))


class TestSendReplyHeaders(unittest.TestCase):
    """Test that _send_reply constructs correct headers (without actually sending)."""

    @mock.patch("tars.email.smtplib.SMTP")
    def test_reply_headers(self, mock_smtp_class):
        from tars.email import _send_reply

        mock_smtp = mock.MagicMock()
        mock_smtp_class.return_value.__enter__ = mock.Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = mock.Mock(return_value=False)

        original = MIMEText("question", "plain")
        original["From"] = "bill@example.com"
        original["Message-ID"] = "<orig123@example.com>"
        original["Subject"] = "Hello tars"

        config = {
            "address": "tars@gmail.com",
            "password": "fake-password",
        }

        _send_reply(config, original, "Here is my reply")

        # Verify send_message was called
        mock_smtp.send_message.assert_called_once()
        sent = mock_smtp.send_message.call_args[0][0]
        self.assertEqual(sent["In-Reply-To"], "<orig123@example.com>")
        self.assertEqual(sent["References"], "<orig123@example.com>")
        self.assertEqual(sent["Subject"], "Re: Hello tars")
        self.assertEqual(sent["To"], "bill@example.com")

    @mock.patch("tars.email.smtplib.SMTP")
    def test_reply_preserves_re_subject(self, mock_smtp_class):
        from tars.email import _send_reply

        mock_smtp = mock.MagicMock()
        mock_smtp_class.return_value.__enter__ = mock.Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = mock.Mock(return_value=False)

        original = MIMEText("follow up", "plain")
        original["From"] = "bill@example.com"
        original["Message-ID"] = "<orig456@example.com>"
        original["Subject"] = "Re: Hello tars"
        original["References"] = "<root@example.com>"

        config = {
            "address": "tars@gmail.com",
            "password": "fake-password",
        }

        _send_reply(config, original, "Reply again")

        sent = mock_smtp.send_message.call_args[0][0]
        self.assertEqual(sent["Subject"], "Re: Hello tars")
        self.assertEqual(
            sent["References"], "<root@example.com> <orig456@example.com>"
        )


class TestEmailConfig(unittest.TestCase):
    def test_missing_config(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_email_config())

    def test_partial_config(self):
        with mock.patch.dict(
            os.environ, {"TARS_EMAIL_ADDRESS": "x@y.com"}, clear=True
        ):
            self.assertIsNone(_email_config())

    def test_full_config(self):
        env = {
            "TARS_EMAIL_ADDRESS": "tars@gmail.com",
            "TARS_EMAIL_PASSWORD": "secret",
            "TARS_EMAIL_ALLOW": "a@b.com, c@d.com",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = _email_config()
            self.assertIsNotNone(cfg)
            self.assertEqual(cfg["address"], "tars@gmail.com")
            self.assertEqual(cfg["allow"], ["a@b.com", "c@d.com"])
            self.assertEqual(cfg["poll_interval"], 60)

    def test_custom_interval(self):
        env = {
            "TARS_EMAIL_ADDRESS": "tars@gmail.com",
            "TARS_EMAIL_PASSWORD": "secret",
            "TARS_EMAIL_ALLOW": "a@b.com",
            "TARS_EMAIL_POLL_INTERVAL": "30",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = _email_config()
            self.assertEqual(cfg["poll_interval"], 30)

    def test_invalid_interval_defaults(self):
        env = {
            "TARS_EMAIL_ADDRESS": "tars@gmail.com",
            "TARS_EMAIL_PASSWORD": "secret",
            "TARS_EMAIL_ALLOW": "a@b.com",
            "TARS_EMAIL_POLL_INTERVAL": "bad",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = _email_config()
            self.assertEqual(cfg["poll_interval"], 60)


class TestSlashCommand(unittest.TestCase):
    def test_not_a_command(self):
        self.assertIsNone(_handle_slash_command("just a message"))

    def test_unrecognized_command(self):
        self.assertIsNone(_handle_slash_command("/unknown stuff"))

    @mock.patch("tars.email.run_tool", return_value='{"ok": true}')
    def test_weather_command(self, mock_run):
        result = _handle_slash_command("/weather")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with("weather_now", {}, quiet=True)

    @mock.patch("tars.email.run_tool", return_value='{"ok": true}')
    def test_todoist_today(self, mock_run):
        result = _handle_slash_command("/todoist today")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with("todoist_today", {}, quiet=True)

    @mock.patch("tars.email.run_tool", return_value='{"ok": true}')
    def test_todoist_add(self, mock_run):
        result = _handle_slash_command("/todoist add buy eggs --due tomorrow")
        self.assertIsNotNone(result)
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][0], "todoist_add_task")
        self.assertEqual(call_args[0][1]["content"], "buy eggs")
        self.assertEqual(call_args[0][1]["due"], "tomorrow")

    @mock.patch("tars.email.run_tool", return_value='{"ok": true}')
    def test_note_command(self, mock_run):
        result = _handle_slash_command("/note check out that new KEF")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with(
            "note_daily", {"content": "check out that new KEF"}, quiet=True
        )

    @mock.patch("tars.email.run_tool", return_value='{"url": "https://example.com", "content": "hello", "truncated": false}')
    def test_read_command(self, mock_run):
        result = _handle_slash_command("/read https://example.com")
        self.assertIsNotNone(result)
        mock_run.assert_called_once_with(
            "web_read", {"url": "https://example.com"}, quiet=True,
        )

    def test_read_no_url(self):
        self.assertIsNone(_handle_slash_command("/read"))

    @mock.patch("tars.email.run_tool", return_value='{"ok": true}')
    @mock.patch("tars.email._parse_todoist_natural", return_value={"content": "eggs", "project": "Groceries"})
    def test_todoist_add_natural_language(self, mock_parse, mock_run):
        result = _handle_slash_command("/todoist add eggs to Groceries", "ollama", "llama3.1:8b")
        self.assertIsNotNone(result)
        mock_parse.assert_called_once_with("eggs to Groceries", "ollama", "llama3.1:8b")
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][1]["content"], "eggs")
        self.assertEqual(call_args[0][1]["project"], "Groceries")

    @mock.patch("tars.email.run_tool", return_value='{"ok": true}')
    def test_todoist_add_flags_bypass_model(self, mock_run):
        """Flags in the text should use the flag parser, not the model."""
        result = _handle_slash_command("/todoist add buy eggs --project Groceries", "ollama", "llama3.1:8b")
        self.assertIsNotNone(result)
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][1]["content"], "buy eggs")
        self.assertEqual(call_args[0][1]["project"], "Groceries")

    def test_todoist_add_no_content(self):
        result = _handle_slash_command("/todoist add")
        self.assertIsNotNone(result)
        self.assertIn("Usage", result)

    @mock.patch("tars.email.run_tool", side_effect=Exception("boom"))
    def test_tool_error(self, mock_run):
        result = _handle_slash_command("/weather")
        self.assertIn("Tool error", result)


if __name__ == "__main__":
    unittest.main()
