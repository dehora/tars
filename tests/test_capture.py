import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars.capture import _conversation_context, _extract_title, _sanitize_filename, capture


class ExtractTitleTests(unittest.TestCase):
    def test_from_heading(self) -> None:
        self.assertEqual(_extract_title("# My Post\n\nBody text", "https://dehora.net/tars-test"), "My Post")

    def test_skips_subheadings(self) -> None:
        self.assertEqual(_extract_title("## Not this\n\nBody", "https://dehora.net/tars-test/my-post"), "My Post")

    def test_from_url_slug(self) -> None:
        self.assertEqual(_extract_title("no headings here", "https://dehora.net/tars-test/cool-article"), "Cool Article")

    def test_url_no_path(self) -> None:
        self.assertEqual(_extract_title("text", "https://dehora.net/"), "Untitled")


class SanitizeFilenameTests(unittest.TestCase):
    def test_removes_unsafe_chars(self) -> None:
        self.assertEqual(_sanitize_filename('foo: bar "baz"'), "foo bar baz")

    def test_collapses_whitespace(self) -> None:
        self.assertEqual(_sanitize_filename("  lots   of  spaces  "), "lots of spaces")

    def test_truncates_long_titles(self) -> None:
        long = "a" * 200
        self.assertLessEqual(len(_sanitize_filename(long)), 120)


class CaptureTests(unittest.TestCase):
    def test_no_notes_dir(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            result = json.loads(capture("https://dehora.net/tars-test", "ollama", "fake"))
        self.assertIn("error", result)
        self.assertIn("TARS_NOTES_DIR", result["error"])

    def test_empty_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(capture("", "ollama", "fake"))
        self.assertIn("error", result)

    def test_fetch_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.dict(os.environ, {"TARS_NOTES_DIR": tmpdir}),
                mock.patch("tars.capture._run_web_tool", return_value=json.dumps({"error": "fetch failed"})),
            ):
                result = json.loads(capture("https://dehora.net/tars-test", "ollama", "fake"))
        self.assertIn("error", result)

    def test_capture_saves_file(self) -> None:
        web_result = json.dumps({"url": "https://dehora.net/tars-test/post", "content": "Some article text", "truncated": False})
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.dict(os.environ, {"TARS_NOTES_DIR": tmpdir}),
                mock.patch("tars.capture._run_web_tool", return_value=web_result),
                mock.patch(
                    "tars.capture.chat",
                    side_effect=[
                        '{"title":"Great Post","author":"Bill","created":"2026-02-01","description":"TLDR here."}',
                        "# Great Post\n\nSummarized content.",
                    ],
                ),
            ):
                result = json.loads(capture("https://dehora.net/tars-test/post", "ollama", "fake"))
            self.assertTrue(result["ok"])
            self.assertEqual(result["title"], "Great Post")
            path = Path(result["path"])
            self.assertTrue(path.exists())
            content = path.read_text(encoding="utf-8")
            self.assertIn('title: "Great Post"', content)
            self.assertIn('author: "Bill"', content)
            self.assertIn('created: "2026-02-01"', content)
            self.assertIn("captured:", content)
            self.assertIn('description: "TLDR here."', content)
            self.assertIn("tags:", content)
            self.assertIn("  - capture", content)
            self.assertIn("  - tars", content)
            self.assertIn('source: "https://dehora.net/tars-test/post"', content)
            self.assertIn("Summarized content.", content)
            # Check it's in the right directory
            self.assertIn("17 tars captures", str(path))

    def test_description_falls_back_to_body(self) -> None:
        web_result = json.dumps({"url": "https://dehora.net/tars-test/post", "content": "Some article text", "truncated": False})
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.dict(os.environ, {"TARS_NOTES_DIR": tmpdir}),
                mock.patch("tars.capture._run_web_tool", return_value=web_result),
                mock.patch(
                    "tars.capture.chat",
                    side_effect=[
                        '{"title":"Great Post","author":"","created":"","description":""}',
                        "Here is the cleaned markdown article content:\n\n# Great Post\n\nThis is the first paragraph.\nStill first paragraph.\n\nSecond paragraph.",
                    ],
                ),
            ):
                result = json.loads(capture("https://dehora.net/tars-test/post", "ollama", "fake"))
            self.assertTrue(result["ok"])
            content = Path(result["path"]).read_text(encoding="utf-8")
            self.assertIn('description: "This is the first paragraph. Still first paragraph."', content)

    def test_capture_raw_skips_summary(self) -> None:
        web_result = json.dumps({"url": "https://dehora.net/tars-test/post", "content": "# Raw Title\n\nRaw content here.", "truncated": False})
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.dict(os.environ, {"TARS_NOTES_DIR": tmpdir}),
                mock.patch("tars.capture._run_web_tool", return_value=web_result),
                mock.patch("tars.capture.chat", return_value='{"title":"Raw Title","author":"","created":"","description":"Raw TLDR"}') as mock_chat,
            ):
                result = json.loads(capture("https://dehora.net/tars-test/post", "ollama", "fake", raw=True))
            self.assertEqual(mock_chat.call_count, 1)
            self.assertTrue(result["ok"])
            path = Path(result["path"])
            content = path.read_text(encoding="utf-8")
            self.assertIn("Raw content here.", content)

    def test_capture_creates_directory(self) -> None:
        web_result = json.dumps({"url": "https://dehora.net/tars-test", "content": "text", "truncated": False})
        with tempfile.TemporaryDirectory() as tmpdir:
            captures_dir = Path(tmpdir) / "17 tars captures"
            self.assertFalse(captures_dir.exists())
            with (
                mock.patch.dict(os.environ, {"TARS_NOTES_DIR": tmpdir}),
                mock.patch("tars.capture._run_web_tool", return_value=web_result),
                mock.patch("tars.capture.chat", return_value="# Title\n\nBody"),
            ):
                capture("https://dehora.net/tars-test", "ollama", "fake")
            self.assertTrue(captures_dir.exists())

    def test_chat_called_with_use_tools_false(self) -> None:
        web_result = json.dumps({"url": "https://dehora.net/tars-test", "content": "text", "truncated": False})
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.dict(os.environ, {"TARS_NOTES_DIR": tmpdir}),
                mock.patch("tars.capture._run_web_tool", return_value=web_result),
                mock.patch(
                    "tars.capture.chat",
                    side_effect=[
                        '{"title":"T","author":"","created":"","description":""}',
                        "# T\n\nBody",
                    ],
                ) as mock_chat,
            ):
                capture("https://dehora.net/tars-test", "ollama", "fake")
            for _, kwargs in mock_chat.call_args_list:
                self.assertFalse(kwargs["use_tools"])


class ConversationContextTests(unittest.TestCase):
    def test_extracts_recent_messages(self) -> None:
        conv = mock.Mock()
        conv.messages = [
            {"role": "user", "content": "what is AI routing?"},
            {"role": "assistant", "content": "AI routing is..."},
        ]
        ctx = _conversation_context(conv)
        self.assertIn("user: what is AI routing?", ctx)
        self.assertIn("assistant: AI routing is...", ctx)

    def test_truncates_long_messages(self) -> None:
        conv = mock.Mock()
        conv.messages = [{"role": "user", "content": "x" * 300}]
        ctx = _conversation_context(conv)
        self.assertLessEqual(len(ctx.splitlines()[0]), 210)  # "user: " + 200 chars

    def test_empty_conversation(self) -> None:
        conv = mock.Mock()
        conv.messages = []
        self.assertEqual(_conversation_context(conv), "")

    def test_none_conversation(self) -> None:
        self.assertEqual(_conversation_context(None), "")

    def test_limits_to_last_six(self) -> None:
        conv = mock.Mock()
        conv.messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        ctx = _conversation_context(conv)
        lines = ctx.strip().splitlines()
        self.assertEqual(len(lines), 6)
        self.assertIn("msg 4", lines[0])


class CaptureWithContextTests(unittest.TestCase):
    def test_context_included_in_prompt(self) -> None:
        web_result = json.dumps({"url": "https://dehora.net/tars-test", "content": "article text", "truncated": False})
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.dict(os.environ, {"TARS_NOTES_DIR": tmpdir}),
                mock.patch("tars.capture._run_web_tool", return_value=web_result),
                mock.patch(
                    "tars.capture.chat",
                    side_effect=[
                        '{"title":"T","author":"","created":"","description":""}',
                        "# T\n\nBody",
                    ],
                ) as mock_chat,
            ):
                capture("https://dehora.net/tars-test", "ollama", "fake", context="user: what about routing?")
            prompt = mock_chat.call_args_list[1][0][0][0]["content"]
            self.assertIn("what about routing?", prompt)
            self.assertIn("relevant to this context", prompt)

    def test_no_context_omits_block(self) -> None:
        web_result = json.dumps({"url": "https://dehora.net/tars-test", "content": "article text", "truncated": False})
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.dict(os.environ, {"TARS_NOTES_DIR": tmpdir}),
                mock.patch("tars.capture._run_web_tool", return_value=web_result),
                mock.patch(
                    "tars.capture.chat",
                    side_effect=[
                        '{"title":"T","author":"","created":"","description":""}',
                        "# T\n\nBody",
                    ],
                ) as mock_chat,
            ):
                capture("https://dehora.net/tars-test", "ollama", "fake")
            prompt = mock_chat.call_args_list[1][0][0][0]["content"]
            self.assertNotIn("relevant to this context", prompt)


if __name__ == "__main__":
    unittest.main()
