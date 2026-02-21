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

from tars.capture import _extract_title, _sanitize_filename, capture


class ExtractTitleTests(unittest.TestCase):
    def test_from_heading(self) -> None:
        self.assertEqual(_extract_title("# My Post\n\nBody text", "https://x.com"), "My Post")

    def test_skips_subheadings(self) -> None:
        self.assertEqual(_extract_title("## Not this\n\nBody", "https://x.com/my-post"), "My Post")

    def test_from_url_slug(self) -> None:
        self.assertEqual(_extract_title("no headings here", "https://x.com/cool-article"), "Cool Article")

    def test_url_no_path(self) -> None:
        self.assertEqual(_extract_title("text", "https://x.com/"), "Untitled")


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
            result = json.loads(capture("https://x.com", "ollama", "fake"))
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
                result = json.loads(capture("https://x.com", "ollama", "fake"))
        self.assertIn("error", result)

    def test_capture_saves_file(self) -> None:
        web_result = json.dumps({"url": "https://x.com/post", "content": "Some article text", "truncated": False})
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.dict(os.environ, {"TARS_NOTES_DIR": tmpdir}),
                mock.patch("tars.capture._run_web_tool", return_value=web_result),
                mock.patch("tars.capture.chat", return_value="# Great Post\n\nSummarized content."),
            ):
                result = json.loads(capture("https://x.com/post", "ollama", "fake"))
            self.assertTrue(result["ok"])
            self.assertEqual(result["title"], "Great Post")
            path = Path(result["path"])
            self.assertTrue(path.exists())
            content = path.read_text(encoding="utf-8")
            self.assertIn("source: https://x.com/post", content)
            self.assertIn("captured:", content)
            self.assertIn("Summarized content.", content)
            # Check it's in the right directory
            self.assertIn("17 tars captures", str(path))

    def test_capture_raw_skips_summary(self) -> None:
        web_result = json.dumps({"url": "https://x.com/post", "content": "# Raw Title\n\nRaw content here.", "truncated": False})
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.dict(os.environ, {"TARS_NOTES_DIR": tmpdir}),
                mock.patch("tars.capture._run_web_tool", return_value=web_result),
                mock.patch("tars.capture.chat") as mock_chat,
            ):
                result = json.loads(capture("https://x.com/post", "ollama", "fake", raw=True))
            mock_chat.assert_not_called()
            self.assertTrue(result["ok"])
            path = Path(result["path"])
            content = path.read_text(encoding="utf-8")
            self.assertIn("Raw content here.", content)

    def test_capture_creates_directory(self) -> None:
        web_result = json.dumps({"url": "https://x.com", "content": "text", "truncated": False})
        with tempfile.TemporaryDirectory() as tmpdir:
            captures_dir = Path(tmpdir) / "17 tars captures"
            self.assertFalse(captures_dir.exists())
            with (
                mock.patch.dict(os.environ, {"TARS_NOTES_DIR": tmpdir}),
                mock.patch("tars.capture._run_web_tool", return_value=web_result),
                mock.patch("tars.capture.chat", return_value="# Title\n\nBody"),
            ):
                capture("https://x.com", "ollama", "fake")
            self.assertTrue(captures_dir.exists())

    def test_chat_called_with_use_tools_false(self) -> None:
        web_result = json.dumps({"url": "https://x.com", "content": "text", "truncated": False})
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.dict(os.environ, {"TARS_NOTES_DIR": tmpdir}),
                mock.patch("tars.capture._run_web_tool", return_value=web_result),
                mock.patch("tars.capture.chat", return_value="# T\n\nBody") as mock_chat,
            ):
                capture("https://x.com", "ollama", "fake")
            mock_chat.assert_called_once()
            _, kwargs = mock_chat.call_args
            self.assertFalse(kwargs["use_tools"])


if __name__ == "__main__":
    unittest.main()
