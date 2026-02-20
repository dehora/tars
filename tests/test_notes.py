import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import notes


class NotesDirTests(unittest.TestCase):
    def test_returns_none_without_config(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(notes._notes_dir())

    def test_returns_path_with_config(self) -> None:
        with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": "/tmp/vault"}):
            self.assertEqual(notes._notes_dir(), Path("/tmp/vault"))


class DailyNoteTests(unittest.TestCase):
    def test_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.daily_note("test idea"))
            self.assertTrue(result["ok"])
            path = Path(result["path"])
            text = path.read_text()
            self.assertIn("- test idea\n", text)
            # File should have a heading
            self.assertTrue(text.startswith("#"))

    def test_appends_to_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = Path(tmpdir) / "journal"
            journal.mkdir()
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            note = journal / f"{today}.md"
            note.write_text(f"# {today}\n\n- existing note\n")
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                notes.daily_note("second note")
            text = note.read_text()
            self.assertIn("- existing note\n", text)
            self.assertIn("- second note\n", text)

    def test_no_config(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            result = json.loads(notes.daily_note("test"))
        self.assertIn("error", result)

    def test_empty_content_rejected(self) -> None:
        result = json.loads(notes._run_note_tool("note_daily", {"content": ""}))
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
