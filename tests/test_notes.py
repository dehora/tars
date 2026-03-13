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


class ValidateNotePathTests(unittest.TestCase):
    def test_rejects_traversal_dotdot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                _, err = notes._validate_note_path("../secret.md")
        self.assertIn("traversal", err)

    def test_rejects_nested_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                _, err = notes._validate_note_path("foo/../../secret.md")
        self.assertIn("traversal", err)

    def test_rejects_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                _, err = notes._validate_note_path("/etc/passwd")
        self.assertIsNotNone(err)
        self.assertIn("absolute", err)

    def test_auto_appends_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                resolved, err = notes._validate_note_path("Pain Log")
        self.assertIsNone(err)
        self.assertTrue(str(resolved).endswith(".md"))

    def test_accepts_nested_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                resolved, err = notes._validate_note_path("000 Self/Exercise Pain Log.md")
        self.assertIsNone(err)
        self.assertIn("000 Self", str(resolved))

    def test_rejects_empty_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                _, err = notes._validate_note_path("")
        self.assertIn("required", err)

    def test_no_config_returns_error(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            _, err = notes._validate_note_path("test.md")
        self.assertIn("not configured", err)


class NoteWriteTests(unittest.TestCase):
    def test_creates_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_write("test.md", "# Test\nHello"))
                self.assertTrue(result["ok"])
                self.assertTrue(result["created"])
                self.assertEqual((Path(tmpdir) / "test.md").read_text(), "# Test\nHello")

    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_write("deep/nested/note.md", "content"))
                self.assertTrue(result["ok"])
                self.assertTrue((Path(tmpdir) / "deep" / "nested" / "note.md").exists())

    def test_refuses_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "exists.md").write_text("original")
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_write("exists.md", "new content"))
                self.assertIn("error", result)
                self.assertIn("already exists", result["error"])
                self.assertEqual((Path(tmpdir) / "exists.md").read_text(), "original")

    def test_overwrite_flag_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "exists.md").write_text("original")
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_write("exists.md", "replaced", overwrite=True))
                self.assertTrue(result["ok"])
                self.assertTrue(result["overwritten"])
                self.assertEqual((Path(tmpdir) / "exists.md").read_text(), "replaced")

    def test_no_config_returns_error(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            result = json.loads(notes.note_write("test.md", "content"))
        self.assertIn("error", result)

    def test_traversal_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_write("../evil.md", "pwned"))
        self.assertIn("error", result)

    def test_empty_content_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_write("test.md", ""))
        self.assertIn("error", result)


class NoteReadTests(unittest.TestCase):
    def test_reads_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "hello.md").write_text("# Hello\nWorld")
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_read("hello.md"))
                self.assertTrue(result["ok"])
                self.assertEqual(result["content"], "# Hello\nWorld")
                self.assertFalse(result["truncated"])

    def test_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_read("nope.md"))
        self.assertIn("error", result)
        self.assertIn("not found", result["error"])

    def test_truncates_large_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            big = "x" * (notes._NOTE_READ_MAX_BYTES + 1000)
            (Path(tmpdir) / "big.md").write_text(big)
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_read("big.md"))
                self.assertTrue(result["truncated"])
                self.assertEqual(len(result["content"]), notes._NOTE_READ_MAX_BYTES)

    def test_traversal_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_read("../../etc/passwd"))
        self.assertIn("error", result)

    def test_no_config_returns_error(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            result = json.loads(notes.note_read("test.md"))
        self.assertIn("error", result)


class NoteAppendTests(unittest.TestCase):
    def test_creates_file_if_not_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_append("log.md", "first entry"))
                self.assertTrue(result["ok"])
                self.assertTrue(result["created"])
                self.assertEqual((Path(tmpdir) / "log.md").read_text(), "first entry")

    def test_appends_to_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "log.md").write_text("line 1")
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_append("log.md", "line 2"))
                self.assertTrue(result["ok"])
                self.assertFalse(result["created"])
                text = (Path(tmpdir) / "log.md").read_text()
                self.assertIn("line 1", text)
                self.assertIn("line 2", text)

    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_append("sub/dir/log.md", "entry"))
                self.assertTrue(result["ok"])
                self.assertTrue((Path(tmpdir) / "sub" / "dir" / "log.md").exists())

    def test_traversal_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_append("../evil.md", "pwned"))
        self.assertIn("error", result)

    def test_no_config_returns_error(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            result = json.loads(notes.note_append("log.md", "entry"))
        self.assertIn("error", result)

    def test_empty_content_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes.note_append("log.md", "  "))
        self.assertIn("error", result)


class RunNoteToolDispatchTests(unittest.TestCase):
    def test_dispatches_note_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes._run_note_tool("note_write", {"path": "t.md", "content": "hi"}))
                self.assertTrue(result["ok"])

    def test_dispatches_note_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "t.md").write_text("hello")
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes._run_note_tool("note_read", {"path": "t.md"}))
                self.assertEqual(result["content"], "hello")

    def test_dispatches_note_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"TARS_NOTES_DIR": tmpdir}):
                result = json.loads(notes._run_note_tool("note_append", {"path": "t.md", "content": "entry"}))
                self.assertTrue(result["ok"])

    def test_unknown_tool_returns_error(self) -> None:
        result = json.loads(notes._run_note_tool("note_unknown", {}))
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
