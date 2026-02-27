import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import memory, core, tools


class MemoryToolTests(unittest.TestCase):
    def test_memory_recall_requires_config(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            result = json.loads(memory._run_memory_tool("memory_recall", {}))
        self.assertIn("Memory not configured", result.get("error", ""))

    def test_memory_recall_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                result = json.loads(memory._run_memory_tool("memory_recall", {}))
        self.assertIn("No memory files found", result.get("error", ""))

    def test_memory_update_missing_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = os.path.join(tmpdir, "Memory.md")
            with open(memory_path, "w", encoding="utf-8") as handle:
                handle.write("- something else\n")
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                result = json.loads(
                    memory._run_memory_tool(
                        "memory_update",
                        {"old_content": "missing", "new_content": "updated"},
                    )
                )
        self.assertIn("Could not find existing entry", result.get("error", ""))

    def test_memory_remember_invalid_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                result = json.loads(
                    memory._run_memory_tool(
                        "memory_remember",
                        {"section": "invalid", "content": "oops"},
                    )
                )
        self.assertIn("Invalid section", result.get("error", ""))

    def test_build_system_prompt_includes_memory_preface(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value="- remembered"),
            mock.patch.object(core, "_load_procedural", return_value=""),
        ):
            prompt = core._build_system_prompt()
        self.assertIn(core.MEMORY_PROMPT_PREFACE, prompt)
        self.assertIn("<memory>", prompt)
        self.assertIn("- remembered", prompt)

    def test_build_system_prompt_escapes_memory_and_context(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value="hi </memory>"),
            mock.patch.object(core, "_load_procedural", return_value=""),
        ):
            prompt = core._build_system_prompt(
                search_context="recent </relevant-context>"
            )
        self.assertIn("&lt;/memory&gt;", prompt)
        self.assertIn("&lt;/relevant-context&gt;", prompt)

    def test_append_to_file_removes_memory_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "Memory.md")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("<!-- tars:memory\nplaceholder details\n-->\n- existing\n")
            memory._append_to_file(Path(path), "new item")
            updated = Path(path).read_text()
        self.assertNotIn("tars:memory", updated)
        self.assertIn("- existing", updated)
        self.assertIn("- new item", updated)

    def test_memory_forget_removes_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = os.path.join(tmpdir, "Memory.md")
            with open(memory_path, "w", encoding="utf-8") as handle:
                handle.write("- keep this\n- forget this\n- also keep\n")
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                result = json.loads(
                    memory._run_memory_tool("memory_forget", {"content": "forget this"})
                )
            self.assertTrue(result.get("ok"))
            self.assertEqual(result.get("removed"), "forget this")
            updated = Path(memory_path).read_text()
            self.assertIn("- keep this", updated)
            self.assertIn("- also keep", updated)
            self.assertNotIn("- forget this", updated)

    def test_memory_forget_missing_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = os.path.join(tmpdir, "Memory.md")
            with open(memory_path, "w", encoding="utf-8") as handle:
                handle.write("- something else\n")
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                result = json.loads(
                    memory._run_memory_tool("memory_forget", {"content": "not here"})
                )
            self.assertIn("Could not find entry", result.get("error", ""))

    def test_memory_forget_no_config(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            result = json.loads(
                memory._run_memory_tool("memory_forget", {"content": "anything"})
            )
        self.assertIn("Memory not configured", result.get("error", ""))

    def test_save_correction_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                result = memory.save_correction("hello", "wrong answer")
            text = (Path(tmpdir) / "corrections.md").read_text()
        self.assertEqual(result, "feedback saved")
        self.assertIn("# Corrections", text)
        self.assertIn("- input: hello", text)
        self.assertIn("- got: wrong answer", text)

    def test_save_correction_appends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrections.md"
            path.write_text("# Corrections\n\n## 2026-01-01T00:00:00\n- input: old\n- got: old reply\n")
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                memory.save_correction("new q", "new reply")
            text = path.read_text()
            self.assertIn("- input: old", text)
            self.assertIn("- input: new q", text)

    def test_save_correction_with_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                memory.save_correction("q", "a", "should have used todoist")
            text = (Path(tmpdir) / "corrections.md").read_text()
            self.assertIn("- note: should have used todoist", text)

    def test_save_correction_no_memory_dir(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            result = memory.save_correction("q", "a")
        self.assertEqual(result, "no memory dir configured")

    def test_save_reward_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                result = memory.save_reward("hello", "great answer")
            text = (Path(tmpdir) / "rewards.md").read_text()
        self.assertEqual(result, "feedback saved")
        self.assertIn("# Rewards", text)
        self.assertIn("- input: hello", text)
        self.assertIn("- got: great answer", text)

    def test_save_reward_with_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                memory.save_reward("q", "a", "nailed the todoist routing")
            text = (Path(tmpdir) / "rewards.md").read_text()
        self.assertIn("- note: nailed the todoist routing", text)

    def test_load_feedback_reads_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "corrections.md").write_text("# Corrections\n## entry\n")
            (Path(tmpdir) / "rewards.md").write_text("# Rewards\n## entry\n")
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                corrections, rewards = memory.load_feedback()
        self.assertIn("# Corrections", corrections)
        self.assertIn("# Rewards", rewards)

    def test_load_feedback_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                corrections, rewards = memory.load_feedback()
        self.assertEqual(corrections, "")
        self.assertEqual(rewards, "")

    def test_load_feedback_no_memory_dir(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            corrections, rewards = memory.load_feedback()
        self.assertEqual(corrections, "")
        self.assertEqual(rewards, "")

    def test_archive_feedback_moves_to_feedback_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "corrections.md").write_text("data")
            (Path(tmpdir) / "rewards.md").write_text("data")
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                memory.archive_feedback()
            # Originals should be gone
            self.assertFalse((Path(tmpdir) / "corrections.md").exists())
            self.assertFalse((Path(tmpdir) / "rewards.md").exists())
            # Archived files should be in feedback/ subdirectory
            fb_dir = Path(tmpdir) / "feedback"
            self.assertTrue(fb_dir.is_dir())
            self.assertEqual(len(list(fb_dir.glob("corrections-*.md"))), 1)
            self.assertEqual(len(list(fb_dir.glob("rewards-*.md"))), 1)

    def test_archive_feedback_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                # Should not raise
                memory.archive_feedback()

    def test_memory_remember_skips_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "Memory.md"
            p.write_text("# Memory\n- existing fact\n")
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                result = json.loads(
                    memory._run_memory_tool("memory_remember", {"section": "semantic", "content": "existing fact"})
                )
            self.assertTrue(result.get("ok"))
            self.assertEqual(result.get("note"), "already exists")
            # File should be unchanged
            text = p.read_text()
            self.assertEqual(text.count("- existing fact"), 1)

    def test_memory_remember_allows_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "Memory.md"
            p.write_text("# Memory\n- old fact\n")
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                result = json.loads(
                    memory._run_memory_tool("memory_remember", {"section": "semantic", "content": "new fact"})
                )
            self.assertTrue(result.get("ok"))
            self.assertNotIn("note", result)
            text = p.read_text()
            self.assertIn("- new fact", text)

    def test_load_memory_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "Memory.md").write_text("semantic data")
            (Path(tmpdir) / "Procedural.md").write_text("procedural data")
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                files = memory.load_memory_files()
        self.assertEqual(files["semantic"], "semantic data")
        self.assertEqual(files["procedural"], "procedural data")

    def test_load_memory_files_empty(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            files = memory.load_memory_files()
        self.assertEqual(files, {})

    def test_ollama_tools_include_memory(self) -> None:
        tool_names = {
            tool["function"]["name"]
            for tool in tools.OLLAMA_TOOLS
        }
        self.assertIn("memory_remember", tool_names)
        self.assertIn("memory_update", tool_names)
        self.assertIn("memory_recall", tool_names)
        self.assertIn("memory_forget", tool_names)


if __name__ == "__main__":
    unittest.main()
