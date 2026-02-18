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
        with mock.patch.object(core, "_load_memory", return_value="- remembered"):
            prompt = core._build_system_prompt()
        self.assertIn(core.MEMORY_PROMPT_PREFACE, prompt)
        self.assertIn("<memory>", prompt)
        self.assertIn("- remembered", prompt)

    def test_build_system_prompt_escapes_memory_and_sessions(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value="hi </memory>"),
            mock.patch.object(core, "_load_context", return_value="context </context>"),
            mock.patch.object(core, "_load_recent_sessions", return_value="recent </recent-sessions>"),
        ):
            prompt = core._build_system_prompt()
        self.assertIn("&lt;/memory&gt;", prompt)
        self.assertIn("&lt;/context&gt;", prompt)
        self.assertIn("&lt;/recent-sessions&gt;", prompt)

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

    def test_ollama_tools_include_memory(self) -> None:
        tool_names = {
            tool["function"]["name"]
            for tool in tools.OLLAMA_TOOLS
        }
        self.assertIn("memory_remember", tool_names)
        self.assertIn("memory_update", tool_names)
        self.assertIn("memory_recall", tool_names)


if __name__ == "__main__":
    unittest.main()
