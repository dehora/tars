import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import cli


class MemoryToolTests(unittest.TestCase):
    def test_memory_recall_requires_config(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            result = json.loads(cli._run_memory_tool("memory_recall", {}))
        self.assertIn("Memory not configured", result.get("error", ""))

    def test_memory_recall_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                result = json.loads(cli._run_memory_tool("memory_recall", {}))
        self.assertIn("No memory files found", result.get("error", ""))

    def test_memory_update_missing_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = os.path.join(tmpdir, "Memory.md")
            with open(memory_path, "w", encoding="utf-8") as handle:
                handle.write("- something else\n")
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                result = json.loads(
                    cli._run_memory_tool(
                        "memory_update",
                        {"old_content": "missing", "new_content": "updated"},
                    )
                )
        self.assertIn("Could not find existing entry", result.get("error", ""))

    def test_memory_remember_invalid_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                result = json.loads(
                    cli._run_memory_tool(
                        "memory_remember",
                        {"section": "invalid", "content": "oops"},
                    )
                )
        self.assertIn("Invalid section", result.get("error", ""))

    def test_build_system_prompt_includes_memory_preface(self) -> None:
        with mock.patch.object(cli, "_load_memory", return_value="- remembered"):
            prompt = cli._build_system_prompt()
        self.assertIn(cli.MEMORY_PROMPT_PREFACE, prompt)
        self.assertIn("<memory>", prompt)
        self.assertIn("- remembered", prompt)


if __name__ == "__main__":
    unittest.main()
