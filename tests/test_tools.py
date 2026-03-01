import json
import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars.tools import _clean_args, run_tool


class CleanArgsTests(unittest.TestCase):
    def test_strips_empty_strings(self) -> None:
        self.assertEqual(_clean_args({"content": "eggs", "due": "", "project": ""}), {"content": "eggs"})

    def test_strips_none(self) -> None:
        self.assertEqual(_clean_args({"content": "eggs", "due": None}), {"content": "eggs"})

    def test_keeps_valid_values(self) -> None:
        args = {"content": "eggs", "due": "tomorrow", "project": "Groceries"}
        self.assertEqual(_clean_args(args), args)

    def test_keeps_zero(self) -> None:
        self.assertEqual(_clean_args({"days": 0, "priority": 0}), {"days": 0, "priority": 0})

    def test_keeps_false(self) -> None:
        self.assertEqual(_clean_args({"flag": False}), {"flag": False})

    def test_empty_dict(self) -> None:
        self.assertEqual(_clean_args({}), {})


class RequiredFieldValidationTests(unittest.TestCase):
    def test_todoist_add_task_empty_content(self) -> None:
        result = json.loads(run_tool("todoist_add_task", {"content": ""}, quiet=True))
        self.assertIn("error", result)
        self.assertIn("content", result["error"])

    def test_memory_update_missing_fields(self) -> None:
        result = json.loads(run_tool("memory_update", {}, quiet=True))
        self.assertIn("error", result)
        self.assertIn("old_content", result["error"])

    def test_todoist_today_no_required_fields(self) -> None:
        with mock.patch("tars.tools._resolve_td", return_value=None):
            result = json.loads(run_tool("todoist_today", {}, quiet=True))
        self.assertNotIn("missing required", result.get("error", ""))


if __name__ == "__main__":
    unittest.main()
