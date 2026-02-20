import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars.cli import _parse_todoist_add


class ParseTodoistAddTests(unittest.TestCase):
    def test_simple(self) -> None:
        args = _parse_todoist_add(["buy", "eggs"])
        self.assertEqual(args["content"], "buy eggs")

    def test_single_word_due(self) -> None:
        args = _parse_todoist_add(["buy", "eggs", "--due", "today"])
        self.assertEqual(args["content"], "buy eggs")
        self.assertEqual(args["due"], "today")

    def test_multi_word_due(self) -> None:
        args = _parse_todoist_add(["buy", "eggs", "--due", "tomorrow", "3pm"])
        self.assertEqual(args["content"], "buy eggs")
        self.assertEqual(args["due"], "tomorrow 3pm")

    def test_due_with_date_string(self) -> None:
        args = _parse_todoist_add(["buy", "eggs", "--due", "next", "monday"])
        self.assertEqual(args["due"], "next monday")
        self.assertEqual(args["content"], "buy eggs")

    def test_all_flags(self) -> None:
        args = _parse_todoist_add([
            "buy", "eggs", "--due", "tomorrow", "3pm",
            "--project", "Groceries", "--priority", "4",
        ])
        self.assertEqual(args["content"], "buy eggs")
        self.assertEqual(args["due"], "tomorrow 3pm")
        self.assertEqual(args["project"], "Groceries")
        self.assertEqual(args["priority"], 4)

    def test_priority_is_int(self) -> None:
        args = _parse_todoist_add(["task", "--priority", "2"])
        self.assertEqual(args["priority"], 2)


if __name__ == "__main__":
    unittest.main()
