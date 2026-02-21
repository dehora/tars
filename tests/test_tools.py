import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars.tools import _clean_args


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


if __name__ == "__main__":
    unittest.main()
