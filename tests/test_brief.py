"""Tests for the daily brief module."""

import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars.brief import build_brief_sections, format_brief_cli, format_brief_text


class BuildBriefSectionsTests(unittest.TestCase):
    @mock.patch("tars.brief._load_pinned", return_value="- watching Severance S2\n")
    @mock.patch("tars.brief.run_tool", return_value='{"tasks": []}')
    def test_pinned_included_when_present(self, mock_run, mock_pinned) -> None:
        sections = build_brief_sections()
        labels = [s[0] for s in sections]
        self.assertIn("pinned", labels)
        self.assertEqual(labels[0], "pinned")
        pinned_content = next(c for l, c in sections if l == "pinned")
        self.assertIn("watching Severance S2", pinned_content)

    @mock.patch("tars.brief._load_pinned", return_value="")
    @mock.patch("tars.brief.run_tool", return_value='{"tasks": []}')
    def test_pinned_omitted_when_empty(self, mock_run, mock_pinned) -> None:
        sections = build_brief_sections()
        labels = [s[0] for s in sections]
        self.assertNotIn("pinned", labels)

    @mock.patch("tars.brief._load_pinned", return_value="   \n  \n")
    @mock.patch("tars.brief.run_tool", return_value='{"tasks": []}')
    def test_pinned_omitted_when_whitespace_only(self, mock_run, mock_pinned) -> None:
        sections = build_brief_sections()
        labels = [s[0] for s in sections]
        self.assertNotIn("pinned", labels)


    @mock.patch("tars.brief._load_pinned", side_effect=OSError("disk error"))
    @mock.patch("tars.brief.run_tool", return_value='{"tasks": []}')
    def test_pinned_load_failure_degrades(self, mock_run, mock_pinned) -> None:
        sections = build_brief_sections()
        labels = [s[0] for s in sections]
        self.assertNotIn("pinned", labels)
        self.assertIn("tasks", labels)


class FormatBriefTextTests(unittest.TestCase):
    def test_pinned_section_formatted(self) -> None:
        sections = [("pinned", "- watching Severance S2"), ("tasks", "no tasks")]
        result = format_brief_text(sections)
        self.assertIn("[pinned]", result)
        self.assertIn("watching Severance S2", result)
        self.assertIn("[tasks]", result)


class FormatBriefCliTests(unittest.TestCase):
    def test_pinned_section_formatted(self) -> None:
        sections = [("pinned", "- item one"), ("tasks", "no tasks")]
        result = format_brief_cli(sections)
        self.assertIn("pinned", result)
        self.assertIn("item one", result)


if __name__ == "__main__":
    unittest.main()
