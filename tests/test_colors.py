"""Tests for tars.colors — ANSI color helpers and OSC 8 sanitization."""

import unittest
from unittest import mock


class LinkSanitizationTests(unittest.TestCase):
    """Test that OSC 8 link() strips control characters (Fix 7)."""

    def _link_enabled(self, url: str, text: str) -> str:
        """Call link() with colors enabled."""
        import tars.colors as colors
        old = colors._ENABLED
        colors._ENABLED = True
        try:
            return colors.link(url, text)
        finally:
            colors._ENABLED = old

    def test_strips_escape_from_url(self) -> None:
        result = self._link_enabled("https://ok.com/\x1b]8;;evil\x1b\\", "click")
        self.assertNotIn("\x1b]8;;evil", result.replace("\x1b]8;;https", ""))

    def test_strips_bel_from_url(self) -> None:
        result = self._link_enabled("https://ok.com/\x07injected", "click")
        # BEL should be stripped from the URL portion
        url_part = result.split("\033\\")[0]
        self.assertNotIn("\x07", url_part)

    def test_strips_control_from_text(self) -> None:
        result = self._link_enabled("https://ok.com", "safe\x1btext")
        # The visible text (between opening ST and closing OSC) should be clean.
        # OSC 8 format: \033]8;;URL\033\TEXT\033]8;;\033\
        # "safetext" should appear (ESC stripped), not "safe\x1btext"
        self.assertIn("safetext", result)
        self.assertNotIn("safe\x1btext", result)

    def test_normal_url_unchanged(self) -> None:
        result = self._link_enabled("https://example.com", "Example")
        self.assertIn("https://example.com", result)
        self.assertIn("Example", result)

    def test_disabled_returns_text_only(self) -> None:
        import tars.colors as colors
        old = colors._ENABLED
        colors._ENABLED = False
        try:
            result = colors.link("https://example.com", "Example")
            self.assertEqual(result, "Example")
        finally:
            colors._ENABLED = old


if __name__ == "__main__":
    unittest.main()
