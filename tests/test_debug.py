"""Tests for the debug module."""

import os
import sys
import unittest
from io import StringIO
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import debug


class ConfigureTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = debug.VERBOSE
        debug.VERBOSE = False

    def tearDown(self) -> None:
        debug.VERBOSE = self._orig

    def test_configure_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"TARS_VERBOSE": "1"}):
            debug.configure(from_env=True)
        self.assertTrue(debug.VERBOSE)

    def test_configure_from_env_not_set(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            debug.configure(from_env=True)
        self.assertFalse(debug.VERBOSE)

    def test_configure_enable(self) -> None:
        debug.configure(enable=True)
        self.assertTrue(debug.VERBOSE)

    def test_configure_enable_false_clears(self) -> None:
        debug.VERBOSE = True
        debug.configure(enable=False)
        self.assertFalse(debug.VERBOSE)

    def test_configure_from_env_unset_clears(self) -> None:
        debug.VERBOSE = True
        with mock.patch.dict(os.environ, {}, clear=True):
            debug.configure(from_env=True)
        self.assertFalse(debug.VERBOSE)

    def test_default_verbose_is_false(self) -> None:
        debug.VERBOSE = False
        self.assertFalse(debug.VERBOSE)


class VerboseOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = debug.VERBOSE

    def tearDown(self) -> None:
        debug.VERBOSE = self._orig

    def test_verbose_prints_when_enabled(self) -> None:
        debug.VERBOSE = True
        buf = StringIO()
        with mock.patch("sys.stderr", buf):
            debug.verbose("test message")
        self.assertIn("test message", buf.getvalue())

    def test_verbose_silent_when_disabled(self) -> None:
        debug.VERBOSE = False
        buf = StringIO()
        with mock.patch("sys.stderr", buf):
            debug.verbose("should not appear")
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
