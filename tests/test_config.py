import os
import unittest
from unittest import mock

from tars.config import apply_cli_overrides, load_model_config


class ModelConfigTests(unittest.TestCase):
    def test_default_model_from_env(self) -> None:
        with mock.patch.dict(os.environ, {
            "TARS_MODEL_DEFAULT": "ollama:llama3.1:8b",
        }, clear=False):
            config = load_model_config()
        self.assertEqual(config.primary_provider, "ollama")
        self.assertEqual(config.primary_model, "llama3.1:8b")

    def test_remote_model_rejects_claude_alias(self) -> None:
        with mock.patch.dict(os.environ, {
            "TARS_MODEL_REMOTE": "claude:sonnet",
        }, clear=False):
            with self.assertRaises(ValueError):
                load_model_config()

    def test_cli_overrides_apply(self) -> None:
        with mock.patch.dict(os.environ, {
            "TARS_MODEL_DEFAULT": "ollama:llama3.1:8b",
            "TARS_MODEL_REMOTE": "claude:claude-sonnet-4-5-20250929",
        }, clear=False):
            base = load_model_config()
        updated = apply_cli_overrides(base, "ollama:llama3.2:latest", None)
        self.assertEqual(updated.primary_provider, "ollama")
        self.assertEqual(updated.primary_model, "llama3.2:latest")

    def test_remote_model_none_disables(self) -> None:
        with mock.patch.dict(os.environ, {
            "TARS_MODEL_REMOTE": "none",
        }, clear=False):
            config = load_model_config()
        self.assertIsNone(config.remote_provider)
        self.assertIsNone(config.remote_model)
