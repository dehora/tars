import os
import unittest
from unittest import mock

from tars.router import route_message


class TestRouter(unittest.TestCase):

    @mock.patch.dict(os.environ, {}, clear=False)
    @mock.patch("tars.router.escalation_config", return_value=None)
    def test_no_escalation_configured(self, _mock_esc):
        provider, model = route_message("hello", "ollama", "llama3.1:8b")
        self.assertEqual(provider, "ollama")
        self.assertEqual(model, "llama3.1:8b")

    @mock.patch("tars.router.escalation_config", return_value=("claude", "sonnet"))
    def test_already_claude(self, _mock_esc):
        provider, model = route_message("add task buy milk", "claude", "sonnet")
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")

    @mock.patch("tars.router.escalation_config", return_value=("claude", "sonnet"))
    def test_todoist_keywords(self, _mock_esc):
        for msg in ["add buy milk to todoist", "remind me to call mom", "buy eggs"]:
            provider, model = route_message(msg, "ollama", "llama3.1:8b")
            self.assertEqual(provider, "claude", f"failed for: {msg}")
            self.assertEqual(model, "sonnet", f"failed for: {msg}")

    @mock.patch("tars.router.escalation_config", return_value=("claude", "sonnet"))
    def test_weather_keywords(self, _mock_esc):
        provider, model = route_message("what's the weather", "ollama", "llama3.1:8b")
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")

    @mock.patch("tars.router.escalation_config", return_value=("claude", "sonnet"))
    def test_plain_chat(self, _mock_esc):
        provider, model = route_message("hello how are you", "ollama", "llama3.1:8b")
        self.assertEqual(provider, "ollama")
        self.assertEqual(model, "llama3.1:8b")

    @mock.patch("tars.router.escalation_config", return_value=("claude", "sonnet"))
    def test_case_insensitive(self, _mock_esc):
        provider, model = route_message("BUY EGGS", "ollama", "llama3.1:8b")
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")

    @mock.patch("tars.router.escalation_config", return_value=("claude", "sonnet"))
    def test_memory_keywords(self, _mock_esc):
        provider, model = route_message(
            "remember that I like coffee", "ollama", "llama3.1:8b",
        )
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")

    @mock.patch("tars.router.escalation_config", return_value=("claude", "sonnet"))
    def test_note_keywords(self, _mock_esc):
        provider, model = route_message(
            "note: interesting idea", "ollama", "llama3.1:8b",
        )
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")

    @mock.patch("tars.router.escalation_config", return_value=("claude", "sonnet"))
    def test_direct_tool_name(self, _mock_esc):
        provider, model = route_message(
            "use weather_now for london", "ollama", "llama3.1:8b",
        )
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")


if __name__ == "__main__":
    unittest.main()
