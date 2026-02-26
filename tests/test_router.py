import unittest

from tars.config import ModelConfig
from tars.router import route_message


class TestRouter(unittest.TestCase):
    def test_no_escalation_configured(self):
        config = ModelConfig(
            primary_provider="ollama",
            primary_model="llama3.1:8b",
            remote_provider=None,
            remote_model=None,
            routing_policy="tool",
        )
        provider, model = route_message("hello", config)
        self.assertEqual(provider, "ollama")
        self.assertEqual(model, "llama3.1:8b")

    def test_already_claude(self):
        config = ModelConfig(
            primary_provider="claude",
            primary_model="sonnet",
            remote_provider="claude",
            remote_model="sonnet",
            routing_policy="tool",
        )
        provider, model = route_message("add task buy milk", config)
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")

    def test_todoist_keywords(self):
        config = ModelConfig(
            primary_provider="ollama",
            primary_model="llama3.1:8b",
            remote_provider="claude",
            remote_model="sonnet",
            routing_policy="tool",
        )
        for msg in ["add buy milk to todoist", "remind me to call mom", "buy eggs"]:
            provider, model = route_message(msg, config)
            self.assertEqual(provider, "claude", f"failed for: {msg}")
            self.assertEqual(model, "sonnet", f"failed for: {msg}")

    def test_weather_keywords(self):
        config = ModelConfig(
            primary_provider="ollama",
            primary_model="llama3.1:8b",
            remote_provider="claude",
            remote_model="sonnet",
            routing_policy="tool",
        )
        provider, model = route_message("what's the weather", config)
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")

    def test_plain_chat(self):
        config = ModelConfig(
            primary_provider="ollama",
            primary_model="llama3.1:8b",
            remote_provider="claude",
            remote_model="sonnet",
            routing_policy="tool",
        )
        provider, model = route_message("hello how are you", config)
        self.assertEqual(provider, "ollama")
        self.assertEqual(model, "llama3.1:8b")

    def test_case_insensitive(self):
        config = ModelConfig(
            primary_provider="ollama",
            primary_model="llama3.1:8b",
            remote_provider="claude",
            remote_model="sonnet",
            routing_policy="tool",
        )
        provider, model = route_message("BUY EGGS", config)
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")

    def test_memory_keywords(self):
        config = ModelConfig(
            primary_provider="ollama",
            primary_model="llama3.1:8b",
            remote_provider="claude",
            remote_model="sonnet",
            routing_policy="tool",
        )
        provider, model = route_message("remember that I like coffee", config)
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")

    def test_note_keywords(self):
        config = ModelConfig(
            primary_provider="ollama",
            primary_model="llama3.1:8b",
            remote_provider="claude",
            remote_model="sonnet",
            routing_policy="tool",
        )
        provider, model = route_message("note: interesting idea", config)
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")

    def test_notes_keywords(self):
        config = ModelConfig(
            primary_provider="ollama",
            primary_model="llama3.1:8b",
            remote_provider="claude",
            remote_model="sonnet",
            routing_policy="tool",
        )
        for msg in ["check my notes", "search my daily note", "find it in obsidian"]:
            provider, model = route_message(msg, config)
            self.assertEqual(provider, "claude", f"failed for: {msg}")
            self.assertEqual(model, "sonnet", f"failed for: {msg}")

    def test_direct_tool_name(self):
        config = ModelConfig(
            primary_provider="ollama",
            primary_model="llama3.1:8b",
            remote_provider="claude",
            remote_model="sonnet",
            routing_policy="tool",
        )
        provider, model = route_message("use weather_now for london", config)
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")


if __name__ == "__main__":
    unittest.main()
