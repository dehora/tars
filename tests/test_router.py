import unittest

from tars.config import ModelConfig
from tars.router import RouteResult, route_message


_ESC_CONFIG = ModelConfig(
    primary_provider="ollama",
    primary_model="llama3.1:8b",
    remote_provider="claude",
    remote_model="sonnet",
    routing_policy="tool",
)


class TestRouter(unittest.TestCase):
    def test_no_escalation_configured(self):
        config = ModelConfig(
            primary_provider="ollama",
            primary_model="llama3.1:8b",
            remote_provider=None,
            remote_model=None,
            routing_policy="tool",
        )
        result = route_message("hello", config)
        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.model, "llama3.1:8b")

    def test_already_claude(self):
        config = ModelConfig(
            primary_provider="claude",
            primary_model="sonnet",
            remote_provider="claude",
            remote_model="sonnet",
            routing_policy="tool",
        )
        result = route_message("add task buy milk", config)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")

    def test_todoist_keywords(self):
        for msg in ["add buy milk to todoist", "remind me to call mom", "buy eggs"]:
            result = route_message(msg, _ESC_CONFIG)
            self.assertEqual(result.provider, "claude", f"failed for: {msg}")
            self.assertEqual(result.model, "sonnet", f"failed for: {msg}")

    def test_weather_keywords(self):
        result = route_message("what's the weather", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")

    def test_plain_chat(self):
        result = route_message("hello how are you", _ESC_CONFIG)
        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.model, "llama3.1:8b")

    def test_case_insensitive(self):
        result = route_message("BUY EGGS", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")

    def test_memory_keywords(self):
        result = route_message("remember that I like coffee", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")

    def test_note_keywords(self):
        result = route_message("note: interesting idea", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")

    def test_notes_keywords(self):
        for msg in ["check my notes", "search my daily note", "find it in obsidian"]:
            result = route_message(msg, _ESC_CONFIG)
            self.assertEqual(result.provider, "claude", f"failed for: {msg}")
            self.assertEqual(result.model, "sonnet", f"failed for: {msg}")

    def test_direct_tool_name(self):
        result = route_message("use weather_now for london", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")

    def test_todoist_hints(self):
        result = route_message("add task buy milk", _ESC_CONFIG)
        self.assertIn("todoist_add_task", result.tool_hints)

    def test_weather_hints(self):
        result = route_message("will it rain today?", _ESC_CONFIG)
        self.assertIn("weather_now", result.tool_hints)

    def test_no_hints_for_chat(self):
        result = route_message("hello how are you", _ESC_CONFIG)
        self.assertEqual(result.tool_hints, [])

    def test_multiple_hints_deduplicated(self):
        result = route_message("add task buy groceries from todoist", _ESC_CONFIG)
        self.assertEqual(len(result.tool_hints), len(set(result.tool_hints)))

    def test_direct_tool_name_hint(self):
        result = route_message("use weather_now", _ESC_CONFIG)
        self.assertEqual(result.tool_hints, ["weather_now"])

    def test_route_result_not_iterable(self):
        result = route_message("hello", _ESC_CONFIG)
        self.assertIsInstance(result, RouteResult)
        with self.assertRaises(TypeError):
            _, _ = result


if __name__ == "__main__":
    unittest.main()
