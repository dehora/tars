import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import core


class ChatRoutingTests(unittest.TestCase):
    def test_chat_routes_to_anthropic(self) -> None:
        with mock.patch.object(core, "chat_anthropic", return_value="hi") as m:
            result = core.chat([{"role": "user", "content": "x"}], "claude", "sonnet")
        self.assertEqual(result, "hi")
        m.assert_called_once()

    def test_chat_routes_to_ollama(self) -> None:
        with mock.patch.object(core, "chat_ollama", return_value="hi") as m:
            result = core.chat([{"role": "user", "content": "x"}], "ollama", "llama3")
        self.assertEqual(result, "hi")
        m.assert_called_once()

    def test_chat_unknown_provider_raises(self) -> None:
        with self.assertRaises(ValueError):
            core.chat([], "unknown", "model")

    def test_chat_stream_routes_to_anthropic(self) -> None:
        with mock.patch.object(core, "chat_anthropic_stream", return_value=iter(["a"])) as m:
            result = list(core.chat_stream([{"role": "user", "content": "x"}], "claude", "sonnet"))
        self.assertEqual(result, ["a"])
        m.assert_called_once()

    def test_chat_stream_routes_to_ollama(self) -> None:
        with mock.patch.object(core, "chat_ollama_stream", return_value=iter(["b"])) as m:
            result = list(core.chat_stream([{"role": "user", "content": "x"}], "ollama", "llama3"))
        self.assertEqual(result, ["b"])
        m.assert_called_once()

    def test_chat_stream_unknown_provider_raises(self) -> None:
        with self.assertRaises(ValueError):
            list(core.chat_stream([], "unknown", "model"))

    def test_chat_passes_search_context(self) -> None:
        with mock.patch.object(core, "chat_ollama", return_value="ok") as m:
            core.chat([], "ollama", "m", search_context="ctx")
        m.assert_called_once_with([], "m", search_context="ctx", use_tools=True)


class SystemPromptContentTests(unittest.TestCase):
    def test_prompt_contains_routing_confidence(self) -> None:
        self.assertIn("ambiguous", core.SYSTEM_PROMPT)
        self.assertIn("clarifying question", core.SYSTEM_PROMPT)

    def test_prompt_no_blanket_must_call(self) -> None:
        self.assertNotIn("You MUST call", core.SYSTEM_PROMPT)


class BuildSystemPromptTests(unittest.TestCase):
    def test_without_context(self) -> None:
        with mock.patch.object(core, "_load_memory", return_value=""):
            prompt = core._build_system_prompt()
        self.assertEqual(prompt, core.SYSTEM_PROMPT)
        self.assertNotIn("<memory>", prompt)

    def test_with_memory(self) -> None:
        with mock.patch.object(core, "_load_memory", return_value="- fact"):
            prompt = core._build_system_prompt()
        self.assertIn("<memory>", prompt)
        self.assertIn("- fact", prompt)

    def test_with_search_context(self) -> None:
        with mock.patch.object(core, "_load_memory", return_value=""):
            prompt = core._build_system_prompt(search_context="recent stuff")
        self.assertIn("<relevant-context>", prompt)
        self.assertIn("recent stuff", prompt)


class SearchRelevantContextTests(unittest.TestCase):
    def test_empty_results(self) -> None:
        with mock.patch("tars.search.search", return_value=[]):
            result = core._search_relevant_context("hello")
        self.assertEqual(result, "")


class ParseModelTests(unittest.TestCase):
    def test_valid_format(self) -> None:
        provider, model = core.parse_model("ollama:llama3")
        self.assertEqual(provider, "ollama")
        self.assertEqual(model, "llama3")

    def test_invalid_format_raises(self) -> None:
        with self.assertRaises(ValueError):
            core.parse_model("nocolon")


if __name__ == "__main__":
    unittest.main()
