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
        m.assert_called_once_with([], "m", search_context="ctx", use_tools=True, tool_hints=None)


class SystemPromptContentTests(unittest.TestCase):
    def test_prompt_contains_routing_confidence(self) -> None:
        self.assertIn("ambiguous", core.SYSTEM_PROMPT)
        self.assertIn("clarifying question", core.SYSTEM_PROMPT)

    def test_prompt_no_blanket_must_call(self) -> None:
        self.assertNotIn("You MUST call", core.SYSTEM_PROMPT)


class BuildSystemPromptTests(unittest.TestCase):
    def test_without_context(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
        ):
            prompt = core._build_system_prompt()
        self.assertEqual(prompt, core.SYSTEM_PROMPT)
        self.assertNotIn("<memory>", prompt)

    def test_with_memory(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value="- fact"),
            mock.patch.object(core, "_load_procedural", return_value=""),
        ):
            prompt = core._build_system_prompt()
        self.assertIn("<memory>", prompt)
        self.assertIn("- fact", prompt)

    def test_with_search_context(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
        ):
            prompt = core._build_system_prompt(search_context="recent stuff")
        self.assertIn("<relevant-context>", prompt)
        self.assertIn("recent stuff", prompt)

    def test_tool_hints_in_prompt(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
        ):
            prompt = core._build_system_prompt(tool_hints=["todoist_add_task", "weather_now"])
        self.assertIn("<tool-hints>", prompt)
        self.assertIn("todoist_add_task", prompt)
        self.assertIn("weather_now", prompt)

    def test_procedural_in_prompt(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value="- always confirm tasks"),
        ):
            prompt = core._build_system_prompt()
        self.assertIn("<procedural-rules>", prompt)
        self.assertIn("- always confirm tasks", prompt)

    def test_procedural_empty_excluded(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
        ):
            prompt = core._build_system_prompt()
        self.assertNotIn("<procedural-rules>", prompt)

    def test_tool_hints_before_untrusted(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value="- fact"),
            mock.patch.object(core, "_load_procedural", return_value=""),
        ):
            prompt = core._build_system_prompt(tool_hints=["weather_now"])
        hints_pos = prompt.index("<tool-hints>")
        preface_pos = prompt.index(core.MEMORY_PROMPT_PREFACE)
        self.assertLess(hints_pos, preface_pos)


class DailyContextCapTests(unittest.TestCase):
    def test_daily_context_capped(self) -> None:
        lines = [f"- {i:02d}:00 tool:weather — ok" for i in range(100)]
        big_daily = "\n".join(lines)
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "load_daily", return_value=big_daily),
        ):
            prompt = core._build_system_prompt()
        self.assertIn("<daily-context ", prompt)
        # Should only contain the last _MAX_DAILY_LINES lines
        daily_section = prompt[prompt.index("<daily-context "):].split(">", 1)[1].split("</daily-context>")[0]
        daily_lines = [l for l in daily_section.strip().splitlines() if l.strip()]
        self.assertLessEqual(len(daily_lines), core._MAX_DAILY_LINES)

    def test_short_daily_not_truncated(self) -> None:
        daily = "- 08:00 tool:weather — sunny\n- 09:00 session compacted"
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "load_daily", return_value=daily),
        ):
            prompt = core._build_system_prompt()
        self.assertIn("tool:weather", prompt)
        self.assertIn("session compacted", prompt)


class SearchRelevantContextTests(unittest.TestCase):
    def test_empty_results(self) -> None:
        with mock.patch("tars.search.search", return_value=[]):
            result = core._search_relevant_context("hello")
        self.assertEqual(result, "")


class ToolLoopBoundTests(unittest.TestCase):
    def test_max_tool_rounds_constant(self) -> None:
        self.assertEqual(core._MAX_TOOL_ROUNDS, 10)

    def test_anthropic_loop_bounded(self) -> None:
        mock_response = mock.Mock()
        mock_response.stop_reason = "tool_use"
        mock_block = mock.Mock()
        mock_block.type = "tool_use"
        mock_block.name = "weather_now"
        mock_block.input = {}
        mock_block.id = "t1"
        mock_response.content = [mock_block]
        mock_client = mock.Mock()
        mock_client.messages.create.return_value = mock_response
        with (
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[{"name": "weather_now"}]),
            mock.patch.object(core, "run_tool", return_value="ok"),
            mock.patch.object(core, "append_daily"),
            mock.patch("anthropic.Anthropic", return_value=mock_client),
        ):
            result = core.chat_anthropic([{"role": "user", "content": "x"}], "sonnet")
        self.assertIn("maximum number of tool calls", result)
        self.assertEqual(mock_client.messages.create.call_count, core._MAX_TOOL_ROUNDS)

    def test_ollama_loop_bounded(self) -> None:
        mock_func = mock.Mock()
        mock_func.name = "weather_now"
        mock_func.arguments = {}
        mock_tool_call = mock.Mock()
        mock_tool_call.function = mock_func
        mock_message = mock.Mock()
        mock_message.tool_calls = [mock_tool_call]
        mock_message.content = ""
        mock_response = mock.Mock()
        mock_response.message = mock_message
        mock_ollama = mock.Mock()
        mock_ollama.chat = mock.Mock(return_value=mock_response)
        with (
            mock.patch.object(core, "ollama", mock_ollama),
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[{"name": "weather_now"}]),
            mock.patch.object(core, "run_tool", return_value="ok"),
            mock.patch.object(core, "append_daily"),
        ):
            result = core.chat_ollama([{"role": "user", "content": "x"}], "llama3")
        self.assertIn("maximum number of tool calls", result)
        self.assertEqual(mock_ollama.chat.call_count, core._MAX_TOOL_ROUNDS)

    def test_anthropic_stream_loop_bounded(self) -> None:
        mock_response = mock.Mock()
        mock_response.stop_reason = "tool_use"
        mock_block = mock.Mock()
        mock_block.type = "tool_use"
        mock_block.name = "weather_now"
        mock_block.input = {}
        mock_block.id = "t1"
        mock_response.content = [mock_block]
        mock_client = mock.Mock()
        mock_client.messages.create.return_value = mock_response
        with (
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[{"name": "weather_now"}]),
            mock.patch.object(core, "run_tool", return_value="ok"),
            mock.patch.object(core, "append_daily"),
            mock.patch("anthropic.Anthropic", return_value=mock_client),
        ):
            result = list(core.chat_anthropic_stream([{"role": "user", "content": "x"}], "sonnet"))
        self.assertEqual(len(result), 1)
        self.assertIn("maximum number of tool calls", result[0])

    def test_ollama_stream_loop_bounded(self) -> None:
        mock_func = mock.Mock()
        mock_func.name = "weather_now"
        mock_func.arguments = {}
        mock_tool_call = mock.Mock()
        mock_tool_call.function = mock_func
        mock_message = mock.Mock()
        mock_message.tool_calls = [mock_tool_call]
        mock_message.content = ""
        mock_response = mock.Mock()
        mock_response.message = mock_message
        mock_ollama = mock.Mock()
        mock_ollama.chat = mock.Mock(return_value=mock_response)
        with (
            mock.patch.object(core, "ollama", mock_ollama),
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[{"name": "weather_now"}]),
            mock.patch.object(core, "run_tool", return_value="ok"),
            mock.patch.object(core, "append_daily"),
        ):
            result = list(core.chat_ollama_stream([{"role": "user", "content": "x"}], "llama3"))
        self.assertEqual(len(result), 1)
        self.assertIn("maximum number of tool calls", result[0])


class StreamUseToolsTests(unittest.TestCase):
    def test_chat_stream_use_tools_false_anthropic(self) -> None:
        with mock.patch.object(
            core, "chat_anthropic_stream", return_value=iter(["hi"]),
        ) as m:
            list(core.chat_stream(
                [{"role": "user", "content": "x"}], "claude", "sonnet",
                use_tools=False,
            ))
        _, kwargs = m.call_args
        self.assertFalse(kwargs["use_tools"])

    def test_chat_stream_use_tools_false_ollama(self) -> None:
        with mock.patch.object(
            core, "chat_ollama_stream", return_value=iter(["hi"]),
        ) as m:
            list(core.chat_stream(
                [{"role": "user", "content": "x"}], "ollama", "llama3",
                use_tools=False,
            ))
        _, kwargs = m.call_args
        self.assertFalse(kwargs["use_tools"])


class StreamNoPreflightTests(unittest.TestCase):
    def test_anthropic_stream_no_preflight_when_no_tools(self) -> None:
        mock_client = mock.Mock()
        mock_stream_ctx = mock.MagicMock()
        mock_stream_ctx.__enter__ = mock.Mock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = mock.Mock(return_value=False)
        mock_stream_ctx.text_stream = iter(["hello"])
        mock_client.messages.stream.return_value = mock_stream_ctx

        with mock.patch("tars.core.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            result = list(core.chat_anthropic_stream(
                [{"role": "user", "content": "hi"}], "sonnet",
                use_tools=False,
            ))
        mock_client.messages.create.assert_not_called()
        self.assertEqual(result, ["hello"])

    def test_ollama_stream_no_preflight_when_no_tools(self) -> None:
        chunk = mock.Mock()
        chunk.message.content = "hello"
        mock_ollama = mock.Mock()
        mock_ollama.chat.return_value = iter([chunk])
        with mock.patch.object(core, "ollama", mock_ollama):
            result = list(core.chat_ollama_stream(
                [{"role": "user", "content": "hi"}], "llama3",
                use_tools=False,
            ))
        self.assertEqual(mock_ollama.chat.call_count, 1)
        _, kwargs = mock_ollama.chat.call_args
        self.assertTrue(kwargs.get("stream"))
        self.assertEqual(result, ["hello"])


class DailyContextProvenanceTests(unittest.TestCase):
    def test_daily_context_has_type_attribute(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "load_daily", return_value="- 08:00 captured: example.com"),
        ):
            prompt = core._build_system_prompt()
        self.assertIn('type="tars-generated', prompt)
        self.assertIn("summarized web content", prompt)


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
