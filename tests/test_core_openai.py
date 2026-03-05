import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("openai", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import core


def _mock_tool_call(tc_id="tc_1", name="weather_now", arguments='{}'):
    tc = mock.Mock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _mock_choice(finish_reason="stop", content="hello", tool_calls=None):
    choice = mock.Mock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    return choice


def _mock_response(finish_reason="stop", content="hello", tool_calls=None):
    resp = mock.Mock()
    resp.choices = [_mock_choice(finish_reason, content, tool_calls)]
    return resp


class OpenAIChatRoutingTests(unittest.TestCase):
    def test_chat_routes_to_openai(self) -> None:
        with mock.patch.object(core, "chat_openai", return_value="hi") as m:
            result = core.chat([{"role": "user", "content": "x"}], "openai", "qwen3.5")
        self.assertEqual(result, "hi")
        m.assert_called_once()

    def test_chat_stream_routes_to_openai(self) -> None:
        with mock.patch.object(core, "chat_openai_stream", return_value=iter(["a"])) as m:
            result = list(core.chat_stream([{"role": "user", "content": "x"}], "openai", "qwen3.5"))
        self.assertEqual(result, ["a"])
        m.assert_called_once()

    def test_chat_passes_search_context(self) -> None:
        with mock.patch.object(core, "chat_openai", return_value="ok") as m:
            core.chat([], "openai", "m", search_context="ctx")
        m.assert_called_once_with([], "m", search_context="ctx", use_tools=True, tool_hints=None)


class OpenAIChatTests(unittest.TestCase):
    def test_simple_reply(self) -> None:
        mock_client = mock.Mock()
        mock_client.chat.completions.create.return_value = _mock_response(content="hi there")
        with (
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[]),
            mock.patch("tars.core.openai") as mock_openai,
        ):
            mock_openai.OpenAI.return_value = mock_client
            result = core.chat_openai([{"role": "user", "content": "hello"}], "qwen3.5")
        self.assertEqual(result, "hi there")

    def test_tool_loop_forwards_tool_call_id(self) -> None:
        tc = _mock_tool_call(tc_id="call_42", name="weather_now", arguments='{"lat": 0}')
        tool_response = _mock_response(finish_reason="tool_calls", tool_calls=[tc])
        final_response = _mock_response(content="sunny")

        mock_client = mock.Mock()
        mock_client.chat.completions.create.side_effect = [tool_response, final_response]

        with (
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[{"type": "function"}]),
            mock.patch.object(core, "run_tool", return_value="22C") as run,
            mock.patch.object(core, "append_daily"),
            mock.patch("tars.core.openai") as mock_openai,
        ):
            mock_openai.OpenAI.return_value = mock_client
            result = core.chat_openai([{"role": "user", "content": "weather?"}], "qwen3.5")

        self.assertEqual(result, "sunny")
        run.assert_called_once_with("weather_now", {"lat": 0})
        # Verify tool_call_id was forwarded in messages
        second_call_msgs = mock_client.chat.completions.create.call_args_list[1][1]["messages"]
        tool_msg = [m for m in second_call_msgs if isinstance(m, dict) and m.get("role") == "tool"]
        self.assertEqual(len(tool_msg), 1)
        self.assertEqual(tool_msg[0]["tool_call_id"], "call_42")

    def test_tool_loop_bounded(self) -> None:
        tc = _mock_tool_call()
        always_tool = _mock_response(finish_reason="tool_calls", tool_calls=[tc])
        mock_client = mock.Mock()
        mock_client.chat.completions.create.return_value = always_tool

        with (
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[{"type": "function"}]),
            mock.patch.object(core, "run_tool", return_value="ok"),
            mock.patch.object(core, "append_daily"),
            mock.patch("tars.core.openai") as mock_openai,
        ):
            mock_openai.OpenAI.return_value = mock_client
            result = core.chat_openai([{"role": "user", "content": "x"}], "qwen3.5")

        self.assertIn("maximum number of tool calls", result)
        self.assertEqual(mock_client.chat.completions.create.call_count, core._MAX_TOOL_ROUNDS)

    def test_json_string_arguments_parsed(self) -> None:
        tc = _mock_tool_call(arguments='{"city": "Dublin"}')
        tool_response = _mock_response(finish_reason="tool_calls", tool_calls=[tc])
        final_response = _mock_response(content="done")

        mock_client = mock.Mock()
        mock_client.chat.completions.create.side_effect = [tool_response, final_response]

        with (
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[{"type": "function"}]),
            mock.patch.object(core, "run_tool", return_value="ok") as run,
            mock.patch.object(core, "append_daily"),
            mock.patch("tars.core.openai") as mock_openai,
        ):
            mock_openai.OpenAI.return_value = mock_client
            core.chat_openai([{"role": "user", "content": "x"}], "qwen3.5")

        run.assert_called_once_with("weather_now", {"city": "Dublin"})

    def test_invalid_json_arguments_fallback(self) -> None:
        tc = _mock_tool_call(arguments="not json")
        tool_response = _mock_response(finish_reason="tool_calls", tool_calls=[tc])
        final_response = _mock_response(content="done")

        mock_client = mock.Mock()
        mock_client.chat.completions.create.side_effect = [tool_response, final_response]

        with (
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[{"type": "function"}]),
            mock.patch.object(core, "run_tool", return_value="ok") as run,
            mock.patch.object(core, "append_daily"),
            mock.patch("tars.core.openai") as mock_openai,
        ):
            mock_openai.OpenAI.return_value = mock_client
            core.chat_openai([{"role": "user", "content": "x"}], "qwen3.5")

        run.assert_called_once_with("weather_now", {})


class OpenAIStreamTests(unittest.TestCase):
    def test_yields_content_chunks(self) -> None:
        chunks = []
        for text in ["hel", "lo ", "world"]:
            c = mock.Mock()
            c.choices = [mock.Mock()]
            c.choices[0].delta.content = text
            chunks.append(c)

        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.Mock(return_value=iter(chunks))
        mock_stream.__exit__ = mock.Mock(return_value=False)

        mock_client = mock.Mock()
        mock_client.chat.completions.create.return_value = mock_stream

        with (
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[]),
            mock.patch("tars.core.openai") as mock_openai,
        ):
            mock_openai.OpenAI.return_value = mock_client
            result = list(core.chat_openai_stream(
                [{"role": "user", "content": "hi"}], "qwen3.5",
            ))
        self.assertEqual(result, ["hel", "lo ", "world"])

    def test_no_preflight_when_no_tools(self) -> None:
        chunk = mock.Mock()
        chunk.choices = [mock.Mock()]
        chunk.choices[0].delta.content = "hello"

        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.Mock(return_value=iter([chunk]))
        mock_stream.__exit__ = mock.Mock(return_value=False)

        mock_client = mock.Mock()
        mock_client.chat.completions.create.return_value = mock_stream

        with (
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[]),
            mock.patch("tars.core.openai") as mock_openai,
        ):
            mock_openai.OpenAI.return_value = mock_client
            result = list(core.chat_openai_stream(
                [{"role": "user", "content": "hi"}], "qwen3.5",
                use_tools=False,
            ))

        self.assertEqual(mock_client.chat.completions.create.call_count, 1)
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertTrue(kwargs.get("stream"))
        self.assertEqual(result, ["hello"])

    def test_stream_tool_loop_bounded(self) -> None:
        tc = _mock_tool_call()
        always_tool = _mock_response(finish_reason="tool_calls", tool_calls=[tc])
        mock_client = mock.Mock()
        mock_client.chat.completions.create.return_value = always_tool

        with (
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[{"type": "function"}]),
            mock.patch.object(core, "run_tool", return_value="ok"),
            mock.patch.object(core, "append_daily"),
            mock.patch("tars.core.openai") as mock_openai,
        ):
            mock_openai.OpenAI.return_value = mock_client
            result = list(core.chat_openai_stream(
                [{"role": "user", "content": "x"}], "qwen3.5",
            ))

        self.assertEqual(len(result), 1)
        self.assertIn("maximum number of tool calls", result[0])


class ParseToolArgumentsTests(unittest.TestCase):
    def test_dict_passthrough(self) -> None:
        self.assertEqual(core._parse_tool_arguments({"a": 1}), {"a": 1})

    def test_json_string(self) -> None:
        self.assertEqual(core._parse_tool_arguments('{"a": 1}'), {"a": 1})

    def test_invalid_json(self) -> None:
        self.assertEqual(core._parse_tool_arguments("not json"), {})

    def test_none(self) -> None:
        self.assertEqual(core._parse_tool_arguments(None), {})


if __name__ == "__main__":
    unittest.main()
