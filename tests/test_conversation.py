import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("openai", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

# Ensure the anthropic mock has real exception classes for fallback tests.
# (the mock may have been created by another test file loaded first)
_APIStatusError = type("APIStatusError", (Exception,), {"status_code": 0})
sys.modules["anthropic"].APIStatusError = _APIStatusError
sys.modules["anthropic"].RateLimitError = type("RateLimitError", (_APIStatusError,), {"status_code": 429})
sys.modules["anthropic"].BadRequestError = type("BadRequestError", (_APIStatusError,), {"status_code": 400})
sys.modules["anthropic"].APIConnectionError = type("APIConnectionError", (Exception,), {})
sys.modules["anthropic"].APITimeoutError = type("APITimeoutError", (Exception,), {})

_OAIAPIStatusError = type("APIStatusError", (Exception,), {"status_code": 0})
sys.modules["openai"].APIStatusError = _OAIAPIStatusError
sys.modules["openai"].APIConnectionError = type("APIConnectionError", (Exception,), {})
sys.modules["openai"].APITimeoutError = type("APITimeoutError", (Exception,), {})

from tars import conversation
from tars.conversation import (
    Conversation,
    _effective_search_context,
    _fetch_daily_brief,
    process_message,
    process_message_stream,
    save_session,
)
from tars.router import RouteResult


class ProcessMessageTests(unittest.TestCase):
    def test_appends_messages_and_returns_reply(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="fake")
        with mock.patch.object(conversation, "chat", return_value="hi there"):
            reply = process_message(conv, "hello")
        self.assertEqual(reply, "hi there")
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0], {"role": "user", "content": "hello"})
        self.assertEqual(conv.messages[1], {"role": "assistant", "content": "hi there"})
        self.assertEqual(conv.msg_count, 1)

    def test_first_message_triggers_search(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="fake")
        with (
            mock.patch.object(conversation, "chat", return_value="ok"),
            mock.patch.object(conversation, "_search_relevant_context", return_value="ctx") as search,
        ):
            process_message(conv, "hello")
        search.assert_called_once_with("hello")
        self.assertEqual(conv.search_context, "ctx")

    def test_second_message_skips_search(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="fake")
        conv.messages.append({"role": "user", "content": "first"})
        conv.search_context = "already set"
        with (
            mock.patch.object(conversation, "chat", return_value="ok"),
            mock.patch.object(conversation, "_search_relevant_context") as search,
        ):
            process_message(conv, "second")
        search.assert_not_called()

    def test_compaction_triggers_at_interval(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="fake")
        with tempfile.TemporaryDirectory() as tmpdir:
            session_file = Path(tmpdir) / "session.md"
            with (
                mock.patch.object(conversation, "SESSION_COMPACTION_INTERVAL", 2),
                mock.patch.object(conversation, "chat", return_value="ok"),
                mock.patch.object(conversation, "_summarize_session", return_value="summary") as summarize,
                mock.patch.object(conversation, "_save_session") as save,
            ):
                process_message(conv, "msg 1", session_file)
                self.assertEqual(summarize.call_count, 0)
                process_message(conv, "msg 2", session_file)
                self.assertEqual(summarize.call_count, 1)
            save.assert_called_once()
            self.assertTrue(save.call_args.kwargs.get("is_compaction"))


class LastModelTrackingTests(unittest.TestCase):
    def test_process_message_sets_last_provider_model(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("claude", "sonnet")),
            mock.patch.object(conversation, "chat", return_value="ok"),
        ):
            process_message(conv, "hello")
        self.assertEqual(conv.last_provider, "claude")
        self.assertEqual(conv.last_model, "sonnet")

    def test_process_message_fallback_sets_default_model(self) -> None:
        _anthropic = sys.modules["anthropic"]
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        err = _anthropic.RateLimitError("error")
        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("claude", "sonnet")),
            mock.patch.object(conversation, "chat", side_effect=[err, "fallback"]),
        ):
            process_message(conv, "hello")
        self.assertEqual(conv.last_provider, "ollama")
        self.assertEqual(conv.last_model, "llama3.1:8b")

    def test_process_message_stream_sets_last_provider_model(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("ollama", "llama3.1:8b")),
            mock.patch.object(conversation, "chat_stream", return_value=iter(["ok"])),
        ):
            list(process_message_stream(conv, "hello"))
        self.assertEqual(conv.last_provider, "ollama")
        self.assertEqual(conv.last_model, "llama3.1:8b")

    def test_stream_escalated_sets_provider_model(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("claude", "sonnet")),
            mock.patch.object(conversation, "chat", return_value="buffered"),
        ):
            list(process_message_stream(conv, "hello"))
        self.assertEqual(conv.last_provider, "claude")
        self.assertEqual(conv.last_model, "sonnet")


class ProcessMessageStreamTests(unittest.TestCase):
    def test_yields_deltas(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="fake")
        with mock.patch.object(conversation, "chat_stream", return_value=iter(["hel", "lo ", "world"])):
            deltas = list(process_message_stream(conv, "hi"))
        self.assertEqual(deltas, ["hel", "lo ", "world"])

    def test_builds_full_reply_in_messages(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="fake")
        with mock.patch.object(conversation, "chat_stream", return_value=iter(["one", "two"])):
            list(process_message_stream(conv, "hi"))
        self.assertEqual(conv.messages[-1], {"role": "assistant", "content": "onetwo"})
        self.assertEqual(conv.msg_count, 1)

    def test_first_message_triggers_search(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="fake")
        with (
            mock.patch.object(conversation, "chat_stream", return_value=iter(["ok"])),
            mock.patch.object(conversation, "_search_relevant_context", return_value="ctx") as search,
        ):
            list(process_message_stream(conv, "hello"))
        search.assert_called_once_with("hello")
        self.assertEqual(conv.search_context, "ctx")


class EscalationFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        conversation._PROVIDER_ERRORS = (
            sys.modules["anthropic"].APIStatusError,
            sys.modules["anthropic"].APIConnectionError,
            sys.modules["anthropic"].APITimeoutError,
            sys.modules["openai"].APIStatusError,
            sys.modules["openai"].APIConnectionError,
            sys.modules["openai"].APITimeoutError,
        )

    def _api_error(self, cls_name: str = "RateLimitError"):
        _anthropic = sys.modules["anthropic"]
        return getattr(_anthropic, cls_name)("error")

    def test_rate_limit_falls_back_to_default(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("claude", "sonnet")),
            mock.patch.object(
                conversation, "chat",
                side_effect=[self._api_error("RateLimitError"), "fallback reply"],
            ) as chat_mock,
        ):
            reply = process_message(conv, "add task buy milk")
        self.assertEqual(reply, "fallback reply")
        self.assertEqual(chat_mock.call_count, 2)
        args = chat_mock.call_args_list[1][0]
        self.assertEqual(args[1], "ollama")
        self.assertEqual(args[2], "llama3.1:8b")

    def test_bad_request_does_not_fall_back(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("claude", "sonnet")),
            mock.patch.object(
                conversation, "chat",
                side_effect=self._api_error("BadRequestError"),
            ),
        ):
            _anthropic = sys.modules["anthropic"]
            with self.assertRaises(_anthropic.APIStatusError):
                process_message(conv, "what's the weather")

    def test_api_error_reraises_when_not_escalated(self) -> None:
        _anthropic = sys.modules["anthropic"]
        conv = Conversation(id="test", provider="claude", model="sonnet")
        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("claude", "sonnet")),
            mock.patch.object(
                conversation, "chat",
                side_effect=self._api_error("BadRequestError"),
            ),
        ):
            with self.assertRaises(_anthropic.APIStatusError):
                process_message(conv, "hello")

    def test_stream_rate_limit_falls_back(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        err = self._api_error("RateLimitError")

        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("claude", "sonnet")),
            mock.patch.object(conversation, "chat", side_effect=err),
            mock.patch.object(conversation, "chat_stream", return_value=iter(["fall", "back"])),
        ):
            deltas = list(process_message_stream(conv, "what's the weather"))
        self.assertEqual(deltas, ["fall", "back"])
        self.assertEqual(conv.messages[-1], {"role": "assistant", "content": "fallback"})

    def test_escalated_uses_chat_not_stream(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")

        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("claude", "sonnet")),
            mock.patch.object(conversation, "chat", return_value="buffered response") as mock_chat,
            mock.patch.object(conversation, "chat_stream") as mock_stream,
        ):
            deltas = list(process_message_stream(conv, "what's the weather"))

        mock_chat.assert_called_once()
        mock_stream.assert_not_called()
        self.assertEqual(deltas, ["buffered response"])
        self.assertEqual(conv.messages[-1]["content"], "buffered response")

    def test_stream_fallback_yields_single_response(self) -> None:
        """Escalated failure falls back cleanly — no partial output before fallback."""
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        err = self._api_error("RateLimitError")

        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("claude", "sonnet")),
            mock.patch.object(conversation, "chat", side_effect=err),
            mock.patch.object(conversation, "chat_stream", return_value=iter(["fall", "back"])),
        ):
            deltas = list(process_message_stream(conv, "what's the weather"))

        self.assertEqual(deltas, ["fall", "back"])
        self.assertNotIn("buffered response", deltas)

    def test_non_escalated_uses_stream_not_chat(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")

        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("ollama", "llama3.1:8b")),
            mock.patch.object(conversation, "chat") as mock_chat,
            mock.patch.object(conversation, "chat_stream", return_value=iter(["streamed"])) as mock_stream,
        ):
            deltas = list(process_message_stream(conv, "hello"))

        mock_stream.assert_called_once()
        mock_chat.assert_not_called()
        self.assertEqual(deltas, ["streamed"])

    def test_stream_bad_request_does_not_fall_back(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        err = self._api_error("BadRequestError")

        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("claude", "sonnet")),
            mock.patch.object(conversation, "chat", side_effect=err),
        ):
            _anthropic = sys.modules["anthropic"]
            with self.assertRaises(_anthropic.APIStatusError):
                list(process_message_stream(conv, "what's the weather"))


class SanitizeFactTests(unittest.TestCase):
    def test_strips_newlines(self) -> None:
        result = conversation._sanitize_fact("line one\nline two\nline three")
        self.assertNotIn("\n", result)
        self.assertIn("line one line two line three", result)

    def test_strips_carriage_returns(self) -> None:
        result = conversation._sanitize_fact("line\rone\r\ntwo")
        self.assertNotIn("\r", result)

    def test_strips_html_comments(self) -> None:
        result = conversation._sanitize_fact("fact <!-- injected --> content")
        self.assertNotIn("<!--", result)
        self.assertNotIn("-->", result)
        self.assertIn("fact", result)
        self.assertIn("content", result)

    def test_truncates_long_facts(self) -> None:
        long_fact = "x" * 300
        result = conversation._sanitize_fact(long_fact)
        self.assertEqual(len(result), conversation._MAX_FACT_LENGTH)

    def test_empty_input(self) -> None:
        self.assertEqual(conversation._sanitize_fact(""), "")
        self.assertEqual(conversation._sanitize_fact("   "), "")

    def test_normal_fact_unchanged(self) -> None:
        result = conversation._sanitize_fact("user prefers dark mode")
        self.assertEqual(result, "user prefers dark mode")


class SaveSessionTests(unittest.TestCase):
    def test_saves_final_summary(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="fake")
        conv.messages = [{"role": "user", "content": "hi"}]
        conv.msg_count = 1
        with (
            mock.patch.object(conversation, "_summarize_session", return_value="final") as summarize,
            mock.patch.object(conversation, "_save_session") as save,
        ):
            save_session(conv, Path("/tmp/session.md"))
        summarize.assert_called_once()
        save.assert_called_once()
        self.assertFalse(save.call_args.kwargs.get("is_compaction", False))

    def test_skips_when_no_new_messages(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="fake")
        conv.msg_count = 5
        conv.last_compaction = 5
        with mock.patch.object(conversation, "_summarize_session") as summarize:
            save_session(conv, Path("/tmp/session.md"))
        summarize.assert_not_called()

    def test_skips_when_no_session_file(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="fake")
        conv.messages = [{"role": "user", "content": "hi"}]
        conv.msg_count = 1
        with mock.patch.object(conversation, "_summarize_session") as summarize:
            save_session(conv, None)
        summarize.assert_not_called()


class OpenAIEscalationFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        conversation._PROVIDER_ERRORS = (
            sys.modules["anthropic"].APIStatusError,
            sys.modules["anthropic"].APIConnectionError,
            sys.modules["anthropic"].APITimeoutError,
            sys.modules["openai"].APIStatusError,
            sys.modules["openai"].APIConnectionError,
            sys.modules["openai"].APITimeoutError,
        )

    def _oai_error(self, status_code: int):
        cls = type("OAIStatusError", (sys.modules["openai"].APIStatusError,), {"status_code": status_code})
        return cls("error")

    def _oai_connection_error(self):
        return sys.modules["openai"].APIConnectionError("error")

    def _oai_timeout_error(self):
        return sys.modules["openai"].APITimeoutError("error")

    def test_openai_rate_limit_falls_back(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("openai", "qwen3.5")),
            mock.patch.object(
                conversation, "chat",
                side_effect=[self._oai_error(429), "fallback reply"],
            ) as chat_mock,
        ):
            reply = process_message(conv, "hello")
        self.assertEqual(reply, "fallback reply")
        self.assertEqual(chat_mock.call_count, 2)

    def test_openai_connection_error_falls_back(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("openai", "qwen3.5")),
            mock.patch.object(
                conversation, "chat",
                side_effect=[self._oai_connection_error(), "fallback reply"],
            ),
        ):
            reply = process_message(conv, "hello")
        self.assertEqual(reply, "fallback reply")

    def test_openai_timeout_falls_back(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("openai", "qwen3.5")),
            mock.patch.object(
                conversation, "chat",
                side_effect=[self._oai_timeout_error(), "fallback reply"],
            ),
        ):
            reply = process_message(conv, "hello")
        self.assertEqual(reply, "fallback reply")

    def test_openai_bad_request_does_not_fall_back(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("openai", "qwen3.5")),
            mock.patch.object(
                conversation, "chat",
                side_effect=self._oai_error(400),
            ),
        ):
            _openai = sys.modules["openai"]
            with self.assertRaises(_openai.APIStatusError):
                process_message(conv, "hello")

    def test_stream_openai_rate_limit_falls_back(self) -> None:
        conv = Conversation(id="test", provider="ollama", model="llama3.1:8b")
        with (
            mock.patch.object(conversation, "route_message", return_value=RouteResult("openai", "qwen3.5")),
            mock.patch.object(conversation, "chat", side_effect=self._oai_error(429)),
            mock.patch.object(conversation, "chat_stream", return_value=iter(["fall", "back"])),
        ):
            deltas = list(process_message_stream(conv, "hello"))
        self.assertEqual(deltas, ["fall", "back"])


class EffectiveSearchContextTests(unittest.TestCase):
    def test_both_present(self) -> None:
        conv = Conversation(id="t", provider="ollama", model="m")
        conv.daily_brief = "[tasks]\nBuy milk"
        conv.search_context = "some search results"
        result = _effective_search_context(conv)
        self.assertIn("[tasks]", result)
        self.assertIn("some search results", result)

    def test_only_brief(self) -> None:
        conv = Conversation(id="t", provider="ollama", model="m")
        conv.daily_brief = "[tasks]\nBuy milk"
        result = _effective_search_context(conv)
        self.assertEqual(result, "[tasks]\nBuy milk")

    def test_only_search(self) -> None:
        conv = Conversation(id="t", provider="ollama", model="m")
        conv.search_context = "search results"
        result = _effective_search_context(conv)
        self.assertEqual(result, "search results")

    def test_neither(self) -> None:
        conv = Conversation(id="t", provider="ollama", model="m")
        result = _effective_search_context(conv)
        self.assertEqual(result, "")


class FetchDailyBriefTests(unittest.TestCase):
    @mock.patch("tars.conversation.build_daily_context", return_value="[tasks]\nStuff")
    def test_stores_on_conv(self, mock_ctx) -> None:
        conv = Conversation(id="t", provider="ollama", model="m")
        _fetch_daily_brief(conv)
        self.assertEqual(conv.daily_brief, "[tasks]\nStuff")

    @mock.patch("tars.conversation.build_daily_context", side_effect=Exception("boom"))
    def test_swallows_errors(self, mock_ctx) -> None:
        conv = Conversation(id="t", provider="ollama", model="m")
        _fetch_daily_brief(conv)
        self.assertEqual(conv.daily_brief, "")

    @mock.patch("tars.conversation.build_daily_context", return_value="[tasks]\nDo things")
    def test_process_message_fetches_on_first(self, mock_ctx) -> None:
        conv = Conversation(id="t", provider="ollama", model="m")
        with (
            mock.patch.object(conversation, "chat", return_value="ok"),
            mock.patch.object(conversation, "_search_relevant_context", return_value=""),
        ):
            process_message(conv, "hello")
        mock_ctx.assert_called_once()
        self.assertEqual(conv.daily_brief, "[tasks]\nDo things")

    @mock.patch("tars.conversation.build_daily_context", return_value="[tasks]\nDo things")
    def test_not_fetched_on_second_message(self, mock_ctx) -> None:
        conv = Conversation(id="t", provider="ollama", model="m")
        with (
            mock.patch.object(conversation, "chat", return_value="ok"),
            mock.patch.object(conversation, "_search_relevant_context", return_value=""),
        ):
            process_message(conv, "hello")
            process_message(conv, "again")
        mock_ctx.assert_called_once()

    @mock.patch("tars.conversation.build_daily_context", return_value="[tasks]\nDo things")
    def test_skipped_when_search_context_preset(self, mock_ctx) -> None:
        conv = Conversation(id="t", provider="ollama", model="m")
        conv.search_context = "[one-shot]"
        with mock.patch.object(conversation, "chat", return_value="ok"):
            process_message(conv, "hello")
        mock_ctx.assert_not_called()


if __name__ == "__main__":
    unittest.main()
