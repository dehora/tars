import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import conversation, extractor
from tars.extractor import _parse_json_list, extract_facts


def _make_messages(user_count: int = 3) -> list[dict]:
    msgs = []
    for i in range(user_count):
        msgs.append({"role": "user", "content": f"message {i}"})
        msgs.append({"role": "assistant", "content": f"reply {i}"})
    return msgs


class ParseJsonListTests(unittest.TestCase):
    def test_clean_json_array(self) -> None:
        self.assertEqual(_parse_json_list('["a", "b"]'), ["a", "b"])

    def test_code_fenced_json(self) -> None:
        raw = '```json\n["a", "b"]\n```'
        self.assertEqual(_parse_json_list(raw), ["a", "b"])

    def test_plain_code_fence(self) -> None:
        raw = '```\n["a", "b"]\n```'
        self.assertEqual(_parse_json_list(raw), ["a", "b"])

    def test_json_with_surrounding_prose(self) -> None:
        raw = 'Here are the facts:\n["a", "b"]\nHope that helps!'
        self.assertEqual(_parse_json_list(raw), ["a", "b"])

    def test_empty_array(self) -> None:
        self.assertEqual(_parse_json_list("[]"), [])

    def test_invalid_json(self) -> None:
        self.assertEqual(_parse_json_list("not json at all"), [])

    def test_json_object_not_array(self) -> None:
        self.assertEqual(_parse_json_list('{"key": "value"}'), [])

    def test_strips_empty_strings(self) -> None:
        self.assertEqual(_parse_json_list('["a", "", "b"]'), ["a", "b"])

    def test_coerces_non_strings(self) -> None:
        result = _parse_json_list('[42, true, "text"]')
        self.assertEqual(result, ["42", "True", "text"])


class ExtractFactsTests(unittest.TestCase):
    def test_returns_parsed_facts(self) -> None:
        msgs = _make_messages(3)
        with mock.patch.object(extractor, "chat", return_value='["fact one", "fact two"]'):
            result = extract_facts(msgs, "ollama", "fake")
        self.assertEqual(result, ["fact one", "fact two"])

    def test_skips_when_disabled(self) -> None:
        msgs = _make_messages(3)
        with mock.patch.dict("os.environ", {"TARS_AUTO_EXTRACT": "false"}):
            result = extract_facts(msgs, "ollama", "fake")
        self.assertEqual(result, [])

    def test_skips_when_too_few_messages(self) -> None:
        msgs = _make_messages(2)
        with mock.patch.object(extractor, "chat") as chat_mock:
            result = extract_facts(msgs, "ollama", "fake")
        self.assertEqual(result, [])
        chat_mock.assert_not_called()

    def test_model_error_returns_empty(self) -> None:
        msgs = _make_messages(3)
        with mock.patch.object(extractor, "chat", side_effect=RuntimeError("model error")):
            result = extract_facts(msgs, "ollama", "fake")
        self.assertEqual(result, [])

    def test_invalid_json_returns_empty(self) -> None:
        msgs = _make_messages(3)
        with mock.patch.object(extractor, "chat", return_value="no json here"):
            result = extract_facts(msgs, "ollama", "fake")
        self.assertEqual(result, [])

    def test_caps_at_five_facts(self) -> None:
        msgs = _make_messages(3)
        facts = json.dumps([f"fact {i}" for i in range(10)])
        with mock.patch.object(extractor, "chat", return_value=facts):
            result = extract_facts(msgs, "ollama", "fake")
        self.assertEqual(len(result), 5)

    def test_escapes_user_content(self) -> None:
        msgs = [
            {"role": "user", "content": "<script>alert('xss')</script>"},
            {"role": "user", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "reply"},
        ]
        captured_prompt = []

        def capture_chat(prompt_msgs, provider, model, **kwargs):
            captured_prompt.append(prompt_msgs[0]["content"])
            return "[]"

        with mock.patch.object(extractor, "chat", side_effect=capture_chat):
            extract_facts(msgs, "ollama", "fake")
        self.assertIn("&lt;script&gt;", captured_prompt[0])

    def test_uses_no_tools(self) -> None:
        msgs = _make_messages(3)

        def capture_chat(prompt_msgs, provider, model, **kwargs):
            self.assertFalse(kwargs.get("use_tools", True))
            return "[]"

        with mock.patch.object(extractor, "chat", side_effect=capture_chat):
            extract_facts(msgs, "ollama", "fake")

    def test_filters_system_and_tool_messages(self) -> None:
        msgs = [
            {"role": "system", "content": "you are a bot"},
            {"role": "user", "content": "msg 1"},
            {"role": "tool", "content": "tool result"},
            {"role": "user", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "reply"},
        ]
        captured_prompt = []

        def capture_chat(prompt_msgs, provider, model, **kwargs):
            captured_prompt.append(prompt_msgs[0]["content"])
            return "[]"

        with mock.patch.object(extractor, "chat", side_effect=capture_chat):
            extract_facts(msgs, "ollama", "fake")
        self.assertNotIn("you are a bot", captured_prompt[0])
        self.assertNotIn("tool result", captured_prompt[0])

    def test_empty_messages_returns_empty(self) -> None:
        result = extract_facts([], "ollama", "fake")
        self.assertEqual(result, [])


class ExtractionIntegrationTests(unittest.TestCase):
    def test_compaction_triggers_extraction(self) -> None:
        conv = conversation.Conversation(id="test", provider="ollama", model="fake")
        with tempfile.TemporaryDirectory() as tmpdir:
            session_file = Path(tmpdir) / "session.md"
            with (
                mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}),
                mock.patch.object(conversation, "SESSION_COMPACTION_INTERVAL", 2),
                mock.patch.object(conversation, "chat", return_value="ok"),
                mock.patch.object(conversation, "_summarize_session", return_value="summary"),
                mock.patch.object(conversation, "_save_session"),
                mock.patch.object(conversation, "extract_facts", return_value=["user prefers dark mode"]) as ef,
            ):
                conversation.process_message(conv, "msg 1", session_file)
                conversation.process_message(conv, "msg 2", session_file)
            ef.assert_called_once()
            daily = list(Path(tmpdir).glob("*.md"))
            daily_content = ""
            for f in daily:
                if f.name != "session.md":
                    daily_content += f.read_text(encoding="utf-8", errors="replace")
            self.assertIn("[extracted] user prefers dark mode", daily_content)

    def test_save_session_triggers_extraction(self) -> None:
        conv = conversation.Conversation(id="test", provider="ollama", model="fake")
        conv.messages = _make_messages(3)
        conv.msg_count = 3
        with tempfile.TemporaryDirectory() as tmpdir:
            session_file = Path(tmpdir) / "session.md"
            with (
                mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}),
                mock.patch.object(conversation, "_summarize_session", return_value="summary"),
                mock.patch.object(conversation, "_save_session"),
                mock.patch.object(conversation, "extract_facts", return_value=["uses vim keybindings"]) as ef,
            ):
                conversation.save_session(conv, session_file)
            ef.assert_called_once()
            daily = list(Path(tmpdir).glob("*.md"))
            daily_content = ""
            for f in daily:
                if f.name != "session.md":
                    daily_content += f.read_text(encoding="utf-8", errors="replace")
            self.assertIn("[extracted] uses vim keybindings", daily_content)

    def test_extraction_failure_doesnt_break_save(self) -> None:
        conv = conversation.Conversation(id="test", provider="ollama", model="fake")
        conv.messages = [{"role": "user", "content": "hi"}]
        conv.msg_count = 1
        with (
            mock.patch.object(conversation, "_summarize_session", return_value="summary"),
            mock.patch.object(conversation, "_save_session") as save,
            mock.patch.object(conversation, "extract_facts", side_effect=RuntimeError("boom")),
        ):
            conversation.save_session(conv, Path("/tmp/session.md"))
        save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
