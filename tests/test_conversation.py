import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import conversation
from tars.conversation import Conversation, process_message, process_message_stream, save_session


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


if __name__ == "__main__":
    unittest.main()
